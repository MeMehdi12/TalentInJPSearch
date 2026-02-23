"""
Hybrid Search Service - Qdrant Dense + Sparse with RRF Fusion
==============================================================
Main search orchestrator for the talent search system.

Features:
- Dense (semantic) + Sparse (keyword) hybrid search
- Reciprocal Rank Fusion (RRF) for merging results
- Skill expansion from relationships
- Metadata fetching from DuckDB
- Bonus re-ranking for final scoring

Performance target: <150ms total latency
"""

import time
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
import duckdb
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, 
    Prefetch, 
    FusionQuery, 
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)
from sentence_transformers import SentenceTransformer

from config import get_config
from sparse_encoder import get_sparse_encoder
from search_schema import (
    HybridSearchQuery, 
    SearchResponse, 
    CandidateResult, 
    RankingFactors
)
from ranking_service import RankingService


class HybridSearchService:
    """
    Production-ready hybrid search service.
    
    Flow:
    1. Encode query → dense embedding + sparse vector
    2. Qdrant prefetch (dense 300 + sparse 300)
    3. RRF fusion → top 100
    4. Fetch full metadata from DuckDB
    5. Apply ranking bonuses
    6. Return top N results
    """
    
    def __init__(self):
        self.config = get_config()
        self._qdrant: Optional[QdrantClient] = None
        self._model: Optional[SentenceTransformer] = None
        self._sparse_encoder = None
        self._skill_relations: Optional[Dict[str, List[Tuple[str, float]]]] = None
        self._ranking_service = RankingService()
    
    @property
    def qdrant(self) -> QdrantClient:
        """Lazy load Qdrant client"""
        if self._qdrant is None:
            # DEBUG: Check API Key
            api_key = self.config.qdrant_api_key
            
            # If cloud mode and no key, force correct it
            if self.config.is_cloud and not api_key:
                logger.warning("HybridSearch: API Key missing in config! Force loading .env...")
                import os
                from dotenv import load_dotenv
                load_dotenv(override=True)
                api_key = os.getenv("QDRANT_API_KEY")

            if self.config.is_cloud:
                if api_key:
                    logger.info("HybridSearch: Connecting to Qdrant Cloud (API key configured)")
                else:
                    logger.error("HybridSearch: CRITICAL - NO API KEY for Cloud Qdrant!")

                self._qdrant = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=api_key,
                    timeout=30
                )
            else:
                self._qdrant = QdrantClient(path=self.config.qdrant_url)
        return self._qdrant
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model
    
    @property
    def sparse_encoder(self):
        """Lazy load sparse encoder"""
        if self._sparse_encoder is None:
            self._sparse_encoder = get_sparse_encoder()
        return self._sparse_encoder
    
    def get_db_connection(self) -> duckdb.DuckDBPyConnection:
        """Get DuckDB connection"""
        return duckdb.connect(self.config.duckdb_path, read_only=True)
    
    # =========================================================================
    # SKILL EXPANSION
    # =========================================================================
    
    def _load_skill_relations(self) -> Dict[str, List[Tuple[str, float]]]:
        """Load skill relationships from DuckDB"""
        if self._skill_relations is not None:
            return self._skill_relations
        
        conn = self.get_db_connection()
        try:
            relations = conn.execute("""
                SELECT skill_a, skill_b, 
                       COALESCE(semantic_confidence, 0.5) as confidence
                FROM skill_relationships
                WHERE COALESCE(semantic_confidence, 0.5) >= 0.5
                ORDER BY confidence DESC
            """).fetchall()
            
            self._skill_relations = {}
            for skill_a, skill_b, score in relations:
                a_lower = skill_a.lower()
                b_lower = skill_b.lower()
                
                if a_lower not in self._skill_relations:
                    self._skill_relations[a_lower] = []
                if b_lower not in self._skill_relations:
                    self._skill_relations[b_lower] = []
                
                self._skill_relations[a_lower].append((skill_b, score))
                self._skill_relations[b_lower].append((skill_a, score))
            
            return self._skill_relations
        finally:
            conn.close()
    
    def expand_skills(self, skills: List[str], max_per_skill: int = 5) -> Tuple[List[str], Dict[str, List[str]]]:
        """Expand skills to include related skills"""
        relations = self._load_skill_relations()
        
        expanded = set()
        expansion_map = {}
        
        for skill in skills:
            skill_lower = skill.lower()
            expanded.add(skill_lower)
            
            if skill_lower in relations:
                related = relations[skill_lower][:max_per_skill]
                related_skills = [r[0].lower() for r in related]
                expanded.update(related_skills)
                expansion_map[skill_lower] = related_skills
        
        return list(expanded), expansion_map
    
    # =========================================================================
    # QUERY ENCODING
    # =========================================================================
    
    def encode_dense(self, text: str) -> List[float]:
        """Encode text to dense embedding"""
        return self.model.encode(text).tolist()
    
    def encode_sparse(self, keywords: List[str]) -> Tuple[List[int], List[float]]:
        """Encode keywords to sparse vector with skill expansion"""
        relations = self._load_skill_relations()
        return self.sparse_encoder.encode_with_expansion(keywords, relations)
    
    # =========================================================================
    # MAIN SEARCH
    # =========================================================================
    
    def search(self, query: HybridSearchQuery) -> SearchResponse:
        """
        Execute hybrid search.
        
        Args:
            query: HybridSearchQuery with search_text, keywords, boosts, etc.
        
        Returns:
            SearchResponse with ranked candidates
        """
        start_time = time.time()
        timings = {}
        
        # 1. Prepare dense query
        t0 = time.time()
        search_text = query.search_text or "professional"
        dense_embedding = self.encode_dense(search_text)
        timings['dense_encode'] = int((time.time() - t0) * 1000)
        
        # 2. Prepare sparse query
        t0 = time.time()
        keywords = list(query.keywords) if query.keywords else []
        
        # Add boost keywords to sparse search
        if query.boost_skills:
            keywords.extend(query.boost_skills)
        if query.boost_companies:
            keywords.extend(query.boost_companies)
        if query.boost_schools:
            keywords.extend(query.boost_schools)
        if query.boost_titles:
            keywords.extend(query.boost_titles)
        if query.boost_location and query.boost_location.city:
            keywords.append(query.boost_location.city)
        
        sparse_indices, sparse_values = self.encode_sparse(keywords) if keywords else ([], [])
        timings['sparse_encode'] = int((time.time() - t0) * 1000)
        
        # 3. Build filter for hard excludes
        qdrant_filter = self._build_filter(query)
        
        # 4. Execute hybrid search
        t0 = time.time()
        
        prefetch_limit = 300
        fusion_limit = min(query.limit * 2, 200)  # Over-fetch for ranking
        
        try:
            if sparse_indices:
                # Full hybrid search with RRF
                results = self.qdrant.query_points(
                    collection_name=self.config.qdrant_collection,
                    prefetch=[
                        Prefetch(
                            query=dense_embedding,
                            using="dense",
                            limit=prefetch_limit
                        ),
                        Prefetch(
                            query=SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            ),
                            using="sparse",
                            limit=prefetch_limit
                        )
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    query_filter=qdrant_filter,
                    limit=fusion_limit,
                    with_payload=True
                )
            else:
                # Dense-only search (no keywords)
                results = self.qdrant.query_points(
                    collection_name=self.config.qdrant_collection,
                    query=dense_embedding,
                    using="dense",
                    query_filter=qdrant_filter,
                    limit=fusion_limit,
                    with_payload=True
                )
        except Exception as e:
            # Fallback to simple dense search
            print(f"Hybrid search failed, falling back to dense: {e}")
            results = self.qdrant.query_points(
                collection_name=self.config.qdrant_collection,
                query=dense_embedding,
                using="dense",
                query_filter=qdrant_filter,
                limit=fusion_limit,
                with_payload=True
            )
        
        timings['qdrant_search'] = int((time.time() - t0) * 1000)
        
        # Extract points
        points = results.points if hasattr(results, 'points') else results
        
        if not points:
            return SearchResponse(
                total_matches=0,
                returned=0,
                took_ms=int((time.time() - start_time) * 1000),
                results=[],
                timings=timings
            )
        
        # 5. Get forager_ids and base scores
        forager_ids = [p.payload.get("forager_id", p.id) for p in points]
        base_scores = {p.payload.get("forager_id", p.id): p.score for p in points}
        
        # 6. Fetch full metadata from DuckDB
        t0 = time.time()
        metadata = self._fetch_metadata(forager_ids)
        timings['duckdb_fetch'] = int((time.time() - t0) * 1000)
        
        # 7. Apply ranking bonuses
        t0 = time.time()
        ranked_results = self._ranking_service.rank(
            candidates=metadata,
            base_scores=base_scores,
            query=query
        )
        timings['ranking'] = int((time.time() - t0) * 1000)
        
        # 8. Build response
        final_results = ranked_results[:query.limit]
        
        total_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            total_matches=len(points),
            returned=len(final_results),
            took_ms=total_ms,
            results=final_results,
            timings=timings
        )
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _build_filter(self, query: HybridSearchQuery) -> Optional[Filter]:
        """Build Qdrant filter for hard excludes"""
        must_not = []
        must = []
        
        # Experience range (hard filter if specified)
        if query.experience_range:
            if query.experience_range.min_years is not None:
                must.append(
                    FieldCondition(
                        key="years_experience",
                        range=Range(gte=float(query.experience_range.min_years))
                    )
                )
            if query.experience_range.max_years is not None:
                must.append(
                    FieldCondition(
                        key="years_experience",
                        range=Range(lte=float(query.experience_range.max_years))
                    )
                )
        
        if not must and not must_not:
            return None
        
        return Filter(
            must=must if must else None,
            must_not=must_not if must_not else None
        )
    
    def _fetch_metadata(self, forager_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch full profile metadata from DuckDB"""
        if not forager_ids:
            return {}
        
        conn = self.get_db_connection()
        try:
            # Main profile data
            placeholders = ','.join(['?' for _ in forager_ids])
            
            profiles = conn.execute(f"""
                SELECT 
                    person_id,
                    full_name,
                    canonical_city,
                    canonical_state,
                    canonical_country,
                    primary_domain,
                    canonical_skills,
                    years_experience,
                    current_role_title,
                    current_role_company,
                    profile_completeness
                FROM processed_profiles
                WHERE person_id IN ({placeholders})
            """, forager_ids).fetchall()
            
            # Build metadata dict
            metadata = {}
            for row in profiles:
                pid = row[0]
                metadata[pid] = {
                    'person_id': pid,
                    'full_name': row[1],
                    'city': row[2],
                    'state': row[3],
                    'country': row[4],
                    'domain': row[5],
                    'skills': list(row[6]) if row[6] else [],
                    'years_experience': row[7],
                    'current_title': row[8],
                    'current_company': row[9],
                    'profile_completeness': row[10],
                    'companies': [],
                    'schools': [],
                    'certifications': []
                }
            
            # Fetch companies (roles)
            companies = conn.execute(f"""
                SELECT forager_id, organization_name, role_title
                FROM roles
                WHERE forager_id IN ({placeholders})
                  AND organization_name IS NOT NULL
            """, forager_ids).fetchall()
            
            for fid, company, title in companies:
                if fid in metadata:
                    metadata[fid]['companies'].append(company.lower() if company else '')
            
            # Fetch schools (educations)
            schools = conn.execute(f"""
                SELECT forager_id, school_name, degree, field_of_study
                FROM educations
                WHERE forager_id IN ({placeholders})
                  AND school_name IS NOT NULL
            """, forager_ids).fetchall()
            
            for fid, school, degree, field in schools:
                if fid in metadata:
                    metadata[fid]['schools'].append(school.lower() if school else '')
            
            # Fetch certifications
            certs = conn.execute(f"""
                SELECT forager_id, certificate_name
                FROM certifications
                WHERE forager_id IN ({placeholders})
                  AND certificate_name IS NOT NULL
            """, forager_ids).fetchall()
            
            for fid, cert in certs:
                if fid in metadata:
                    metadata[fid]['certifications'].append(cert.lower() if cert else '')
            
            return metadata
            
        finally:
            conn.close()
    
    # =========================================================================
    # ADDITIONAL METHODS
    # =========================================================================
    
    def get_profile_detail(self, forager_id: str) -> Optional[Dict]:
        """Get full profile details by ID"""
        conn = self.get_db_connection()
        try:
            # Get main profile
            profile = conn.execute("""
                SELECT 
                    forager_id, full_name, city_name, state_name, country_name,
                    headline, summary, profile_text, total_experience_years
                FROM processed_profiles
                WHERE forager_id = ?
            """, [forager_id]).fetchone()
            
            if not profile:
                return None
            
            result = {
                "forager_id": profile[0],
                "full_name": profile[1],
                "location": {
                    "city": profile[2],
                    "state": profile[3],
                    "country": profile[4],
                },
                "headline": profile[5],
                "summary": profile[6],
                "total_experience_years": profile[8],
                "roles": [],
                "education": [],
                "certifications": [],
                "skills": [],
            }
            
            # Get roles
            roles = conn.execute("""
                SELECT company_name, title, start_date, end_date, description
                FROM roles
                WHERE forager_id = ?
                ORDER BY start_date DESC
            """, [forager_id]).fetchall()
            
            for r in roles:
                result["roles"].append({
                    "company": r[0],
                    "title": r[1],
                    "start_date": r[2],
                    "end_date": r[3],
                    "description": r[4],
                })
            
            # Get education
            edu = conn.execute("""
                SELECT school_name, degree, field_of_study, start_date, end_date
                FROM educations
                WHERE forager_id = ?
                ORDER BY start_date DESC
            """, [forager_id]).fetchall()
            
            for e in edu:
                result["education"].append({
                    "school": e[0],
                    "degree": e[1],
                    "field": e[2],
                    "start_date": e[3],
                    "end_date": e[4],
                })
            
            # Get certifications
            certs = conn.execute("""
                SELECT certificate_name, authority, license_number
                FROM certifications
                WHERE forager_id = ?
            """, [forager_id]).fetchall()
            
            for c in certs:
                result["certifications"].append({
                    "name": c[0],
                    "authority": c[1],
                    "license_number": c[2],
                })
            
            # Get skills from profile text (or skills table if exists)
            try:
                skills = conn.execute("""
                    SELECT skill_name
                    FROM profile_skills
                    WHERE forager_id = ?
                """, [forager_id]).fetchall()
                result["skills"] = [s[0] for s in skills]
            except Exception as e:
                logger.debug(f"Skills table query skipped: {e}")  # Table might not exist
            
            return result
            
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "collection_name": self.config.qdrant_collection,
            "embedding_model": self.config.embedding_model,
        }
        
        # Get Qdrant collection info
        try:
            collection_info = self.qdrant.get_collection(self.config.qdrant_collection)
            stats["qdrant"] = {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
            }
        except Exception as e:
            stats["qdrant"] = {"error": str(e)}
        
        # Get DuckDB stats
        conn = self.get_db_connection()
        try:
            profile_count = conn.execute(
                "SELECT COUNT(*) FROM processed_profiles"
            ).fetchone()[0]
            stats["duckdb"] = {
                "profile_count": profile_count,
            }
        except Exception as e:
            stats["duckdb"] = {"error": str(e)}
        finally:
            conn.close()
        
        return stats
    
    def close(self):
        """Close connections"""
        if self._qdrant:
            self._qdrant.close()


# Global instance
_service: Optional[HybridSearchService] = None


def get_hybrid_search_service() -> HybridSearchService:
    """Get singleton search service"""
    global _service
    if _service is None:
        _service = HybridSearchService()
    return _service
