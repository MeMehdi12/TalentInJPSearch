"""
Search Service - Core search logic with skill expansion and intelligent ranking
This is the brain of the search system.
"""
import time
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import duckdb
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
from sentence_transformers import SentenceTransformer

from config import get_config
from search_schema import (
    ParsedSearchQuery, SearchResponse, CandidateResult,
    SkillFilters, LocationFilter, ExperienceFilter
)


class SearchService:
    """
    Production-ready search service with:
    - Skill expansion (python -> django, flask, etc.)
    - Hybrid search (vector + filters)
    - Intelligent ranking
    - DuckDB hydration for full profiles
    """
    
    def __init__(self):
        self.config = get_config()
        self._qdrant: Optional[QdrantClient] = None
        self._model: Optional[SentenceTransformer] = None
        self._skill_relations: Optional[Dict[str, List[Tuple[str, float]]]] = None
        
    @property
    def qdrant(self) -> QdrantClient:
        """Lazy load Qdrant client"""
        if self._qdrant is None:
            if self.config.qdrant_api_key:
                # Production: Qdrant Cloud
                self._qdrant = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                # Local: File-based Qdrant
                self._qdrant = QdrantClient(path=self.config.qdrant_url)
        return self._qdrant
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._model is None:
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model
    
    def get_db_connection(self) -> duckdb.DuckDBPyConnection:
        """Get DuckDB connection"""
        return duckdb.connect(self.config.duckdb_path, read_only=True)
    
    # =========================================================================
    # SKILL EXPANSION - The secret sauce
    # =========================================================================
    
    def _load_skill_relations(self) -> Dict[str, List[Tuple[str, float]]]:
        """Load skill relationships from DuckDB (cached)"""
        if self._skill_relations is not None:
            return self._skill_relations
        
        conn = self.get_db_connection()
        try:
            # Get all skill relationships
            relations = conn.execute("""
                SELECT skill_a, skill_b, relationship_type, confidence_score
                FROM skill_relationships
                WHERE confidence_score >= 0.5
                ORDER BY confidence_score DESC
            """).fetchall()
            
            # Build bidirectional mapping
            self._skill_relations = {}
            for skill_a, skill_b, rel_type, score in relations:
                skill_a_lower = skill_a.lower()
                skill_b_lower = skill_b.lower()
                
                if skill_a_lower not in self._skill_relations:
                    self._skill_relations[skill_a_lower] = []
                if skill_b_lower not in self._skill_relations:
                    self._skill_relations[skill_b_lower] = []
                
                # Add relationship (weighted by score)
                self._skill_relations[skill_a_lower].append((skill_b_lower, score))
                self._skill_relations[skill_b_lower].append((skill_a_lower, score))
            
            return self._skill_relations
            
        finally:
            conn.close()
    
    def expand_skills(self, skills: List[str], max_per_skill: int = 5) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Expand skills to include related skills.
        
        Example: "python" -> ["python", "django", "flask", "fastapi", "pandas"]
        
        Returns:
            expanded_skills: Full list of skills to search for
            expansion_map: Mapping of original skill -> expanded skills (for debugging)
        """
        relations = self._load_skill_relations()
        
        expanded = set()
        expansion_map = {}
        
        for skill in skills:
            skill_lower = skill.lower()
            expanded.add(skill_lower)
            
            # Get related skills
            if skill_lower in relations:
                related = relations[skill_lower]
                # Sort by confidence score and take top N
                related_sorted = sorted(related, key=lambda x: x[1], reverse=True)[:max_per_skill]
                related_skills = [r[0] for r in related_sorted]
                
                expanded.update(related_skills)
                expansion_map[skill_lower] = related_skills
        
        return list(expanded), expansion_map
    
    # =========================================================================
    # FILTER BUILDING
    # =========================================================================
    
    def _build_qdrant_filter(
        self, 
        skills: SkillFilters,
        experience: ExperienceFilter,
        location: LocationFilter,
        domain: Optional[str],
        expanded_must_have: List[str]
    ) -> Optional[Filter]:
        """Build Qdrant filter from parsed query"""
        
        must_conditions = []
        must_not_conditions = []
        
        # Skills - use expanded skills with MatchAny (OR logic within expanded set)
        if expanded_must_have:
            must_conditions.append(
                FieldCondition(
                    key="skills",
                    match=MatchAny(any=expanded_must_have)
                )
            )
        
        # Excluded skills
        for skill in skills.exclude:
            must_not_conditions.append(
                FieldCondition(
                    key="skills",
                    match=MatchValue(value=skill.lower())
                )
            )
        
        # Experience range
        if experience.min_years is not None:
            must_conditions.append(
                FieldCondition(
                    key="years_experience",
                    range=Range(gte=float(experience.min_years))
                )
            )
        
        if experience.max_years is not None:
            must_conditions.append(
                FieldCondition(
                    key="years_experience",
                    range=Range(lte=float(experience.max_years))
                )
            )
        
        # Location filters
        if location.city:
            must_conditions.append(
                FieldCondition(key="city", match=MatchValue(value=location.city))
            )
        
        if location.state:
            must_conditions.append(
                FieldCondition(key="state", match=MatchValue(value=location.state))
            )
        
        if location.country:
            must_conditions.append(
                FieldCondition(key="country", match=MatchValue(value=location.country))
            )
        
        # Domain filter
        if domain:
            must_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        
        # Build final filter
        if not must_conditions and not must_not_conditions:
            return None
        
        return Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )
    
    # =========================================================================
    # MAIN SEARCH
    # =========================================================================
    
    def search(self, query: ParsedSearchQuery) -> SearchResponse:
        """
        Execute search with skill expansion and intelligent ranking.
        
        Flow:
        1. Expand must_have skills
        2. Generate query embedding
        3. Search Qdrant with filters
        4. Hydrate from DuckDB
        5. Apply nice_to_have boost
        6. Return ranked results
        """
        start_time = time.time()
        
        # 1. Expand skills (always on - this is core to ranking intelligence)
        expanded_must_have, expansion_map = self.expand_skills(query.filters.skills.must_have)
        expanded_nice_to_have, _ = self.expand_skills(query.filters.skills.nice_to_have)
        
        # 2. Build search text for embedding
        search_parts = []
        if query.search_text:
            search_parts.append(query.search_text)
        if query.filters.job_titles:
            search_parts.extend(query.filters.job_titles)
        
        search_text = " ".join(search_parts) if search_parts else "professional"
        
        # 3. Generate query embedding
        query_embedding = self.model.encode(search_text).tolist()
        
        # 4. Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(
            skills=query.filters.skills,
            experience=query.filters.experience,
            location=query.filters.location,
            domain=query.filters.domain,
            expanded_must_have=expanded_must_have
        )
        
        # 5. Search Qdrant - get more than needed for ranking
        search_limit = min(query.limit * 3, 300)
        
        search_results = self.qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=search_limit,
            with_payload=True
        ).points
        
        if not search_results:
            return SearchResponse(
                total_matches=0,
                returned=0,
                took_ms=int((time.time() - start_time) * 1000),
                results=[],
                expanded_skills=expansion_map if expansion_map else None
            )
        
        # 6. Hydrate from DuckDB
        forager_ids = [hit.payload["forager_id"] for hit in search_results]
        score_map = {hit.payload["forager_id"]: hit.score for hit in search_results}
        
        profiles = self._hydrate_profiles(forager_ids)
        
        # 7. Apply nice_to_have boost and build results
        nice_to_have_set = set(s.lower() for s in expanded_nice_to_have)
        must_have_set = set(s.lower() for s in expanded_must_have)
        
        results = []
        for profile in profiles:
            forager_id = profile["person_id"]
            base_score = score_map.get(forager_id, 0)
            
            # Get profile skills
            profile_skills = profile.get("canonical_skills") or []
            if isinstance(profile_skills, str):
                profile_skills = [profile_skills]
            profile_skills_lower = [s.lower() for s in profile_skills if s]
            
            # Calculate matched skills
            matched_skills = [s for s in profile_skills if s.lower() in must_have_set]
            
            # Nice-to-have boost (up to 10% boost)
            nice_to_have_matches = sum(1 for s in profile_skills_lower if s in nice_to_have_set)
            boost = min(nice_to_have_matches * 0.02, 0.10)
            
            final_score = base_score + boost
            
            results.append(CandidateResult(
                forager_id=forager_id,
                score=round(final_score, 4),
                full_name=profile.get("full_name") or "Unknown",
                current_title=profile.get("current_role_title"),
                current_company=profile.get("current_role_company"),
                city=profile.get("canonical_city"),
                state=profile.get("canonical_state"),
                country=profile.get("canonical_country"),
                years_experience=profile.get("years_experience"),
                domain=profile.get("primary_domain"),
                skills=list(profile_skills)[:15],
                matched_skills=matched_skills,
                profile_completeness=profile.get("profile_completeness"),
                headline=profile.get("headline"),
                linkedin_url=profile.get("linkedin_url"),
            ))
        
        # 8. Sort by final score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:query.limit]
        
        took_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            total_matches=len(search_results),
            returned=len(results),
            took_ms=took_ms,
            results=results,
            expanded_skills=expansion_map if expansion_map else None
        )
    
    def _hydrate_profiles(self, forager_ids: List[int]) -> List[Dict]:
        """Fetch full profile data from DuckDB"""
        if not forager_ids:
            return []
        
        conn = self.get_db_connection()
        try:
            ids_str = ",".join(str(id) for id in forager_ids)
            
            profiles = conn.execute(f"""
                SELECT 
                    pp.person_id,
                    pp.full_name,
                    pp.current_role_title,
                    pp.current_role_company,
                    pp.canonical_city,
                    pp.canonical_state,
                    pp.canonical_country,
                    pp.years_experience,
                    pp.primary_domain,
                    pp.canonical_skills,
                    pp.profile_completeness,
                    p.headline,
                    p.linkedin_profile_url as linkedin_url
                FROM processed_profiles pp
                LEFT JOIN persons p ON pp.person_id = p.forager_id
                WHERE pp.person_id IN ({ids_str})
            """).fetchall()
            
            columns = [
                "person_id", "full_name", "current_role_title", "current_role_company",
                "canonical_city", "canonical_state", "canonical_country", "years_experience",
                "primary_domain", "canonical_skills", "profile_completeness", "headline", "linkedin_url"
            ]
            
            return [dict(zip(columns, row)) for row in profiles]
            
        finally:
            conn.close()


# Global service instance
_search_service: Optional[SearchService] = None

def get_search_service() -> SearchService:
    """Get the global search service instance"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
