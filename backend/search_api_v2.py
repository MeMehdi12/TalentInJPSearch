"""
Search API v2 - Clean implementation for external LLM integration
=================================================================

Flow:
1. Receive parsed query JSON from external LLM API
2. Normalize inputs (aliases, abbreviations, case)
3. Expand skills using skill_relationships table
4. Query Qdrant (dense + sparse hybrid)
5. Hydrate results from DuckDB
6. Apply ranking bonuses
7. Return formatted results

Endpoints:
  POST /api/v2/search - Main search endpoint
  GET  /api/v2/health - Health check
  GET  /api/v2/stats  - Database statistics
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import time
import logging
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
from contextlib import asynccontextmanager

import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    SparseVector, Prefetch, FusionQuery, Fusion
)

from config import get_config, Config
from search_schema import (
    ParsedQueryV2, SearchResponseV2, CandidateResultV2, 
    LocationFacet, SearchFacets, CityCount
)
from normalizers import (
    normalize_company, normalize_city, normalize_state, normalize_country,
    normalize_skill, expand_company_search, expand_skill_search,
    normalize_text, clean_list
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL STATE (Thread-safe initialization)
# =============================================================================
import threading

_embedding_model: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None
_sparse_encoder = None
_skill_relations: Optional[Dict[str, List[Tuple[str, float]]]] = None

# Thread locks for safe initialization
_model_lock = threading.Lock()
_qdrant_lock = threading.Lock()
_relations_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Lazy load embedding model (thread-safe)"""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:  # Double-check pattern
                config = get_config()
                logger.info(f"Loading embedding model: {config.embedding_model}")
                _embedding_model = SentenceTransformer(config.embedding_model)
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """Lazy load Qdrant client (thread-safe)"""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:  # Double-check pattern
                config = get_config()
                if config.is_cloud:
                    logger.info(f"Connecting to Qdrant Cloud: {config.qdrant_url}")
                    
                    api_key = config.qdrant_api_key
                    if not api_key:
                        logger.warning("API Key missing in config! Attempting direct refetch from .env...")
                        import os
                        from dotenv import load_dotenv
                        load_dotenv(override=True)
                        api_key = os.getenv("QDRANT_API_KEY")
                    
                    if api_key:
                        logger.info("Qdrant API Key: configured")
                    else:
                        logger.error("CRITICAL: Qdrant API Key is MISSING!")
                    
                    _qdrant_client = QdrantClient(
                        url=config.qdrant_url,
                        api_key=api_key,
                        timeout=30
                    )
                else:
                    logger.info(f"Using local Qdrant: {config.qdrant_url}")
                    _qdrant_client = QdrantClient(path=config.qdrant_url)
    return _qdrant_client
    return _qdrant_client


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Get DuckDB connection"""
    config = get_config()
    return duckdb.connect(config.duckdb_path, read_only=True)


def load_skill_relations() -> Dict[str, List[Tuple[str, float]]]:
    """Load skill relationships from DuckDB for expansion (thread-safe)"""
    global _skill_relations
    if _skill_relations is not None:
        return _skill_relations
    
    with _relations_lock:
        if _skill_relations is not None:  # Double-check pattern
            return _skill_relations
        
        conn = get_db_connection()
        try:
            rows = conn.execute("""
                SELECT skill_a, skill_b, semantic_confidence
                FROM skill_relationships
                WHERE semantic_confidence >= 0.7
                ORDER BY semantic_confidence DESC
            """).fetchall()
            
            relations: Dict[str, List[Tuple[str, float]]] = {}
            for skill_a, skill_b, sim in rows:
                if skill_a not in relations:
                    relations[skill_a] = []
                if skill_b not in relations:
                    relations[skill_b] = []
                relations[skill_a].append((skill_b, sim))
                relations[skill_b].append((skill_a, sim))
            
            _skill_relations = relations
            logger.info(f"Loaded {len(relations)} skills with relationships")
            return relations
        except Exception as e:
            logger.warning(f"Could not load skill relations: {e}")
            _skill_relations = {}
            return {}
        finally:
            conn.close()


# =============================================================================
# SKILL EXPANSION
# =============================================================================

def expand_skills(skills: List[str], max_per_skill: int = 5) -> Dict[str, List[str]]:
    """
    Expand skills using skill_relationships table.
    
    Returns dict mapping original skill -> [related skills]
    """
    if not skills:
        return {}
    
    relations = load_skill_relations()
    config = get_config()
    
    expanded = {}
    for skill in skills:
        skill_lower = skill.lower().strip()
        if skill_lower in relations:
            # Get top related skills by similarity
            related = relations[skill_lower][:max_per_skill]
            expanded[skill] = [r[0] for r in related]
        else:
            expanded[skill] = []
    
    return expanded


# =============================================================================
# QDRANT SEARCH
# =============================================================================

def build_qdrant_filter(query: ParsedQueryV2) -> Optional[Filter]:
    """
    Build Qdrant filter from parsed query.
    Normalizes all inputs for robust matching (handles aliases, abbreviations, case).
    """
    must_conditions = []
    must_not_conditions = []
    filters = query.filters
    
    def with_case_variations(values: list) -> list:
        """Add case variations to handle unknown storage format."""
        expanded = set()
        for v in values:
            expanded.add(v.lower())
            expanded.add(v.title())
            expanded.add(v.upper())
            expanded.add(v)  # original
        return list(expanded)
    
    # Skills - must_have (AND) - expand aliases + case variations
    for skill in filters.skills.must_have:
        skill_variations = expand_skill_search([skill])
        # Add case variations for each alias
        skill_variations = with_case_variations(skill_variations)
        must_conditions.append(
            FieldCondition(key="skills", match=MatchAny(any=skill_variations))
        )
    
    # Skills - exclude (NOT)
    for skill in filters.skills.exclude:
        skill_variations = expand_skill_search([skill])
        skill_variations = with_case_variations(skill_variations)
        must_not_conditions.append(
            FieldCondition(key="skills", match=MatchAny(any=skill_variations))
        )
    
    # Location: NOT used as a hard Qdrant filter
    # Reason: (1) location data is stored inconsistently across payload fields,
    # (2) the dense vector embedding already captures location semantics from search text,
    # (3) the smart_rerank function handles location boost via profile hydration from DuckDB.
    # Skills and experience work reliably as hard filters since their payload fields are consistent.
    
    # Companies - worked_at with alias expansion + case variations
    if filters.companies.worked_at:
        company_values = expand_company_search(filters.companies.worked_at)
        company_values = with_case_variations(company_values)
        if filters.companies.current_only:
            must_conditions.append(
                FieldCondition(key="current_company", match=MatchAny(any=company_values))
            )
        else:
            must_conditions.append(
                FieldCondition(key="companies", match=MatchAny(any=company_values))
            )
    
    # Companies - exclude (NOT)
    for company in filters.companies.exclude:
        company_variations = expand_company_search([company])
        company_variations = with_case_variations(company_variations)
        must_not_conditions.append(
            FieldCondition(key="companies", match=MatchAny(any=company_variations))
        )
    
    # Legacy current_company filter
    if filters.current_company:
        company_normalized = normalize_company(filters.current_company)
        must_conditions.append(
            FieldCondition(key="current_company", match=MatchAny(any=with_case_variations([company_normalized])))
        )
    
    # Domain
    if filters.domain:
        domain_val = normalize_text(filters.domain)
        must_conditions.append(
            FieldCondition(key="domain", match=MatchAny(any=with_case_variations([domain_val])))
        )
    
    # Experience range
    if filters.experience.min_years is not None:
        must_conditions.append(
            FieldCondition(key="years_experience", range=Range(gte=float(filters.experience.min_years)))
        )
    if filters.experience.max_years is not None:
        must_conditions.append(
            FieldCondition(key="years_experience", range=Range(lte=float(filters.experience.max_years)))
        )
    
    if must_conditions or must_not_conditions:
        return Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None
        )
    return None


def get_sparse_encoder():
    """Get BM25 sparse encoder for query encoding"""
    global _sparse_encoder
    if _sparse_encoder is None:
        from pathlib import Path
        import json
        import re
        import math
        from collections import Counter
        
        base_dir = Path(__file__).parent.parent
        cache_dir = base_dir / "Database" / "sparse_cache"
        
        class QueryEncoder:
            TOKEN_PATTERN = re.compile(r'[a-zA-Z0-9]+|[\u3040-\u309F]+|[\u30A0-\u30FF]+|[\u4E00-\u9FFF]+')
            STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were'}
            
            def __init__(self):
                self.vocabulary = {}
                self.idf = {}
                self.avg_doc_len = 50.0
                self.k1 = 1.5
                self.b = 0.75
                
                try:
                    with open(cache_dir / "vocabulary.json", 'r', encoding='utf-8') as f:
                        self.vocabulary = json.load(f)
                    with open(cache_dir / "idf.json", 'r', encoding='utf-8') as f:
                        self.idf = {int(k): v for k, v in json.load(f).items()}
                    with open(cache_dir / "stats.json", 'r', encoding='utf-8') as f:
                        stats = json.load(f)
                        self.avg_doc_len = stats.get('avg_doc_len', 50.0)
                    logger.info(f"Loaded sparse encoder: {len(self.vocabulary)} tokens")
                except Exception as e:
                    logger.warning(f"Could not load sparse encoder: {e}")
            
            def encode(self, text: str):
                if not text or not self.vocabulary:
                    return [0], [0.001]
                
                text = text.lower()
                tokens = self.TOKEN_PATTERN.findall(text)
                tokens = [t for t in tokens if len(t) >= 2 and t not in self.STOPWORDS]
                
                if not tokens:
                    return [0], [0.001]
                
                tf_counts = Counter(tokens)
                doc_len = len(tokens)
                
                indices, values = [], []
                for token, tf in tf_counts.items():
                    if token not in self.vocabulary:
                        continue
                    idx = self.vocabulary[token]
                    idf = self.idf.get(idx, 0.0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score = idf * (numerator / denominator)
                    if score > 0:
                        indices.append(idx)
                        values.append(float(score))
                
                return indices if indices else [0], values if values else [0.001]
        
        _sparse_encoder = QueryEncoder()
    
    return _sparse_encoder


def search_qdrant_hybrid(
    query_text: str,
    qdrant_filter: Optional[Filter],
    limit: int
) -> List[Tuple[int, float]]:
    """
    Search Qdrant with hybrid (dense + sparse) and RRF fusion.
    Returns list of (forager_id, score) tuples.
    """
    from qdrant_client.models import Prefetch, SparseVector, Fusion, FusionQuery
    
    config = get_config()
    qdrant = get_qdrant_client()
    model = get_embedding_model()
    sparse_encoder = get_sparse_encoder()
    
    # Generate dense embedding
    dense_vector = model.encode(query_text).tolist()
    
    # Generate sparse vector (BM25)
    sparse_indices, sparse_values = sparse_encoder.encode(query_text)
    
    # Check which collection exists
    collections = [c.name for c in qdrant.get_collections().collections]
    
    logger.info(f"Available Qdrant collections: {collections}")
    logger.info(f"Using collection: {config.qdrant_collection}")
    
    # Check collection stats
    try:
        collection_info = qdrant.get_collection(config.qdrant_collection)
        logger.info(f"Collection points count: {collection_info.points_count}")
    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")
    
    if config.qdrant_collection in collections:
        # Hybrid search with RRF fusion
        try:
            results = qdrant.query_points(
                collection_name=config.qdrant_collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit * 3,
                        filter=qdrant_filter
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=limit * 3,
                        filter=qdrant_filter
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit * 2,
                with_payload=["forager_id"]
            ).points
        except Exception as e:
            # Fallback to dense-only if hybrid fails
            logger.warning(f"Hybrid search failed, falling back to dense: {e}")
            results = qdrant.query_points(
                collection_name=config.qdrant_collection,
                query=dense_vector,
                using="dense",
                query_filter=qdrant_filter,
                limit=limit * 2,
                with_payload=["forager_id"]
            ).points
            
    elif config.qdrant_collection_dense in collections:
        # Fall back to dense-only collection
        results = qdrant.query_points(
            collection_name=config.qdrant_collection_dense,
            query=dense_vector,
            query_filter=qdrant_filter,
            limit=limit * 2,
            with_payload=["person_id"]
        ).points
    else:
        raise HTTPException(
            status_code=500,
            detail=f"No Qdrant collection found. Run setup script first."
        )
    
    # Extract forager_ids and scores
    id_scores = []
    for hit in results:
        fid = hit.payload.get("forager_id") or hit.payload.get("person_id")
        if fid:
            id_scores.append((int(fid), float(hit.score)))
    
    return id_scores


# =============================================================================
# DUCKDB HYDRATION
# =============================================================================

def hydrate_profiles_from_duckdb(person_ids: List[int]) -> Dict[int, Dict]:
    """
    Hydrate full profile details from DuckDB for given person IDs.
    Returns dict of {person_id: profile_dict}
    """
    if not person_ids:
        return {}
    
    conn = get_db_connection()
    try:
        ids_str = ','.join(str(i) for i in person_ids)
        
        # Fetch base profile data
        rows = conn.execute(f"""
            SELECT 
                pp.person_id,
                pp.full_name,
                pp.canonical_city as city,
                pp.canonical_state as state,
                pp.canonical_country as country,
                pp.primary_domain as domain,
                pp.years_experience,
                pp.profile_completeness,
                pp.canonical_skills as skills,
                pp.current_role_title as current_title,
                pp.current_role_company as current_company,
                p.photo,
                p.headline,
                p.linkedin_url,
                p.linkedin_slug,
                p.industry,
                p.is_creator,
                p.is_influencer,
                p.search_name,
                p.description,
                p.address,
                p.linkedin_country,
                p.linkedin_area,
                p.date_updated,
                p.primary_locale,
                p.temporary_status,
                p.temporary_emoji_status,
                p.background_picture,
                p.area
            FROM processed_profiles pp
            LEFT JOIN persons p ON pp.person_id = p.forager_id
            WHERE pp.person_id IN ({ids_str})
        """).fetchall()
        
        columns = ['forager_id', 'full_name', 'city', 'state', 'country', 
                   'domain', 'years_experience', 'profile_completeness',
                   'skills', 'current_title', 'current_company', 'photo', 
                   'headline', 'linkedin_url', 'linkedin_slug', 'industry',
                   'is_creator', 'is_influencer', 'search_name', 'description',
                   'address', 'linkedin_country', 'linkedin_area', 'date_updated',
                   'primary_locale', 'temporary_status', 'temporary_emoji_status',
                   'background_picture', 'area']
        
        profiles = {}
        for row in rows:
            profile = dict(zip(columns, row))
            fid = profile['forager_id']
            
            # Initialize additional fields
            profile['certifications'] = []
            profile['work_history'] = []
            profile['education'] = []
            
            profiles[fid] = profile
        
        # Fetch certifications
        cert_rows = conn.execute(f"""
            SELECT forager_id, certificate_name, issue_date, expiry_date
            FROM certifications
            WHERE forager_id IN ({ids_str})
            ORDER BY issue_date DESC NULLS LAST
        """).fetchall()
        
        for fid, cert_name, issue_date, expiry_date in cert_rows:
            if fid in profiles and cert_name:
                profiles[fid]['certifications'].append(cert_name)
        
        # Fetch work history (past roles)
        roles_rows = conn.execute(f"""
            SELECT forager_id, role_title, company_name, start_date, end_date, 
                   location, description
            FROM roles
            WHERE forager_id IN ({ids_str})
            ORDER BY start_date DESC NULLS LAST
            LIMIT 1000
        """).fetchall()
        
        for fid, title, company, start_dt, end_dt, loc, desc in roles_rows:
            if fid in profiles:
                profiles[fid]['work_history'].append({
                    'title': title,
                    'company': company,
                    'start_date': str(start_dt) if start_dt else None,
                    'end_date': str(end_dt) if end_dt else None,
                    'location': loc,
                    'description': desc
                })
        
        # Fetch education
        edu_rows = conn.execute(f"""
            SELECT forager_id, school_name, degree, field_of_study, 
                   start_date, end_date
            FROM educations
            WHERE forager_id IN ({ids_str})
            ORDER BY start_date DESC NULLS LAST
        """).fetchall()
        
        for fid, school, degree, field, start_dt, end_dt in edu_rows:
            if fid in profiles:
                profiles[fid]['education'].append({
                    'school': school,
                    'degree': degree,
                    'field': field,
                    'start_date': str(start_dt) if start_dt else None,
                    'end_date': str(end_dt) if end_dt else None
                })
        
        return profiles
    finally:
        conn.close()


# =============================================================================
# RANKING
# =============================================================================

def calculate_ranking_bonus(
    profile: Dict,
    query: ParsedQueryV2,
    base_score: float
) -> Tuple[float, List[str]]:
    """
    Calculate ranking bonus and matched skills.
    Returns (final_score, matched_skills).
    """
    config = get_config()
    bonus = 0.0
    matched_skills = []
    
    # Skills coverage - search across skills array AND text fields (headline, title, description)
    profile_skills = set(s.lower() for s in (profile.get('skills') or []))
    
    # Helper: match skill against skills set
    def match_skill(query_skill: str, profile_skills_set: set) -> Optional[str]:
        query_skill_lower = query_skill.lower()
        for ps in profile_skills_set:
            if query_skill_lower in ps or ps in query_skill_lower:
                return ps
        return None
    
    # Must have skills - match strictly in skills array
    must_have = [s.lower() for s in query.filters.skills.must_have]
    for skill in must_have:
        matched_ps = match_skill(skill, profile_skills)
        if matched_ps:
            matched_skills.append(skill)
    
    if must_have:
        coverage = len(matched_skills) / len(must_have)
        bonus += coverage * config.bonus_skills_coverage
    
    # Nice to have skills boost - match strictly in skills array
    nice_to_have = [s.lower() for s in query.filters.skills.nice_to_have]
    nice_matched = []
    for skill in nice_to_have:
        matched_ps = match_skill(skill, profile_skills)
        if matched_ps:
            nice_matched.append(skill)
    matched_skills.extend(nice_matched)
    
    if nice_to_have:
        nice_coverage = len(nice_matched) / len(nice_to_have)
        bonus += nice_coverage * 0.10  # Smaller bonus for nice-to-have

    
    # Experience fit
    exp = profile.get('years_experience')
    if exp is not None:
        min_y = query.filters.experience.min_years
        max_y = query.filters.experience.max_years
        
        if min_y is not None and max_y is not None:
            if min_y <= exp <= max_y:
                bonus += config.bonus_experience_fit
            elif abs(exp - min_y) <= 2 or abs(exp - max_y) <= 2:
                bonus += config.bonus_experience_fit * 0.5  # Partial bonus
    
    # Location match
    loc = query.filters.location
    if loc.city and profile.get('city', '').lower() == loc.city.lower():
        bonus += config.bonus_location_exact
    elif loc.state and profile.get('state', '').lower() == loc.state.lower():
        bonus += config.bonus_location_nearby
    elif loc.country and profile.get('country', '').lower() == loc.country.lower():
        bonus += config.bonus_location_nearby * 0.5
    
    # Profile completeness
    completeness = profile.get('profile_completeness') or 0
    bonus += (completeness / 100) * config.bonus_profile_completeness
    
    # Job titles boost - substring matching (not hard filter)
    if query.filters.job_titles:
        current_title = (profile.get('current_title') or '').lower()
        for title_keyword in query.filters.job_titles:
            if title_keyword.lower() in current_title:
                bonus += config.bonus_title_match
                break  # Only count once
    
    final_score = min(base_score + bonus, 1.0)  # Cap at 1.0
    return final_score, list(set(matched_skills))


# =============================================================================
# MAIN SEARCH FUNCTION
# =============================================================================

# Maximum query length to prevent DoS
MAX_QUERY_LENGTH = 10000

async def search_v2(query: ParsedQueryV2) -> SearchResponseV2:
    """
    Main search function implementing the full flow:
    1. Expand skills
    2. Build Qdrant filter
    3. Query Qdrant
    4. Hydrate from DuckDB
    5. Apply ranking
    6. Return results
    """
    start_time = time.time()
    timings = {}
    
    # Validate query length to prevent DoS
    search_text_len = len(query.search_text or "")
    if search_text_len > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"Query too long ({search_text_len} chars). Maximum: {MAX_QUERY_LENGTH}"
        )
    
    config = get_config()
    limit = min(query.options.limit, config.max_limit)
    
    # 1. Expand skills if enabled
    t0 = time.time()
    expanded_skills = {}
    search_skills = list(query.filters.skills.must_have)
    
    if query.options.expand_skills and query.filters.skills.must_have:
        expanded_skills = expand_skills(query.filters.skills.must_have)
        # Add expanded skills to search
        for original, related in expanded_skills.items():
            search_skills.extend(related)
        search_skills = list(set(search_skills))
    
    timings['skill_expansion'] = int((time.time() - t0) * 1000)
    
    # 2. Build enhanced search text for better embeddings
    # Location filtering is handled by Qdrant filters (build_qdrant_filter).
    # Only mention location once here for light semantic context.
    search_text = query.search_text or ""
    
    # Add location for light semantic context (hard filtering is done by Qdrant filters)
    # NOTE: For smart_search, the search_text is pre-enriched in the endpoint before filters
    # are cleared, so these will be no-ops (filters.location.city will be None).
    location_parts = []
    if query.filters.location.city:
        location_parts.append(query.filters.location.city)
    if query.filters.location.state:
        location_parts.append(query.filters.location.state)
    if query.filters.location.country:
        location_parts.append(query.filters.location.country)
    
    if location_parts:
        search_text += " " + " ".join(location_parts)
    
    # Add job titles for context
    if query.filters.job_titles:
        search_text += " " + " ".join(query.filters.job_titles)
    
    # Add top skills for grounding
    if search_skills:
        search_text += " " + " ".join(search_skills[:5])
    
    # Fallback if still empty
    if not search_text.strip():
        search_text = " ".join(search_skills) if search_skills else "professional"
    
    # 3. Build Qdrant filter
    t0 = time.time()
    qdrant_filter = build_qdrant_filter(query)
    timings['filter_build'] = int((time.time() - t0) * 1000)
    
    # DEBUG: Log the actual filter being sent to Qdrant
    logger.info(f"Qdrant filter: {qdrant_filter}")
    logger.info(f"Search text for embedding: '{search_text[:100]}'")
    
    # 4. Query Qdrant
    # CRITICAL FIX: Fetch MORE candidates (up to 1000) to ensure we have enough to filter
    # and to allow accurate pagination. We slice later.
    SEARCH_FETCH_LIMIT = 1000 
    t0 = time.time()
    try:
        id_scores = search_qdrant_hybrid(search_text, qdrant_filter, SEARCH_FETCH_LIMIT)
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    timings['qdrant_search'] = int((time.time() - t0) * 1000)
    
    # 5. Hydrate from DuckDB
    t0 = time.time()
    forager_ids = [fid for fid, _ in id_scores]
    profiles = hydrate_profiles_from_duckdb(forager_ids)
    timings['duckdb_fetch'] = int((time.time() - t0) * 1000)
    
    # 6. Apply ranking and build results
    t0 = time.time()
    results = []
    exclude_skills = set(s.lower() for s in query.filters.skills.exclude)
    
    # Additional strict filters setup
    target_first = (query.filters.first_name or "").lower()
    target_last = (query.filters.last_name or "").lower()
    
    for forager_id, base_score in id_scores:
        profile = profiles.get(forager_id)
        if not profile:
            continue
        
        # Filter out excluded skills
        profile_skills = set(s.lower() for s in (profile.get('skills') or []))
        if exclude_skills & profile_skills:
            continue  # Skip profiles with excluded skills
            
        # STRICT NAME FILTERING (Moved inside loop)
        full_name = (profile.get('full_name') or "").lower()
        if target_first or target_last:
            name_parts = full_name.split()
            p_first = name_parts[0] if name_parts else ""
            p_last = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            
            if target_first and target_first not in p_first:
                continue
            if target_last and target_last not in p_last:
                continue

        # Calculate final score
        final_score, matched_skills = calculate_ranking_bonus(profile, query, base_score)
        
        # Build location string (deduplicate parts like state == country)
        loc_parts = []
        if profile.get('city'):
            loc_parts.append(profile['city'])
        if profile.get('state') and profile['state'] not in loc_parts:
            loc_parts.append(profile['state'])
        if profile.get('country') and profile['country'] != 'Unknown' and profile['country'] not in loc_parts:
            loc_parts.append(profile['country'])
        location = ", ".join(loc_parts) if loc_parts else "Unknown"
        
        results.append(CandidateResultV2(
            forager_id=forager_id,
            score=round(final_score, 4),
            full_name=profile.get('full_name', 'Unknown'),
            current_title=profile.get('current_title'),
            current_company=profile.get('current_company'),
            location=location,
            city=profile.get('city'),
            country=profile.get('country'),
            years_experience=profile.get('years_experience'),
            domain=profile.get('domain'),
            profile_completeness=profile.get('profile_completeness'),
            skills=profile.get('skills', [])[:15],
            matched_skills=matched_skills,
            certifications=profile.get('certifications', []),
            matched_certifications=[],  # Will be calculated in smart_rerank
            work_history=profile.get('work_history', [])[:10],  # Limit to 10 most recent
            education=profile.get('education', []),
            headline=profile.get('headline'),
            linkedin_url=profile.get('linkedin_url'),
            linkedin_slug=profile.get('linkedin_slug'),
            industry=profile.get('industry'),
            search_name=profile.get('search_name'),
            description=profile.get('description'),
            is_creator=profile.get('is_creator'),
            is_influencer=profile.get('is_influencer'),
            photo=profile.get('photo'),
            address=profile.get('address'),
            linkedin_country=profile.get('linkedin_country'),
            linkedin_area=profile.get('linkedin_area'),
            date_updated=profile.get('date_updated'),
            primary_locale=profile.get('primary_locale'),
            temporary_status=profile.get('temporary_status'),
            temporary_emoji_status=profile.get('temporary_emoji_status'),
            background_picture=profile.get('background_picture'),
            area=profile.get('area')
        ))
    
    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)
    
    timings['ranking'] = int((time.time() - t0) * 1000)
    total_time = int((time.time() - start_time) * 1000)
    
    # Calculate total matches based on ACTUAL filtered results
    final_total = len(results)
    
    # Apply Pagination
    start_idx = query.options.offset
    end_idx = start_idx + query.options.limit
    paged_results = results[start_idx:end_idx]
    
    return SearchResponseV2(
        total_matches=final_total,
        returned=len(paged_results),
        took_ms=total_time,
        results=paged_results,
        expanded_skills=expanded_skills if expanded_skills else None,
        timings=timings
    )


# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    config = get_config()
    logger.info("=" * 60)
    logger.info("  TALENTIN SEARCH API v2")
    logger.info("=" * 60)
    logger.info(f"Qdrant: {config.qdrant_url}")
    logger.info(f"DuckDB: {config.duckdb_path}")
    logger.info(f"Cloud Mode: {config.is_cloud}")
    logger.info("=" * 60)
    
    # Pre-load models
    get_embedding_model()
    load_skill_relations()
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="Talentin Search API v2",
    description="Semantic talent search with skill expansion and hybrid ranking",
    version="2.0.0",
    lifespan=lifespan
)

# =============================================================================
# API ENHANCEMENTS - Rate Limiting, Caching, Analytics
# =============================================================================

try:
    from api_enhancements import (
        RateLimitMiddleware,
        RequestLoggingMiddleware,
        get_analytics_tracker,
        get_search_cache,
        get_spending_cap,
        check_spending_cap,
        SearchAnalytics,
        cache_key_from_query,
        get_error_logger
    )
    
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("API enhancements loaded: rate limiting, caching, analytics")
    API_ENHANCEMENTS_ENABLED = True
except ImportError as e:
    logger.warning(f"API enhancements not available: {e}")
    API_ENHANCEMENTS_ENABLED = False

# CORS - Environment-aware origins
import os
_search_env = os.getenv("SEARCH_ENV", "local").lower()

ALLOWED_ORIGINS = [
    "https://jp.talentin.ai",
    "https://talentin.ai",
    "https://www.talentin.ai",
]

# Only allow localhost in development mode
if _search_env in ("local", "development", "dev"):
    ALLOWED_ORIGINS.extend([
        "http://localhost:3000",  # Local dev
        "http://localhost:5173",  # Vite dev
        "http://127.0.0.1:3000",
    ])
    logger.info("CORS: Localhost origins enabled (Development mode)")
else:
    logger.info(f"CORS: Running in {_search_env} mode - Production origins only")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add Cache-Control middleware to prevent browser caching of search results
@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    """Prevent browser caching of search API responses"""
    response = await call_next(request)
    
    # Add no-cache headers for search endpoints
    if "/api/v2/search" in request.url.path or "/api/v2/smart-search" in request.url.path:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response


@app.get("/api/v2/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/api/v2/search", response_model=SearchResponseV2)
async def search_endpoint(query: ParsedQueryV2):
    """
    Main search endpoint.
    
    Accepts parsed query JSON from external LLM API.
    Returns ranked candidates with modified skills.
    """
    response = await search_v2(query)
    
    # Add city breakdown
    from collections import Counter
    from search_schema import CityCount
    city_counts = Counter()
    for result in response.results:
        city = result.city or "Unknown"
        if city:
            city_counts[city] += 1
    
    response.city_breakdown = [
        CityCount(city=city, count=count)
        for city, count in city_counts.most_common()
    ]
    
    return response


class CityFilterRequest(BaseModel):
    """Request to filter results by cities"""
    query_id: str = Field(..., description="Original query ID")
    selected_cities: List[str] = Field(..., description="Cities to filter by")
    all_results: List[CandidateResultV2] = Field(..., description="All results to filter from")


@app.post("/api/v2/filter-by-cities", response_model=SearchResponseV2)
async def filter_by_cities(request: CityFilterRequest):
    """
    Filter existing search results by selected cities.
    
    Frontend sends all results + selected cities, backend filters and returns.
    This avoids re-running the full search.
    """
    if not request.selected_cities:
        # No filter, return all
        filtered = request.all_results
    else:
        # Filter by selected cities
        selected_set = set(c.lower() for c in request.selected_cities)
        filtered = [
            r for r in request.all_results
            if (r.city or "Unknown").lower() in selected_set
        ]
    
    # Recalculate city breakdown for filtered results
    from collections import Counter
    from search_schema import CityCount
    city_counts = Counter()
    for result in filtered:
        city = result.city or "Unknown"
        if city:
            city_counts[city] += 1
    
    city_breakdown = [
        CityCount(city=city, count=count)
        for city, count in city_counts.most_common()
    ]
    
    return SearchResponseV2(
        query_id=request.query_id,
        total_matches=len(request.all_results),
        returned=len(filtered),
        took_ms=0,  # No search time, just filtering
        results=filtered,
        city_breakdown=city_breakdown
    )

# =============================================================================
# LEGACY COMPATIBILITY LAYER (For Integration)
# =============================================================================

# (Legacy compatibility removed to support running separate backends)



@app.get("/api/stats")  # Legacy alias
@app.get("/api/v2/stats")
async def stats():
    """Get database statistics"""
    conn = get_db_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM processed_profiles").fetchone()[0]
        
        unique_countries = conn.execute("""
            SELECT COUNT(DISTINCT canonical_country) 
            FROM processed_profiles 
            WHERE canonical_country IS NOT NULL AND canonical_country != ''
        """).fetchone()[0]
        
        unique_cities = conn.execute("""
            SELECT COUNT(DISTINCT canonical_city) 
            FROM processed_profiles 
            WHERE canonical_city IS NOT NULL AND canonical_city != ''
        """).fetchone()[0]
        
        unique_industries = conn.execute("""
            SELECT COUNT(DISTINCT primary_domain) 
            FROM processed_profiles 
            WHERE primary_domain IS NOT NULL AND primary_domain != ''
        """).fetchone()[0]
        
        total_skills = conn.execute("""
            SELECT COUNT(*) FROM skills
        """).fetchone()[0]
        
        domains = conn.execute("""
            SELECT primary_domain, COUNT(*) as count
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL
            GROUP BY primary_domain
            ORDER BY count DESC
        """).fetchall()
        
        return {
            "total_records": total,
            "total_profiles": total,  # Keep for backwards compatibility
            "unique_countries": unique_countries,
            "unique_cities": unique_cities,
            "unique_industries": unique_industries,
            "total_skills": total_skills,
            "domains": {d[0]: d[1] for d in domains},
            "qdrant_collection": get_config().qdrant_collection
        }
    finally:
        conn.close()


@app.get("/api/v2/expand-skills")
async def expand_skills_endpoint(skills: str):
    """
    Debug endpoint to test skill expansion.
    Pass comma-separated skills.
    """
    skill_list = [s.strip() for s in skills.split(",")]
    expanded = expand_skills(skill_list)
    return {
        "input": skill_list,
        "expanded": expanded
    }


# =============================================================================
# PHASE 3: ADVANCED FILTER ENDPOINTS
# =============================================================================

from filter_service import (
    get_all_facets, autocomplete, get_filter_metadata,
    FacetsResponse, AutocompleteResponse, FilterMetadataResponse,
    clear_facet_cache, get_cache_stats
)


@app.get("/api/v2/filters/facets", response_model=FacetsResponse)
async def facets_endpoint(current_filters: Optional[str] = None):
    """
    Get filter facets with counts.
    
    Returns top skills, locations, companies, experience ranges, and domains
    with counts based on current filter state.
    
    Args:
        current_filters: Optional JSON string of active filters
            Example: {"location": {"city": "Tokyo"}}
    
    Returns:
        FacetsResponse with all facet types and counts
    """
    import json
    
    filters_dict = None
    if current_filters:
        try:
            filters_dict = json.loads(current_filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in current_filters")
    
    return get_all_facets(filters_dict)


@app.get("/api/v2/autocomplete/{field}", response_model=AutocompleteResponse)
async def autocomplete_endpoint(field: str, q: str, limit: int = 10):
    """
    Real-time autocomplete for filter fields.
    
    Args:
        field: Field to autocomplete (skills, companies, locations, titles)
        q: Search query (prefix)
        limit: Max suggestions (default 10)
    
    Returns:
        AutocompleteResponse with matching suggestions and counts
    
    Example:
        GET /api/v2/autocomplete/skills?q=pyth&limit=5
    """
    valid_fields = ["skills", "companies", "locations", "titles"]
    if field.lower() not in valid_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid field. Must be one of: {valid_fields}"
        )
    
    if len(q) < 2:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 2 characters"
        )
    
    return autocomplete(field, q, min(limit, 50))


@app.get("/api/v2/filters/metadata", response_model=FilterMetadataResponse)
async def filter_metadata_endpoint():
    """
    Get metadata about all available filters.
    
    Returns:
        - Total profile count
        - Unique counts for each filter type
        - Top 10 values for skills
        - Experience min/max/avg
    """
    return get_filter_metadata()


@app.post("/api/v2/filters/cache/clear")
async def clear_cache_endpoint():
    """Clear the facet cache (admin endpoint)"""
    clear_facet_cache()
    return {"status": "ok", "message": "Facet cache cleared"}


@app.get("/api/v2/filters/cache/stats")
async def cache_stats_endpoint():
    """Get cache statistics"""
    return get_cache_stats()


# =============================================================================
# API ANALYTICS & MONITORING ENDPOINTS
# =============================================================================

@app.get("/api/v2/analytics/stats")
async def analytics_stats_endpoint():
    """Get search analytics statistics"""
    if not API_ENHANCEMENTS_ENABLED:
        raise HTTPException(status_code=501, detail="Analytics not enabled")
    
    tracker = get_analytics_tracker()
    return tracker.get_stats()


@app.get("/api/v2/analytics/slow-queries")
async def analytics_slow_queries_endpoint(threshold_ms: int = 500):
    """Get slow queries for debugging"""
    if not API_ENHANCEMENTS_ENABLED:
        raise HTTPException(status_code=501, detail="Analytics not enabled")
    
    tracker = get_analytics_tracker()
    return {"slow_queries": tracker.get_slow_queries(threshold_ms)}


@app.get("/api/v2/openai/spending")
async def openai_spending_endpoint():
    """Get OpenAI spending statistics"""
    if not API_ENHANCEMENTS_ENABLED:
        raise HTTPException(status_code=501, detail="Spending tracking not enabled")
    
    cap = get_spending_cap()
    return cap.get_stats()


@app.get("/api/v2/errors/recent")
async def recent_errors_endpoint(limit: int = 20):
    """Get recent errors for debugging"""
    if not API_ENHANCEMENTS_ENABLED:
        raise HTTPException(status_code=501, detail="Error tracking not enabled")
    
    error_logger = get_error_logger()
    return {
        "stats": error_logger.get_stats(),
        "recent_errors": error_logger.get_recent_errors(limit)
    }


# =============================================================================
# PHASE 2: TEXT PARSING ENDPOINTS
# =============================================================================

from text_parser import (
    parse_job_description,
    extract_skills_from_text,
    summarize_profile,
    jd_to_search_query
)
from pydantic import BaseModel

class JDParsingRequest(BaseModel):
    """Request for JD parsing"""
    text: str

class SkillExtractionRequest(BaseModel):
    """Request for skill extraction"""
    text: str

class ProfileSummarizationRequest(BaseModel):
    """Request for profile summarization"""
    profile_data: Dict[str, Any]

@app.post("/api/v2/parse/job-description")
async def parse_jd_endpoint(request: JDParsingRequest):
    """
    Parse a Job Description to extract requirements.
    
    Returns:
        - Required skills
        - Nice-to-have skills
        - Experience level
        - Suggested search query
    """
    # Check spending cap for OpenAI calls
    if API_ENHANCEMENTS_ENABLED:
        check_spending_cap(0.001)
    
    if len(request.text) < 10:
        raise HTTPException(status_code=400, detail="Text too short")
        
    result, cost = jd_to_search_query(request.text)
    
    # Track spending
    if API_ENHANCEMENTS_ENABLED and cost > 0:
        get_spending_cap().add_spend(cost)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to parse job description")
        
    return {
        "parsed_search_query": result,
        "cost_usd": cost
    }


@app.post("/api/v2/parse/extract-skills")
async def extract_skills_endpoint(request: SkillExtractionRequest):
    """
    Extract technical skills from unstructured text.
    
    Returns:
        List of skills with confidence and categories.
    """
    if len(request.text) < 5:
        raise HTTPException(status_code=400, detail="Text too short")
        
    skills, cost = extract_skills_from_text(request.text)
    
    if skills is None:
        raise HTTPException(status_code=500, detail="Failed to extract skills")
        
    return {
        "skills": skills,
        "cost_usd": cost
    }


@app.post("/api/v2/parse/summarize-profile")
async def summarize_profile_endpoint(request: ProfileSummarizationRequest):
    """
    Generate a 2-sentence summary and key highlights for a profile.
    """
    summary, cost = summarize_profile(request.profile_data)
    
    if not summary:
        raise HTTPException(status_code=500, detail="Failed to summarize profile")
        
    return {
        "summary": summary,
        "cost_usd": cost
    }


# =============================================================================
# SMART SEARCH - Natural Language Interface
# =============================================================================

class SmartSearchQuery(BaseModel):
    """Free-text search query - handles ANY human input"""
    query: str  # "python developers in SF", "ML eng at FAANG", "東京のエンジニア"
    limit: int = 20
    location_preference: str = Field(default="preferred", description="Location mode: remote, preferred, or must_match")
    selected_locations: List[str] = Field(default_factory=list, description="Hard filter to these locations")


class SmartSearchResponse(BaseModel):
    """Response from smart search"""
    total_matches: int
    returned: int
    took_ms: int
    query_understanding: Dict
    city_breakdown: List[CityCount] = []  # Legacy field for backward compatibility
    facets: Optional[SearchFacets] = None  # New facets structure
    results: List[CandidateResultV2]


@app.post("/api/v2/smart-search", response_model=SmartSearchResponse)
async def smart_search_endpoint(request: SmartSearchQuery, http_request: Request = None):
    """
    🧠 SMART SEARCH - Natural language talent search.
    
    This endpoint handles ANY human input:
    - "python developers in SF with 5+ years"
    - "ML eng at FAANG companies"
    - "東京のPythonエンジニア"
    - "ex google engineers who know react"
    - "senior backend dev with aws k8s exp in bay area"
    - Typos: "pythin devloper in sanfracisco"
    
    The AI extracts:
    - Skills (with expansion)
    - Companies (with aliases)
    - Location
    - Experience level
    - Job titles
    
    And returns the BEST matching candidates.
    """
    from query_preprocessor import smart_preprocess
    from openai_parser import parse_query_with_openai
    
    # Extract request ID for correlation
    request_id = http_request.headers.get("X-Request-ID", "unknown") if http_request else "unknown"
    
    start_time = time.time()
    config = get_config()
    limit = min(request.limit, config.max_limit)
    location_preference = request.location_preference if hasattr(request, 'location_preference') else "preferred"
    
    # 1. Smart preprocessing with OpenAI
    logger.info("="*80)
    logger.info(f"🔍 NEW SMART SEARCH REQUEST")
    logger.info(f"   Request ID: {request_id}")
    logger.info(f"   Query Text: '{request.query}'")
    logger.info(f"   Query Length: {len(request.query)} chars")
    logger.info(f"   Query Hash: {hash(request.query)}")
    logger.info(f"   Limit: {request.limit}")
    logger.info(f"   Location Mode: {location_preference}")
    logger.info(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Try OpenAI first, fallback to regex preprocessor
    parsed_result, method, cost = parse_query_with_openai(request.query, fallback_to_regex=False)
    
    if parsed_result and method == "openai":
        logger.info(f"✅ OpenAI parsed - Skills: {parsed_result.filters.skills.must_have[:3]}, "
                   f"Titles: {parsed_result.filters.job_titles[:2]}, "
                   f"City: {parsed_result.filters.location.city}, Cost: ${cost:.6f}")
        
        # Save original location before clearing
        original_city = parsed_result.filters.location.city
        original_state = parsed_result.filters.location.state
        original_country = parsed_result.filters.location.country
        
        # Fetch more candidates — for must_match we need extra since we post-filter
        if location_preference == "must_match":
            parsed_result.options.limit = min(limit * 20, 1000)
        else:
            parsed_result.options.limit = min(limit * 5, 500)
        
        # Enrich search_text with titles + location for semantic matching
        # (these are NOT hard Qdrant filters — they rely on embedding similarity)
        enriched = parsed_result.search_text or request.query
        if parsed_result.filters.job_titles:
            enriched += " " + " ".join(parsed_result.filters.job_titles)
        if parsed_result.filters.location.city:
            enriched += " " + parsed_result.filters.location.city
        if parsed_result.filters.location.state:
            enriched += " " + parsed_result.filters.location.state
        if parsed_result.filters.location.country:
            enriched += " " + parsed_result.filters.location.country
        parsed_result.search_text = enriched.strip()
        
        # Clear location from Qdrant filter (data is stored inconsistently in Qdrant payload)
        # Location is handled by: embedding similarity (preferred) + post-filter (must_match)
        parsed_result.filters.location.city = None
        parsed_result.filters.location.state = None
        parsed_result.filters.location.country = None
        
        # Build smart_filters for post-processing (location modes, query understanding)
        smart_filters = {
            "skills": parsed_result.filters.skills.must_have + parsed_result.filters.skills.nice_to_have,
            "expanded_skills": [],
            "companies": parsed_result.filters.companies.worked_at,
            "expanded_companies": [],
            "certifications": parsed_result.filters.certifications if hasattr(parsed_result.filters, 'certifications') else [],
            "city": original_city,
            "state": original_state,
            "country": original_country,
            "min_years": parsed_result.filters.experience.min_years,
            "max_years": parsed_result.filters.experience.max_years,
            "titles": parsed_result.filters.job_titles,
        }
        
        logger.info(f"Search: text='{parsed_result.search_text[:80]}', "
                   f"skills_filter={parsed_result.filters.skills.must_have[:3]}, "
                   f"location_mode={location_preference}, location={original_city or original_state or original_country}")
    else:
        # Fallback to regex preprocessor
        logger.warning(f"⚠️ OpenAI parsing failed, using regex fallback")
        extracted = smart_preprocess(request.query)
        logger.info(f"Extracted - Skills: {extracted.skills[:3]}, City: {extracted.city}, Titles: {extracted.titles[:2]}")
        
        # Build ParsedQueryV2 from regex extraction
        from search_schema import (
            ParsedQueryV2, SkillFiltersV2, ExperienceFilterV2, LocationFilterV2,
            CompanyFilterV2, FiltersV2, SearchOptionsV2
        )
        
        # Skills as hard Qdrant filters
        primary_skills = extracted.skills[:3]
        secondary_skills = extracted.skills[3:]
        
        # Enrich search text with titles + location for semantic matching
        enhanced_search_text = extracted.search_text
        if extracted.titles:
            enhanced_search_text += " " + " ".join(extracted.titles[:3])
        if extracted.city:
            enhanced_search_text += " " + extracted.city
        if extracted.state:
            enhanced_search_text += " " + extracted.state
        if hasattr(extracted, 'country') and extracted.country:
            enhanced_search_text += " " + extracted.country
        
        parsed_result = ParsedQueryV2(
            search_text=enhanced_search_text.strip(),
            filters=FiltersV2(
                skills=SkillFiltersV2(
                    must_have=primary_skills,
                    nice_to_have=secondary_skills + (extracted.expanded_skills[:8] if hasattr(extracted, 'expanded_skills') else []),
                    exclude=[]
                ),
                experience=ExperienceFilterV2(
                    min_years=extracted.min_years,
                    max_years=extracted.max_years
                ),
                location=LocationFilterV2(
                    city=None,      # NOT a hard Qdrant filter — handled via embedding
                    state=None,
                    country=None
                ),
                companies=CompanyFilterV2(
                    worked_at=extracted.companies if hasattr(extracted, 'companies') else [],
                    current_only=False
                ),
                job_titles=extracted.titles,
                domain=None,
                current_company=None
            ),
            options=SearchOptionsV2(
                limit=min(limit * 20, 1000) if location_preference == "must_match" else min(limit * 5, 500),
                expand_skills=True
            )
        )
        
        smart_filters = {
            "skills": extracted.skills,
            "expanded_skills": extracted.expanded_skills if hasattr(extracted, 'expanded_skills') else [],
            "companies": extracted.companies if hasattr(extracted, 'companies') else [],
            "expanded_companies": extracted.expanded_companies if hasattr(extracted, 'expanded_companies') else [],
            "certifications": extracted.certifications if hasattr(extracted, 'certifications') else [],
            "city": extracted.city,
            "state": extracted.state,
            "country": extracted.country if hasattr(extracted, 'country') else None,
            "min_years": extracted.min_years,
            "max_years": extracted.max_years,
            "titles": extracted.titles,
        }
    
    # 3. Execute search with parsed query
    logger.info(f"🔍 Executing Qdrant search: text='{parsed_result.search_text[:60]}...', "
               f"skills={parsed_result.filters.skills.must_have[:3]}, "
               f"limit={parsed_result.options.limit}")
    response = await search_v2(parsed_result)
    
    logger.info(f"📊 Qdrant results: {len(response.results)} candidates | "
               f"took={response.took_ms}ms | "
               f"top scores: {[f'{r.score:.3f}' for r in response.results[:5]]}")
    
    # 4. SMART RE-RANKING: Apply comprehensive ranking based on skills, titles, location, etc.
    # This is what makes smart search "smart" - aggressive bonuses for exact matches
    logger.info(f"🎯 Applying smart re-ranking to {len(response.results)} candidates...")
    logger.info(f"   Smart filters: skills={smart_filters.get('skills', [])[:3]}, "
               f"city={smart_filters.get('city')}, titles={smart_filters.get('titles', [])[:2]}")
    reranked_results = smart_rerank(response.results, smart_filters, location_preference)
    logger.info(f"✅ After re-ranking: top_score={reranked_results[0].score:.3f if reranked_results else 0}, "
               f"candidate='{reranked_results[0].full_name[:30]}' if reranked_results else 'none'")
    logger.info(f"   Top 5 scores: {[f'{r.score:.3f}' for r in reranked_results[:5]]}")
    
    # Calculate facets from results
    facets = calculate_location_facets(reranked_results)
    
    # 5. LOCATION MODE HANDLING (post-filtering after re-ranking)
    location_preference = request.location_preference if hasattr(request, 'location_preference') else "preferred"
    results = reranked_results
    
    target_city = smart_filters.get("city")
    target_state = smart_filters.get("state")
    target_country = smart_filters.get("country")
    target_location = (target_city or target_state or target_country or "").lower()
    
    if location_preference == "must_match" and target_location:
        # MUST MATCH: Hard post-filter on DuckDB hydrated city/location
        before = len(results)
        results = [
            r for r in results
            if target_location in (r.city or "").lower()
            or target_location in (getattr(r, 'location', '') or "").lower()
            or target_location in (r.country or "").lower()
        ]
        logger.info(f"MUST_MATCH: {before} → {len(results)} (location: {target_location})")
    elif location_preference == "remote":
        # REMOTE: No location filtering at all — show everyone
        logger.info(f"REMOTE mode: showing all {len(results)} results")
    else:
        # PREFERRED: Qdrant already ranked by embedding similarity (which includes location)
        # Optional: filter by user-selected locations
        if hasattr(request, 'selected_locations') and request.selected_locations:
            before = len(results)
            results = filter_by_locations(results, request.selected_locations)
            logger.info(f"Selected locations filter: {before} → {len(results)}")
    
    # Limit to requested amount
    final_results = results[:limit]
    
    # City breakdown
    from collections import Counter
    city_counts = Counter()
    for result in final_results:
        city = result.city or "Unknown"
        if city:
            city_counts[city] += 1
    
    from search_schema import CityCount
    city_breakdown = [
        CityCount(city=city, count=count)
        for city, count in city_counts.most_common()
    ]
    
    total_time = int((time.time() - start_time) * 1000)
    
    return SmartSearchResponse(
        total_matches=len(results),
        returned=len(final_results),
        took_ms=total_time,
        city_breakdown=city_breakdown,
        facets=facets,
        query_understanding={
            "original_query": request.query,
            "extracted_skills": smart_filters["skills"],
            "expanded_skills": smart_filters["expanded_skills"][:10] if smart_filters["expanded_skills"] else [],
            "extracted_companies": smart_filters["companies"],
            "expanded_companies": smart_filters["expanded_companies"][:10] if smart_filters["expanded_companies"] else [],
            "extracted_location": {
                "city": smart_filters["city"],
                "state": smart_filters["state"],
                "country": smart_filters["country"]
            },
            "extracted_experience": {
                "min_years": smart_filters["min_years"],
                "max_years": smart_filters["max_years"],
                "level": None
            },
            "extracted_titles": smart_filters["titles"],
            "search_text": parsed_result.search_text,
            "keywords": []
        },
        results=final_results
    )


def smart_rerank(results: List[CandidateResultV2], filters: Dict, location_preference: str = "preferred") -> List[CandidateResultV2]:
    """
    Comprehensive smart re-ranking for natural language search.
    
    Matches across ALL available profile fields:
    - Skills: exact match in skills[], then fallback to headline/description text
    - Titles: exact substring + word overlap + headline matching
    - Location: city/state/country with configurable boost levels
    - Experience: range match + proximity scoring (closer to ideal = higher)
    - Companies: current + work history matching
    
    Args:
        results: List of candidates to rank
        filters: Dict with extracted filters (skills, city, titles, etc.)
        location_preference: "remote" (no location boost), "preferred" (+0.80 boost), "must_match" (+3.0 boost)
    """
    if not results:
        return results
    
    # Extract filter sets for fast lookup
    target_skills = set(s.lower() for s in (filters.get("skills") or []))
    expanded_skills = set(s.lower() for s in (filters.get("expanded_skills") or []))
    target_companies = set(c.lower() for c in (filters.get("companies") or []))
    expanded_companies = set(c.lower() for c in (filters.get("expanded_companies") or []))
    all_companies = target_companies | expanded_companies
    target_certifications = set(c.lower() for c in (filters.get("certifications") or []))
    target_city = (filters.get("city") or "").lower() if location_preference != "remote" else ""
    target_state = (filters.get("state") or "").lower() if location_preference != "remote" else ""
    target_country = (filters.get("country") or "").lower() if location_preference != "remote" else ""
    min_years = filters.get("min_years")
    max_years = filters.get("max_years")
    target_titles = set(t.lower() for t in (filters.get("titles") or []))
    # Build title keywords for fuzzy matching (e.g. "software engineer" -> {"software", "engineer"})
    title_keywords = set()
    for t in target_titles:
        title_keywords.update(w for w in t.split() if len(w) > 2)
    
    logger.info(f"Re-ranking {len(results)} candidates with: location_preference={location_preference}, "
               f"city='{target_city}', titles={list(target_titles)[:2]}, "
               f"skills={list(target_skills)[:3]}, certs={list(target_certifications)[:2]}, exp={min_years}-{max_years}")
    
    # Re-rank each result
    reranked = []
    for result in results:
        bonus = 0.0
        bonus_details = []
        all_matched_skills = []
        
        # ── 1. SKILLS MATCH ──────────────────────────────────────────
        profile_skills = set(s.lower() for s in (result.skills or []))
        
        # 1a. Exact skill matches in skills[] array (highest confidence)
        exact_matches = target_skills & profile_skills
        if target_skills:
            skill_coverage = len(exact_matches) / len(target_skills)
            skill_bonus = skill_coverage * 0.30
            bonus += skill_bonus
            all_matched_skills.extend(exact_matches)
            if skill_bonus > 0:
                bonus_details.append(f"skills+{skill_bonus:.2f}")
        
        # 1b. Skills mentioned in headline/description/title (lower confidence fallback)
        missing_skills = target_skills - exact_matches
        if missing_skills:
            profile_text = " ".join([
                (result.headline or ""),
                (result.current_title or ""),
                (result.description or ""),
            ]).lower()
            text_matched = set()
            for skill in missing_skills:
                if skill in profile_text:
                    text_matched.add(skill)
            if text_matched:
                text_bonus = (len(text_matched) / len(target_skills)) * 0.10  # Lower weight than array match
                bonus += text_bonus
                all_matched_skills.extend(text_matched)
                bonus_details.append(f"skills_text+{text_bonus:.2f}")
        
        # 1c. Expanded skill matches (related skills)
        expanded_matches = expanded_skills & profile_skills
        if expanded_skills and not exact_matches:
            expanded_coverage = len(expanded_matches) / len(expanded_skills)
            exp_bonus = expanded_coverage * 0.12
            bonus += exp_bonus
            all_matched_skills.extend(expanded_matches)
            if exp_bonus > 0:
                bonus_details.append(f"expanded+{exp_bonus:.2f}")
        
        # ── 2. LOCATION MATCH ────────────────────────────────────────
        if location_preference != "remote" and target_city:
            profile_location = (result.location or "").lower()
            profile_city = (result.city or "").lower()
            
            if location_preference == "must_match":
                city_boost, state_boost, country_boost = 3.0, 1.5, 0.75
            else:  # "preferred"
                city_boost, state_boost, country_boost = 0.80, 0.40, 0.25
            
            if target_city and (target_city in profile_location or target_city in profile_city):
                bonus += city_boost
                bonus_details.append(f"city+{city_boost:.2f}")
            elif target_state and target_state in profile_location:
                bonus += state_boost
                bonus_details.append(f"state+{state_boost:.2f}")
            elif target_country and target_country in profile_location:
                bonus += country_boost
                bonus_details.append(f"country+{country_boost:.2f}")
        
        # ── 3. TITLE MATCH (comprehensive) ────────────────────────────
        profile_title = (result.current_title or "").lower()
        profile_headline = (result.headline or "").lower()
        title_matched = False
        
        # 3a. Exact substring match in current_title
        for target_title in target_titles:
            if target_title in profile_title:
                bonus += 0.50
                bonus_details.append("title+0.50")
                title_matched = True
                break
        
        # 3b. Word overlap match (e.g. "software engineer" matches "senior software engineer")
        if not title_matched and title_keywords:
            title_words = set(w for w in profile_title.split() if len(w) > 2)
            overlap = title_keywords & title_words
            if overlap:
                overlap_ratio = len(overlap) / len(title_keywords)
                if overlap_ratio >= 0.5:  # At least half the words match
                    partial_bonus = overlap_ratio * 0.25
                    bonus += partial_bonus
                    bonus_details.append(f"title_partial+{partial_bonus:.2f}")
                    title_matched = True
        
        # 3c. Headline match (candidate's self-description often mentions their role)
        if not title_matched and profile_headline:
            for target_title in target_titles:
                if target_title in profile_headline:
                    bonus += 0.15
                    bonus_details.append("headline_title+0.15")
                    break
        
        # ── 4. COMPANY MATCH ──────────────────────────────────────────
        # 4a. Current company match
        profile_company = (result.current_company or "").lower()
        company_matched = False
        for target_company in all_companies:
            if target_company in profile_company:
                bonus += 0.25
                bonus_details.append("curr_company+0.25")
                company_matched = True
                break
        
        # 4b. Past company match (work history)
        if not company_matched and result.work_history:
            for work_exp in result.work_history[:5]:  # Check top 5 most recent
                past_company = (work_exp.get('company') or "").lower()
                for target_company in all_companies:
                    if target_company in past_company:
                        bonus += 0.15  # Lower than current company
                        bonus_details.append("past_company+0.15")
                        company_matched = True
                        break
                if company_matched:
                    break
        
        # ── 5. CERTIFICATION MATCH ────────────────────────────────────
        matched_certs = []
        if target_certifications and result.certifications:
            profile_certs = set(c.lower() for c in result.certifications)
            # Exact matches
            exact_cert_matches = target_certifications & profile_certs
            matched_certs.extend(exact_cert_matches)
            
            # Partial matches (e.g., "AWS" matches "AWS Certified Solutions Architect")
            if not exact_cert_matches:
                for target_cert in target_certifications:
                    for profile_cert in result.certifications:
                        if target_cert in profile_cert.lower():
                            matched_certs.append(profile_cert)
                            break
            
            if matched_certs:
                cert_coverage = len(matched_certs) / len(target_certifications)
                cert_bonus = cert_coverage * 0.20  # Up to +0.20 for all certs matched
                bonus += cert_bonus
                bonus_details.append(f"certs+{cert_bonus:.2f}")
        
        # ── 6. EXPERIENCE MATCH (with proximity scoring) ──────────────
        exp = result.years_experience
        if exp is not None and (min_years is not None or max_years is not None):
            effective_min = min_years if min_years is not None else 0
            effective_max = max_years if max_years is not None else 50
            
            if effective_min <= exp <= effective_max:
                # In range — full bonus
                bonus += 0.25
                bonus_details.append("exp+0.25")
            else:
                # Out of range — proximity bonus (closer = better)
                if min_years is not None and max_years is not None:
                    ideal = (min_years + max_years) / 2
                else:
                    ideal = min_years or max_years or 0
                diff = abs(exp - ideal)
                if diff <= 2:
                    bonus += 0.15
                    bonus_details.append("exp~+0.15")
                elif diff <= 5:
                    bonus += 0.08
                    bonus_details.append("exp~+0.08")
        
        # ── 7. PROFILE QUALITY BONUS ──────────────────────────────────
        # Reward complete profiles (they're more useful to recruiters)
        completeness = result.profile_completeness or 0
        if completeness > 70:
            quality_bonus = 0.05
            bonus += quality_bonus
        
        # ── FINAL SCORE ───────────────────────────────────────────────
        new_score = min(result.score + bonus, 1.0)
        
        # Log first 5 for debugging
        if len(reranked) < 5:
            logger.info(f"  Candidate: {result.full_name[:25]} | {(result.city or 'no-city')[:15]} | "
                       f"base={result.score:.3f} bonus={bonus:.3f} final={new_score:.3f} | {bonus_details}")
        
        # Create new result with updated score and comprehensive matched skills
        reranked.append(CandidateResultV2(
            forager_id=result.forager_id,
            score=round(new_score, 4),
            full_name=result.full_name,
            current_title=result.current_title,
            current_company=result.current_company,
            location=result.location,
            city=result.city,
            country=result.country,
            years_experience=result.years_experience,
            domain=result.domain,
            profile_completeness=result.profile_completeness,
            skills=result.skills,
            matched_skills=list(set(all_matched_skills)) if all_matched_skills else result.matched_skills,
            certifications=result.certifications,
            matched_certifications=list(set(matched_certs)),
            work_history=result.work_history,
            education=result.education,
            headline=result.headline,
            description=result.description,
            linkedin_url=result.linkedin_url,
            photo=result.photo,
            industry=result.industry,
            linkedin_slug=result.linkedin_slug,
            search_name=result.search_name,
            is_creator=result.is_creator,
            is_influencer=result.is_influencer,
            address=result.address,
            linkedin_country=result.linkedin_country,
            linkedin_area=result.linkedin_area,
            date_updated=result.date_updated,
            primary_locale=result.primary_locale,
            temporary_status=result.temporary_status,
            temporary_emoji_status=result.temporary_emoji_status,
            background_picture=result.background_picture,
            area=result.area
        ))
    
    # Sort by new score (highest first)
    reranked.sort(key=lambda x: x.score, reverse=True)
    
    # Log top 3 results for debugging
    if reranked:
        logger.info(f"Top 3 after re-ranking:")
        for i, r in enumerate(reranked[:3]):
            logger.info(f"  {i+1}. {r.full_name} | {r.current_title} | {r.city}, {r.country} | "
                       f"exp={r.years_experience}y | score={r.score:.3f} | skills={r.matched_skills[:3]}")
    
    return reranked


def calculate_location_facets(results: List[CandidateResultV2]) -> SearchFacets:
    """
    Calculate location facets from candidate results.
    Groups by location, counts candidates, excludes null/unknown.
    """
    location_counts = {}
    remote_count = 0
    
    for result in results:
        # Count remote candidates (if you have a field for this)
        # For now, we'll just track location facets
        
        # Use city if available, fallback to location string
        location_key = result.city or result.location or None
        
        if location_key and location_key.lower() not in ['', 'unknown', 'null', 'none']:
            location_counts[location_key] = location_counts.get(location_key, 0) + 1
    
    # Convert to LocationFacet list and sort by count descending
    location_facets = [
        LocationFacet(name=name, count=count)
        for name, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return SearchFacets(
        remote_available=True,  # Can be refined later
        remote_count=remote_count,
        locations=location_facets
    )


def filter_by_locations(results: List[CandidateResultV2], selected_locations: List[str]) -> List[CandidateResultV2]:
    """
    Hard filter results to only include candidates in selected locations.
    """
    if not selected_locations:
        return results
    
    selected_set = set(loc.lower() for loc in selected_locations)
    
    filtered = []
    for result in results:
        # Check both city and location fields
        result_city = (result.city or "").lower()
        result_location = (result.location or "").lower()
        
        # Include if either field matches any selected location
        if result_city in selected_set or any(loc in result_location for loc in selected_set):
            filtered.append(result)
    
    return filtered


# =============================================================================
# LEGACY API ENDPOINTS (for frontend compatibility)
# =============================================================================

@app.get("/api/stats")
async def legacy_get_stats():
    """Get basic statistics for the dashboard"""
    try:
        conn = get_db_connection()
        total = conn.execute("SELECT COUNT(*) FROM processed_profiles").fetchone()[0]
        countries = conn.execute("SELECT COUNT(DISTINCT canonical_country) FROM processed_profiles WHERE canonical_country IS NOT NULL AND canonical_country != ''").fetchone()[0]
        cities = conn.execute("SELECT COUNT(DISTINCT canonical_city) FROM processed_profiles WHERE canonical_city IS NOT NULL AND canonical_city != ''").fetchone()[0]
        industries = conn.execute("SELECT COUNT(DISTINCT primary_domain) FROM processed_profiles WHERE primary_domain IS NOT NULL AND primary_domain != ''").fetchone()[0]
        conn.close()
        return {
            "total_profiles": total,
            "total_countries": countries,
            "total_cities": cities, 
            "total_industries": industries
        }
    except Exception as e:
        logger.error(f"Error in legacy_get_stats: {e}")
        return {"total_profiles": 0, "total_countries": 0, "total_cities": 0, "total_industries": 0}

@app.get("/api/search")
async def legacy_search(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    skills: Optional[str] = None,
    skill: Optional[str] = None,
    location: Optional[str] = None,
    country: Optional[str] = None,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    # New filters from Classic Search
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    headline: Optional[str] = None,
    role: Optional[str] = None,
    education: Optional[str] = None,
    certification: Optional[str] = None,
    area: Optional[str] = None,
    quick_search: Optional[str] = None,
):
    """Legacy search endpoint - uses direct SQL for proper filtering"""
    try:
        conn = get_db_connection()
        
        # Build dynamic SQL with proper WHERE clauses
        where_clauses = []
        params = []
        joins = []
        
        # Quick search - searches across name, headline, title, company
        if quick_search:
            where_clauses.append("(pp.full_name ILIKE ? OR p.headline ILIKE ? OR pp.current_role_title ILIKE ? OR pp.current_role_company ILIKE ?)")
            params.extend([f"%{quick_search}%", f"%{quick_search}%", f"%{quick_search}%", f"%{quick_search}%"])
        
        # Name filters - search in full_name
        if first_name:
            where_clauses.append("pp.full_name ILIKE ?")
            params.append(f"{first_name}%")
        if last_name:
            where_clauses.append("pp.full_name ILIKE ?")
            params.append(f"%{last_name}")
        
        # Location filters
        if country:
            where_clauses.append("pp.canonical_country ILIKE ?")
            params.append(f"%{country}%")
        if city:
            where_clauses.append("pp.canonical_city ILIKE ?")
            params.append(f"%{city}%")
        if location:
            where_clauses.append("(pp.canonical_city ILIKE ? OR pp.canonical_state ILIKE ?)")
            params.extend([f"%{location}%", f"%{location}%"])
        if area:
            where_clauses.append("(pp.canonical_state ILIKE ? OR p.area ILIKE ?)")
            params.extend([f"%{area}%", f"%{area}%"])
        
        # Industry filter
        if industry:
            where_clauses.append("pp.primary_domain ILIKE ?")
            params.append(f"%{industry}%")
        
        # Headline filter
        if headline:
            where_clauses.append("(p.headline ILIKE ? OR pp.current_role_title ILIKE ?)")
            params.extend([f"%{headline}%", f"%{headline}%"])
        
        # Skills filter - support both 'skill' (singular, from frontend) and 'skills' (plural, for backward compat)
        active_skills = skill or skills
        if active_skills:
            skill_list = [s.strip() for s in active_skills.split(',')]
            for sk in skill_list:
                where_clauses.append("pp.canonical_skills ILIKE ?")
                params.append(f"%{sk}%")
        
        # Role filter - search in roles table
        if role:
            joins.append("LEFT JOIN roles r ON pp.person_id = r.forager_id")
            where_clauses.append("r.role_title ILIKE ?")
            params.append(f"%{role}%")
        
        # Education filter - search in educations table
        if education:
            joins.append("LEFT JOIN educations e ON pp.person_id = e.forager_id")
            where_clauses.append("(e.school_name ILIKE ? OR e.degree ILIKE ? OR e.field_of_study ILIKE ?)")
            params.extend([f"%{education}%", f"%{education}%", f"%{education}%"])
        
        # Certification filter - search in certifications table
        if certification:
            joins.append("LEFT JOIN certifications c ON pp.person_id = c.forager_id")
            where_clauses.append("c.certificate_name ILIKE ?")
            params.append(f"%{certification}%")
        
        # Build the query
        join_sql = " ".join(joins)
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get total count first
        count_sql = f"""
            SELECT COUNT(DISTINCT pp.person_id)
            FROM processed_profiles pp
            LEFT JOIN persons p ON pp.person_id = p.forager_id
            {join_sql}
            WHERE {where_sql}
        """
        total = conn.execute(count_sql, params).fetchone()[0]
        
        # Calculate pagination
        offset = (page - 1) * page_size
        total_pages = max(1, (total + page_size - 1) // page_size)
        
        # Get paginated results
        data_sql = f"""
            SELECT DISTINCT
                pp.person_id as forager_id,
                pp.full_name,
                pp.canonical_city as city,
                pp.canonical_state as state,
                pp.canonical_country as country,
                pp.primary_domain as domain,
                pp.years_experience,
                pp.profile_completeness,
                pp.canonical_skills as skills_raw,
                pp.current_role_title as current_title,
                pp.current_role_company as current_company,
                p.photo,
                p.headline,
                p.linkedin_url,
                p.linkedin_slug,
                p.industry,
                p.is_creator,
                p.is_influencer,
                p.search_name,
                p.description,
                p.address,
                p.linkedin_country,
                p.linkedin_area,
                p.date_updated,
                p.primary_locale,
                p.temporary_status,
                p.temporary_emoji_status,
                p.background_picture,
                p.area
            FROM processed_profiles pp
            LEFT JOIN persons p ON pp.person_id = p.forager_id
            {join_sql}
            WHERE {where_sql}
            ORDER BY 
                pp.profile_completeness DESC NULLS LAST, 
                pp.years_experience DESC NULLS LAST,
                pp.full_name ASC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(data_sql, params + [page_size, offset]).fetchall()
        
        # Build matched_skills tracking for each result
        search_skills_lower = set()
        if active_skills:
            search_skills_lower = set(s.strip().lower() for s in active_skills.split(','))
        
        # Build results
        columns = ['forager_id', 'full_name', 'city', 'state', 'country', 
                   'domain', 'years_experience', 'profile_completeness',
                   'skills_raw', 'current_title', 'current_company', 'photo', 
                   'headline', 'linkedin_url', 'linkedin_slug', 'industry',
                   'is_creator', 'is_influencer', 'search_name', 'description',
                   'address', 'linkedin_country', 'linkedin_area', 'date_updated',
                   'primary_locale', 'temporary_status', 'temporary_emoji_status',
                   'background_picture', 'area']
        
        # Collect forager_ids for batch fetching certifications and work history
        forager_ids = []
        results = []
        
        for row in rows:
            profile = dict(zip(columns, row))
            forager_ids.append(profile['forager_id'])
            # Parse skills
            skills_list = []
            if profile.get('skills_raw'):
                raw = profile['skills_raw']
                if isinstance(raw, list):
                    skills_list = raw
                elif isinstance(raw, str):
                    try:
                        import json
                        skills_list = json.loads(raw)
                    except:
                        skills_list = [s.strip() for s in raw.split(',') if s.strip()]
            
            # Build the full_name for first/last splitting
            full_name = profile.get('full_name') or 'Unknown'
            name_parts = full_name.split(' ')
            first = name_parts[0]
            last = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
            
            # Location string
            loc_parts = [p for p in [profile.get('city'), profile.get('state'), profile.get('country')] if p]
            location_str = ', '.join(loc_parts)
            
            # Calculate matched_skills
            matched_skills = []
            if search_skills_lower and skills_list:
                profile_skills_lower = set(s.lower() for s in skills_list)
                matched_skills = [s for s in skills_list if s.lower() in search_skills_lower]
            
            results.append({
                "forager_id": str(profile.get('forager_id', '')),
                "full_name": full_name,
                "first_name": first,
                "last_name": last,
                "score": 1.0,
                "current_title": profile.get('current_title'),
                "current_company": profile.get('current_company'),
                "location": location_str,
                "city": profile.get('city'),
                "country": profile.get('country'),
                "years_experience": profile.get('years_experience'),
                "domain": profile.get('domain'),
                "profile_completeness": profile.get('profile_completeness'),
                "skills": skills_list,
                "matched_skills": matched_skills,
                "headline": profile.get('headline'),
                "linkedin_url": profile.get('linkedin_url'),
                "photo": profile.get('photo'),
                "linkedin_slug": profile.get('linkedin_slug'),
                "industry": profile.get('industry'),
                "is_creator": profile.get('is_creator'),
                "is_influencer": profile.get('is_influencer'),
                "search_name": profile.get('search_name'),
                "description": profile.get('description'),
                "address": profile.get('address'),
                "linkedin_country": profile.get('linkedin_country'),
                "linkedin_area": profile.get('linkedin_area'),
                "date_updated": str(profile.get('date_updated', '')),
                "primary_locale": profile.get('primary_locale'),
                "temporary_status": profile.get('temporary_status'),
                "temporary_emoji_status": profile.get('temporary_emoji_status'),
                "background_picture": profile.get('background_picture'),
                "area": profile.get('area'),
                "certifications": [],  # Will be populated below
                "work_history": []  # Will be populated below
            })
        
        # Batch fetch certifications and work history
        if forager_ids:
            ids_str = ','.join(str(fid) for fid in forager_ids)
            
            # Fetch certifications
            cert_rows = conn.execute(f"""
                SELECT forager_id, certificate_name
                FROM certifications
                WHERE forager_id IN ({ids_str})
                ORDER BY issue_date DESC NULLS LAST
            """).fetchall()
            
            cert_map = {}
            for fid, cert_name in cert_rows:
                if fid not in cert_map:
                    cert_map[fid] = []
                if cert_name:
                    cert_map[fid].append(cert_name)
            
            # Fetch work history (top 5 most recent per person)
            work_rows = conn.execute(f"""
                SELECT forager_id, role_title, company_name, start_date, end_date
                FROM roles
                WHERE forager_id IN ({ids_str})
                ORDER BY start_date DESC NULLS LAST
                LIMIT 500
            """).fetchall()
            
            work_map = {}
            for fid, title, company, start_dt, end_dt in work_rows:
                if fid not in work_map:
                    work_map[fid] = []
                work_map[fid].append({
                    'title': title,
                    'company': company,
                    'start_date': str(start_dt) if start_dt else None,
                    'end_date': str(end_dt) if end_dt else None
                })
            
            # Add to results
            for result in results:
                fid = int(result['forager_id'])
                result['certifications'] = cert_map.get(fid, [])
                result['work_history'] = work_map.get(fid, [])[:5]  # Limit to 5
        
        conn.close()
        
        return {
            "data": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Error in legacy_search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/filters/skills")
async def legacy_get_skills(limit: int = 100):
    """Get top skills for filters"""
    try:
        from filter_service import get_skill_facets
        conn = get_db_connection()
        facets = get_skill_facets(conn, limit=limit)
        conn.close()
        return [{"value": f.value, "label": f.label, "count": f.count} for f in facets]
    except Exception as e:
        logger.error(f"Error in legacy_get_skills: {e}")
        return []


@app.get("/api/filters/countries")
async def legacy_get_countries():
    """Get all countries"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT DISTINCT canonical_country, COUNT(*) as cnt
            FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != ''
            GROUP BY canonical_country
            ORDER BY cnt DESC
        """).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_countries: {e}")
        return []


@app.get("/api/filters/cities")
async def legacy_get_cities(country: Optional[str] = None):
    """Get cities, optionally filtered by country"""
    try:
        conn = get_db_connection()
        if country:
            results = conn.execute("""
                SELECT DISTINCT canonical_city, COUNT(*) as cnt
                FROM processed_profiles
                WHERE canonical_country = ? 
                  AND canonical_city IS NOT NULL 
                  AND canonical_city != ''
                GROUP BY canonical_city
                ORDER BY cnt DESC
            """, [country]).fetchall()
        else:
            results = conn.execute("""
                SELECT DISTINCT canonical_city, COUNT(*) as cnt
                FROM processed_profiles
                WHERE canonical_city IS NOT NULL AND canonical_city != ''
                GROUP BY canonical_city
                ORDER BY cnt DESC
                LIMIT 200
            """).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_cities: {e}")
        return []


@app.get("/api/filters/locations")
async def legacy_get_locations():
    """Get all unique locations"""
    return await legacy_get_cities()


@app.get("/api/filters/industries")
async def legacy_get_industries():
    """Get all industries"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT DISTINCT primary_domain, COUNT(*) as cnt
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != ''
            GROUP BY primary_domain
            ORDER BY cnt DESC
            LIMIT 100
        """).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_industries: {e}")
        return []


@app.get("/api/filters/roles")
async def legacy_get_roles():
    """Get all role titles"""
    try:
        conn = get_db_connection()
        # Corrected: title -> role_title
        results = conn.execute("""
            SELECT role_title, COUNT(*) as cnt
            FROM roles
            WHERE role_title IS NOT NULL AND role_title != ''
            GROUP BY role_title
            ORDER BY cnt DESC
            LIMIT 100
        """).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_roles: {e}")
        return []


@app.get("/api/filters/schools")
async def legacy_get_schools():
    """Get all schools"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT school_name, COUNT(*) as cnt
            FROM educations
            WHERE school_name IS NOT NULL AND school_name != ''
            GROUP BY school_name
            ORDER BY cnt DESC
            LIMIT 100
        """).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_schools: {e}")
        return []


@app.get("/api/filters/certifications")
async def legacy_get_certifications():
    """Get all certifications"""
    try:
        conn = get_db_connection()
        # Corrected: certification_name -> certificate_name
        results = conn.execute("""
            SELECT certificate_name, COUNT(*) as cnt
            FROM certifications
            WHERE certificate_name IS NOT NULL AND certificate_name != ''
            GROUP BY certificate_name
            ORDER BY cnt DESC
            LIMIT 100
        """).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_certifications: {e}")
        return []


@app.get("/api/analytics/countries")
async def legacy_analytics_countries(limit: int = 10):
    """Get candidate count by country for charts"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT canonical_country as country, COUNT(*) as count
            FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != ''
            GROUP BY canonical_country
            ORDER BY count DESC
            LIMIT ?
        """, [limit]).fetchall()
        conn.close()
        return [{"country": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_analytics_countries: {e}")
        return []


@app.get("/api/analytics/industries")
async def legacy_analytics_industries(limit: int = 10):
    """Get candidate count by industry for charts"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT primary_domain, COUNT(*) as count
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != ''
            GROUP BY primary_domain
            ORDER BY count DESC
            LIMIT ?
        """, [limit]).fetchall()
        conn.close()
        return [{"industry": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_analytics_industries: {e}")
        return []


@app.get("/api/person/{forager_id}/certifications")
async def legacy_get_person_certifications(forager_id: str):
    """Get certifications for a specific person"""
    try:
        conn = get_db_connection()
        # Updated to include authority and url, and CORRECTED column name to certificate_name
        results = conn.execute("""
            SELECT certificate_name, authority, issued_date, certificate_url
            FROM certifications
            WHERE forager_id = ?
            ORDER BY issued_date DESC
        """, [forager_id]).fetchall()
        conn.close()
        return [{
            "name": row[0], 
            "certificate_name": row[0],
            "authority": row[1], 
            "start": row[2], 
            "issued_date": row[2],
            "certificate_url": row[3],
            "url": row[3]
        } for row in results]
    except Exception as e:
        logger.error(f"Error in legacy_get_person_certifications: {e}")
        return []


@app.get("/api/person/{forager_id}/educations")
async def legacy_get_person_educations(forager_id: str):
    """Get education history for a specific person"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT school_name, degree, field_of_study, start_date, end_date
            FROM educations
            WHERE forager_id = ?
            ORDER BY start_date DESC
        """, [forager_id]).fetchall()
        conn.close()
        return [{
            "school": row[0],
            "school_name": row[0],
            "degree": row[1],
            "field": row[2],
            "field_of_study": row[2],
            "start_date": row[3],
            "end_date": row[4]
        } for row in results]
    except Exception as e:
        logger.error(f"Error in legacy_get_person_educations: {e}")
        return []


@app.get("/api/person/{forager_id}/roles")
async def legacy_get_person_roles(forager_id: str):
    """Get role history for a specific person"""
    try:
        conn = get_db_connection()
        # Corrected column names based on schema inspection: organization_name, role_title
        # Added description
        results = conn.execute("""
            SELECT organization_name, role_title, start_date, end_date, is_current, description
            FROM roles
            WHERE forager_id = ?
            ORDER BY start_date DESC
        """, [forager_id]).fetchall()
        conn.close()
        return [{
            "company": row[0],
            "organization_name": row[0],
            "title": row[1],
            "role_title": row[1],
            "start_date": row[2],
            "end_date": row[3],
            "is_current": row[4],
            "description": row[5]
        } for row in results]
    except Exception as e:
        logger.error(f"Error in legacy_get_person_roles: {e}")
        return []


@app.get("/api/person/{forager_id}/skills")
async def legacy_get_person_skills(forager_id: str):
    """Get skills for a specific person"""
    try:
        conn = get_db_connection()
        results = conn.execute("""
            SELECT skills
            FROM skills
            WHERE forager_id = ?
        """, [forager_id]).fetchall()
        conn.close()
        return [{"skill": row[0]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_person_skills: {e}")
        return []


@app.get("/api/export/excel")
async def legacy_export_excel(
    limit: int = 1000,
    skills: Optional[str] = None,
    location: Optional[str] = None,
):
    """Export search results to Excel (placeholder)"""
    raise HTTPException(status_code=501, detail="Excel export not yet implemented in v2 API")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    print("=" * 62)
    print("  TALENTIN SEARCH API v2 - SMART SEARCH")
    print("=" * 62)
    print(f"  Qdrant: {config.qdrant_url[:45]}")
    print(f"  DuckDB: {config.duckdb_path[:45]}")
    print(f"  Cloud:  {str(config.is_cloud)}")
    print("=" * 62)
    print("  Endpoints:")
    print("    POST /api/v2/search         Structured search")
    print("    POST /api/v2/smart-search   AI Natural Language Search")
    print("    GET  /api/v2/health         Health check")
    print("    GET  /api/v2/stats          Statistics")
    print("    GET  /api/v2/expand-skills  Test skill expansion")
    print("=" * 62)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
