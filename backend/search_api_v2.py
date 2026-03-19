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
    SparseVector, Prefetch, FusionQuery, Fusion,
    IsNullCondition, PayloadField
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

# Shared certification alias map (used in both search_v2 filtering and smart_rerank)
CERT_ALIASES: Dict[str, List[str]] = {
    'cpa': ['cpa', 'certified public accountant', 'uscpa', 'us cpa'],
    'uscpa': ['uscpa', 'us cpa', 'cpa', 'certified public accountant'],
    'pmp': ['pmp', 'project management professional'],
    'cfa': ['cfa', 'chartered financial analyst'],
    'cissp': ['cissp'],
    'six sigma': ['six sigma', '6 sigma'],
    'aws certified': ['aws certified', 'aws solutions architect', 'aws developer'],
}


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
                try:
                    _embedding_model = SentenceTransformer(config.embedding_model)
                    logger.info(f"Embedding model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise RuntimeError(f"Could not load embedding model '{config.embedding_model}': {e}")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """Lazy load Qdrant client (thread-safe)"""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:  # Double-check pattern
                config = get_config()
                try:
                    if config.is_cloud:
                        logger.info(f"Connecting to Qdrant Cloud: {config.qdrant_url}")
                        
                        api_key = config.qdrant_api_key
                        if not api_key:
                            logger.warning("API Key missing in config! Attempting direct refetch from .env...")
                            import os
                            from dotenv import load_dotenv
                            load_dotenv(override=True)
                            api_key = os.getenv("QDRANT_API_KEY")
                        
                        if not api_key:
                            raise ValueError("CRITICAL: Qdrant API Key is MISSING! Check QDRANT_API_KEY in .env file")
                        
                        logger.info("Qdrant API Key: configured")
                        
                        _qdrant_client = QdrantClient(
                            url=config.qdrant_url,
                            api_key=api_key,
                            timeout=30
                        )
                    else:
                        logger.info(f"Using local Qdrant: {config.qdrant_url}")
                        _qdrant_client = QdrantClient(path=config.qdrant_url)
                    
                    logger.info("Qdrant client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Qdrant client: {e}")
                    raise
    return _qdrant_client


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Get DuckDB connection"""
    config = get_config()
    try:
        conn = duckdb.connect(config.duckdb_path, read_only=True)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to DuckDB at {config.duckdb_path}: {e}")
        raise RuntimeError(f"Could not connect to DuckDB: {e}")


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
            config = get_config()
            min_sim = config.skill_expansion_min_similarity
            rows = conn.execute("""
                SELECT skill_a, skill_b, semantic_confidence
                FROM skill_relationships
                WHERE semantic_confidence >= ?
                ORDER BY semantic_confidence DESC
            """, [min_sim]).fetchall()
            
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

def build_qdrant_filter(query: ParsedQueryV2, client_id: Optional[str] = None) -> Optional[Filter]:
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
    
# Skills - HARD FILTER: Require at least 1 matching skill from must_have
    config = get_config()
    if query.filters.skills.must_have and len(query.filters.skills.must_have) > 0:
        expanded_must_have, _ = expand_skills(query.filters.skills.must_have)
        if expanded_must_have:
            must_conditions.append(
                FieldCondition(
                    key="skills",
                    match=MatchAny(any=expanded_must_have)
                )
            )
            logger.info(f"Applied skills hard filter: {expanded_must_have[:5]}...")
    
    # Skills - exclude (NOT)
    for skill in filters.skills.exclude:
        skill_variations = expand_skill_search([skill])
        skill_variations = with_case_variations(skill_variations)
        must_not_conditions.append(
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
    
    # Domain: NOT used as a hard Qdrant filter — domain data is sparse/inconsistently
    # tagged on profiles. Instead, domain is added to search_text for semantic matching.
    # (see smart_search_endpoint where domain is appended to enriched search text)
    
    # Experience range — NOT used as hard Qdrant filter.
    # Reason: many profiles have inaccurate or missing years_experience in Qdrant
    # payload. Hard-filtering here eliminates good candidates before they can be
    # rescued by the ranking layer. Experience is instead handled as a ranking
    # bonus in smart_rerank() so profiles in-range float to the top.


    # Client isolation — restrict Qdrant results to this client's candidate data
    # NOTE: ~227K legacy points (client 00) lack a client_id field in Qdrant payload
    # (ingested before multi-tenant support). For client '00', we use OR logic to
    # include points with client_id='00' OR points missing the field entirely.
    if client_id:
        if client_id == '00':
            must_conditions.append(
                Filter(should=[
                    FieldCondition(key="client_id", match=MatchValue(value=client_id)),
                    IsNullCondition(is_null=PayloadField(key="client_id"))
                ])
            )
        else:
            must_conditions.append(
                FieldCondition(key="client_id", match=MatchValue(value=client_id))
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
        cache_dir = base_dir / "database" / "sparse_cache"
        
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
                    logger.debug(f"Sparse encoder not available (dense-only mode): {e}")
            
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
    
    if config.qdrant_collection in collections:
        # Hybrid search with RRF fusion
        try:
            results = qdrant.query_points(
                collection_name=config.qdrant_collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=max(config.prefetch_limit, limit * 3),
                        filter=qdrant_filter
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using="sparse",
                        limit=max(config.prefetch_limit, limit * 3),
                        filter=qdrant_filter
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=max(config.fusion_limit, limit * 2),
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
    
    # Extract forager_ids and normalize RRF scores to 0-1 range
    raw_scores = []
    for hit in results:
        fid = hit.payload.get("forager_id") or hit.payload.get("person_id")
        if fid:
            raw_scores.append((int(fid), float(hit.score)))

    if not raw_scores:
        return []

    # Normalize: map [min_score, max_score] → [0.10, 0.65]
    # This leaves headroom for ranking bonuses to push top candidates toward 1.0
    scores_only = [s for _, s in raw_scores]
    min_s, max_s = min(scores_only), max(scores_only)
    spread = max_s - min_s if max_s > min_s else 1.0
    id_scores = [
        (fid, 0.10 + 0.55 * (score - min_s) / spread)
        for fid, score in raw_scores
    ]

    return id_scores


# =============================================================================
# DUCKDB HYDRATION
# =============================================================================

def _safe_ids_str(ids: List[int]) -> str:
    """Validate all IDs are integers and return a comma-separated string for SQL IN clause.
    Prevents SQL injection if IDs are ever sourced from user input."""
    return ','.join(str(int(i)) for i in ids)


def hydrate_profiles_from_duckdb(person_ids: List[int], client_id: Optional[str] = None) -> Dict[int, Dict]:
    """
    Hydrate full profile details from DuckDB for given person IDs.
    Returns dict of {person_id: profile_dict}
    """
    if not person_ids:
        return {}
    
    conn = get_db_connection()
    try:
        ids_str = _safe_ids_str(person_ids)

        # Client isolation — only return profiles belonging to this client
        client_clause = "AND pp.client_id = ?" if client_id else ""
        client_params = [client_id] if client_id else []
        
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
            WHERE pp.person_id IN ({ids_str}) {client_clause}
        """, client_params).fetchall()
        
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
            SELECT forager_id, certificate_name, start, "end"
            FROM certifications
            WHERE forager_id IN ({ids_str})
            ORDER BY start DESC NULLS LAST
        """).fetchall()
        
        for fid, cert_name, cert_start, cert_end in cert_rows:
            if fid in profiles and cert_name:
                profiles[fid]['certifications'].append(cert_name)
        
        # Fetch work history (past roles)
        roles_rows = conn.execute(f"""
            SELECT forager_id, role_title, organization_name, start_date, end_date,
                   description
            FROM roles
            WHERE forager_id IN ({ids_str})
            ORDER BY start_date DESC NULLS LAST
            LIMIT 1000
        """).fetchall()
        
        for fid, title, company, start_dt, end_dt, desc in roles_rows:
            if fid in profiles:
                profiles[fid]['work_history'].append({
                    'title': title,
                    'company': company,
                    'start_date': str(start_dt) if start_dt else None,
                    'end_date': str(end_dt) if end_dt else None,
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
    PASS-1 ranking bonus applied inside search_v2().
    Lightweight — only computes skill matching and matched_skills list.
    The heavy re-ranking (titles, companies, location, experience, combo)
    is done by smart_rerank() to avoid double-counting.
    """
    config = get_config()
    bonus = 0.0
    matched_skills = []

    profile_skills = set(s.lower() for s in (profile.get('skills') or []))

    # Helper: word-boundary skill match (prevents "ML" matching "Email")
    def match_skill(query_skill: str, profile_skills_set: set) -> Optional[str]:
        q = query_skill.lower()
        for ps in profile_skills_set:
            # Exact match
            if q == ps:
                return ps
            # Query is a multi-word phrase contained in profile skill
            if ' ' in q and q in ps:
                return ps
            # Profile skill is a multi-word phrase containing query
            if ' ' in ps and q in ps:
                return ps
            # Single-token query: only match if it's a distinct token in profile skill
            if ' ' not in q and len(q) >= 3:
                ps_tokens = set(ps.replace('-', ' ').replace('.', ' ').split())
                if q in ps_tokens:
                    return ps
        return None

    # --- Must-have skills ---
    must_have = [s.lower() for s in query.filters.skills.must_have]
    for skill in must_have:
        if match_skill(skill, profile_skills):
            matched_skills.append(skill)

    if must_have:
        coverage = len(matched_skills) / len(must_have)
        # Light bonus — smart_rerank handles comprehensive skill scoring
        bonus += coverage * 0.08

    # --- Nice-to-have skills ---
    nice_to_have = [s.lower() for s in query.filters.skills.nice_to_have]
    nice_matched = []
    for skill in nice_to_have:
        if match_skill(skill, profile_skills):
            nice_matched.append(skill)
    matched_skills.extend(nice_matched)
    if nice_to_have:
        bonus += (len(nice_matched) / len(nice_to_have)) * 0.04

    # --- Profile completeness (small baseline bonus) ---
    completeness = profile.get('profile_completeness') or 0
    bonus += (completeness / 100) * config.bonus_profile_completeness

    final_score = base_score + bonus
    return final_score, list(set(matched_skills))


# =============================================================================
# MAIN SEARCH FUNCTION
# =============================================================================

# Maximum query length to prevent DoS
MAX_QUERY_LENGTH = 10000

async def search_v2(query: ParsedQueryV2, client_id: Optional[str] = None) -> SearchResponseV2:
    """
    Main search function implementing the full flow:
    1. Expand skills
    2. Build Qdrant filter (scoped to client_id)
    3. Query Qdrant
    4. Hydrate from DuckDB (scoped to client_id)
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
    
    # 3. Build Qdrant filter (scoped to client)
    t0 = time.time()
    qdrant_filter = build_qdrant_filter(query, client_id=client_id)
    timings['filter_build'] = int((time.time() - t0) * 1000)
    
    # DEBUG: Log the actual filter being sent to Qdrant
    logger.info(f"Qdrant filter: {qdrant_filter}")
    logger.info(f"Search text for embedding: '{search_text[:100]}'")
    
    # 4. Query Qdrant
    # CRITICAL: Fetch a large pool of candidates so the reranker has enough
    # to work with.  More candidates = better location / skill / title coverage.
    SEARCH_FETCH_LIMIT = 3000
    t0 = time.time()
    try:
        id_scores = search_qdrant_hybrid(search_text, qdrant_filter, SEARCH_FETCH_LIMIT)
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    timings['qdrant_search'] = int((time.time() - t0) * 1000)
    
    # 5. Hydrate from DuckDB (scoped to client)
    t0 = time.time()
    forager_ids = [fid for fid, _ in id_scores]
    profiles = hydrate_profiles_from_duckdb(forager_ids, client_id=client_id)
    timings['duckdb_fetch'] = int((time.time() - t0) * 1000)
    
    # 6. Apply ranking and build results
    t0 = time.time()
    results = []
    exclude_skills = set(s.lower() for s in query.filters.skills.exclude)
    target_certifications = set(c.lower() for c in query.filters.certifications)
    
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
        
        # Filter by certifications if specified (soft match across all text fields)
        if target_certifications:
            # Build a blob of all text from this profile to search in
            cert_text_parts = []
            cert_text_parts += [c.lower() for c in (profile.get('certifications') or [])]
            cert_text_parts.append((profile.get('headline') or '').lower())
            cert_text_parts.append((profile.get('description') or '').lower())
            cert_text_parts.append((profile.get('job_title') or '').lower())
            cert_text_parts.append((profile.get('search_name') or '').lower())
            for wh in (profile.get('work_history') or []):
                cert_text_parts.append((wh.get('title') or '').lower())
                cert_text_parts.append((wh.get('description') or '').lower())
            cert_blob = ' '.join(cert_text_parts)

            has_cert_match = False
            for target_cert in target_certifications:
                aliases = CERT_ALIASES.get(target_cert, [target_cert])
                for alias in aliases:
                    if alias in cert_blob:
                        has_cert_match = True
                        break
                if has_cert_match:
                    break
            if not has_cert_match:
                continue  # Skip profiles without any cert match
            
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
            state=profile.get('state'),
            country=profile.get('country'),
            years_experience=profile.get('years_experience'),
            domain=profile.get('domain'),
            profile_completeness=profile.get('profile_completeness'),
            skills=profile.get('skills', []),
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
    try:
        config = get_config()
        logger.info("=" * 60)
        logger.info("  TALENTIN SEARCH API v2")
        logger.info("=" * 60)
        logger.info(f"Qdrant: {config.qdrant_url}")
        logger.info(f"DuckDB: {config.duckdb_path}")
        logger.info(f"Cloud Mode: {config.is_cloud}")
        logger.info("=" * 60)
        
        # Pre-load models with error handling
        try:
            get_embedding_model()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        try:
            load_skill_relations()
            logger.info("Skill relations loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load skill relations: {e}")
            # Don't fail startup if skill relations can't be loaded
        
        try:
            # Test Qdrant connection
            qdrant = get_qdrant_client()
            collections = qdrant.get_collections()
            logger.info(f"Qdrant connected successfully, {len(collections.collections)} collections found")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # Validate sparse encoder (BM25) availability
        try:
            encoder = get_sparse_encoder()
            if not encoder.vocabulary:
                logger.warning("⚠️  SPARSE ENCODER HAS NO VOCABULARY — running in dense-only mode. "
                              "Keyword matching (BM25) is disabled. Run backfill_qdrant.py to generate sparse cache.")
            else:
                logger.info(f"Sparse encoder loaded: {len(encoder.vocabulary)} tokens")
        except Exception as e:
            logger.warning(f"⚠️  Sparse encoder unavailable: {e} — keyword matching disabled")
        
        logger.info("=" * 60)
        logger.info("  STARTUP COMPLETE - Ready to serve requests")
        logger.info("=" * 60)
        
        yield
        
        logger.info("Shutting down...")
    except Exception as e:
        logger.critical(f"STARTUP FAILED: {e}")
        raise


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

# Mediator backend origin — set MEDIATOR_ORIGIN in .env if the mediator
# frontend/backend lives on a different domain that needs browser CORS.
# For server-to-server calls this is NOT required (CORS is browser-only).
_mediator_origin = os.getenv("MEDIATOR_ORIGIN", "").strip()
if _mediator_origin:
    ALLOWED_ORIGINS.append(_mediator_origin)
    logger.info(f"CORS: Mediator origin added: {_mediator_origin}")

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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Client-ID", "X-Request-ID", "X-Integration-Key"],
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
async def search_endpoint(query: ParsedQueryV2, request: Request):
    """
    Main search endpoint.
    Accepts parsed query JSON from external LLM API.
    Returns ranked candidates with modified skills.
    """
    # Resolve client_id: body field takes priority, then header-based auth
    from client_auth import require_client_id as _require_client_id, get_client_id as _get_client_id
    client_id = query.client_id
    if not client_id:
        client_id = _get_client_id(request)
    if not client_id:
        from config import get_config as _gc
        client_id = _gc().default_client_id
    response = await search_v2(query, client_id=client_id)
    
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



# =============================================================================
# CONSOLIDATED DASHBOARD ENDPOINT
# One request → stats + chart data + all filter options, fully client-scoped.
# Frontend should call GET /api/dashboard instead of the 10+ separate endpoints.
# =============================================================================

@app.get("/api/dashboard")
async def dashboard(request: Request):
    """
    Single consolidated endpoint for all dashboard data.
    Returns stats, chart data, and filter dropdowns in one shot.
    All data is strictly scoped to the requesting client.
    """
    from client_auth import require_client_id as _req
    from collections import Counter
    client_id = _req(request)
    conn = get_db_connection()
    try:
        p = [client_id]  # reusable param list

        total = conn.execute(
            "SELECT COUNT(*) FROM processed_profiles WHERE client_id = ?", p
        ).fetchone()[0]

        unique_countries = conn.execute("""
            SELECT COUNT(DISTINCT canonical_country) FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != '' AND client_id = ?
        """, p).fetchone()[0]

        unique_cities = conn.execute("""
            SELECT COUNT(DISTINCT canonical_city) FROM processed_profiles
            WHERE canonical_city IS NOT NULL AND canonical_city != '' AND client_id = ?
        """, p).fetchone()[0]

        # Chart: top 10 countries
        top_countries = conn.execute("""
            SELECT canonical_country AS country, COUNT(*) AS count
            FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != '' AND client_id = ?
            GROUP BY canonical_country ORDER BY count DESC LIMIT 10
        """, p).fetchall()

        # Chart: top 10 industries
        top_industries = conn.execute("""
            SELECT primary_domain AS industry, COUNT(*) AS count
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != '' AND client_id = ?
            GROUP BY primary_domain ORDER BY count DESC LIMIT 10
        """, p).fetchall()

        # Filters
        f_countries = conn.execute("""
            SELECT canonical_country, COUNT(*) AS cnt FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != '' AND client_id = ?
            GROUP BY canonical_country ORDER BY cnt DESC
        """, p).fetchall()

        f_cities = conn.execute("""
            SELECT canonical_city, COUNT(*) AS cnt FROM processed_profiles
            WHERE canonical_city IS NOT NULL AND canonical_city != '' AND client_id = ?
            GROUP BY canonical_city ORDER BY cnt DESC LIMIT 300
        """, p).fetchall()

        f_industries = conn.execute("""
            SELECT primary_domain, COUNT(*) AS cnt FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != '' AND client_id = ?
            GROUP BY primary_domain ORDER BY cnt DESC LIMIT 100
        """, p).fetchall()

        f_roles = conn.execute("""
            SELECT r.role_title, COUNT(*) AS cnt
            FROM roles r INNER JOIN processed_profiles pp ON r.forager_id = pp.person_id
            WHERE r.role_title IS NOT NULL AND r.role_title != '' AND pp.client_id = ?
            GROUP BY r.role_title ORDER BY cnt DESC LIMIT 100
        """, p).fetchall()

        f_schools = conn.execute("""
            SELECT e.school_name, COUNT(*) AS cnt
            FROM educations e INNER JOIN processed_profiles pp ON e.forager_id = pp.person_id
            WHERE e.school_name IS NOT NULL AND e.school_name != '' AND pp.client_id = ?
            GROUP BY e.school_name ORDER BY cnt DESC LIMIT 100
        """, p).fetchall()

        f_certs = conn.execute("""
            SELECT c.certificate_name, COUNT(*) AS cnt
            FROM certifications c INNER JOIN processed_profiles pp ON c.forager_id = pp.person_id
            WHERE c.certificate_name IS NOT NULL AND c.certificate_name != '' AND pp.client_id = ?
            GROUP BY c.certificate_name ORDER BY cnt DESC LIMIT 100
        """, p).fetchall()

        # Skills from canonical_skills column (avoid expensive unnest join)
        skill_rows = conn.execute("""
            SELECT canonical_skills FROM processed_profiles
            WHERE canonical_skills IS NOT NULL AND client_id = ?
        """, p).fetchall()
    finally:
        conn.close()

    # Aggregate skills
    skill_counter: Counter = Counter()
    for (skills_raw,) in skill_rows:
        if skills_raw:
            lst = skills_raw if isinstance(skills_raw, list) else []
            for sk in lst:
                if sk:
                    skill_counter[sk.strip()] += 1

    def fmt(rows):
        return [{"value": r[0], "label": r[0], "count": r[1]} for r in rows if r[0]]

    return {
        "client_id": client_id,
        "stats": {
            "total_records": total,
            "unique_countries": unique_countries,
            "unique_cities": unique_cities,
            "total_skills": sum(skill_counter.values()),
        },
        "charts": {
            "countries": [{"country": r[0], "count": r[1]} for r in top_countries if r[0]],
            "industries": [{"industry": r[0], "count": r[1]} for r in top_industries if r[0]],
        },
        "filters": {
            "countries":     fmt(f_countries),
            "cities":        fmt(f_cities),
            "industries":    fmt(f_industries),
            "roles":         fmt(f_roles),
            "schools":       fmt(f_schools),
            "certifications":fmt(f_certs),
            "skills":        [{"value": sk, "label": sk, "count": cnt} for sk, cnt in skill_counter.most_common(100) if sk],
        },
    }


@app.get("/api/stats")  # Legacy alias
@app.get("/api/v2/stats")
async def stats(request: Request):
    """Get database statistics — use /api/dashboard for the full data in one call."""
    from client_auth import require_client_id as _req
    client_id = _req(request)
    p = [client_id]

    conn = get_db_connection()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM processed_profiles WHERE client_id = ?", p
        ).fetchone()[0]

        unique_countries = conn.execute("""
            SELECT COUNT(DISTINCT canonical_country) FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != '' AND client_id = ?
        """, p).fetchone()[0]

        unique_cities = conn.execute("""
            SELECT COUNT(DISTINCT canonical_city) FROM processed_profiles
            WHERE canonical_city IS NOT NULL AND canonical_city != '' AND client_id = ?
        """, p).fetchone()[0]

        unique_industries = conn.execute("""
            SELECT COUNT(DISTINCT primary_domain) FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != '' AND client_id = ?
        """, p).fetchone()[0]

        total_skills = conn.execute("""
            SELECT COUNT(*) FROM skills s
            INNER JOIN processed_profiles pp ON s.forager_id = pp.person_id
            WHERE pp.client_id = ?
        """, p).fetchone()[0]

        domains = conn.execute("""
            SELECT primary_domain, COUNT(*) as count FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND client_id = ?
            GROUP BY primary_domain ORDER BY count DESC
        """, p).fetchall()

        return {
            "total_records": total,
            "total_profiles": total,
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
async def facets_endpoint(request: Request, current_filters: Optional[str] = None):
    """
    Get filter facets with counts, scoped to the requesting client.
    
    Returns top skills, locations, companies, experience ranges, and domains
    with counts based on current filter state.
    
    Args:
        current_filters: Optional JSON string of active filters
            Example: {"location": {"city": "Tokyo"}}
    
    Returns:
        FacetsResponse with all facet types and counts
    """
    import json
    from client_auth import require_client_id as _req
    client_id = _req(request)
    
    filters_dict = None
    if current_filters:
        try:
            filters_dict = json.loads(current_filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in current_filters")
    
    return get_all_facets(filters_dict, client_id=client_id)


@app.get("/api/v2/autocomplete/{field}", response_model=AutocompleteResponse)
async def autocomplete_endpoint(request: Request, field: str, q: str, limit: int = 10):
    """
    Real-time autocomplete for filter fields, scoped to the requesting client.
    
    Args:
        field: Field to autocomplete (skills, companies, locations, titles)
        q: Search query (prefix)
        limit: Max suggestions (default 10)
    
    Returns:
        AutocompleteResponse with matching suggestions and counts
    
    Example:
        GET /api/v2/autocomplete/skills?q=pyth&limit=5
    """
    from client_auth import require_client_id as _req
    client_id = _req(request)

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
    
    return autocomplete(field, q, min(limit, 50), client_id=client_id)


@app.get("/api/v2/filters/metadata", response_model=FilterMetadataResponse)
async def filter_metadata_endpoint(request: Request):
    """
    Get metadata about all available filters, scoped to the requesting client.
    
    Returns:
        - Total profile count
        - Unique counts for each filter type
        - Top 10 values for skills
        - Experience min/max/avg
    """
    from client_auth import require_client_id as _req
    client_id = _req(request)
    return get_filter_metadata(client_id=client_id)


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
    limit: int = 50  # Increased default for more results
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
        
        # Enrich search_text with titles + skills + location + domain for semantic matching
        # (these are NOT hard Qdrant filters — they rely on embedding similarity)
        enriched = parsed_result.search_text or request.query
        if parsed_result.filters.job_titles:
            enriched += " " + " ".join(parsed_result.filters.job_titles)
        if parsed_result.filters.skills.must_have:
            # Skills are no longer hard Qdrant filters — ensure they're in semantic text
            enriched += " " + " ".join(parsed_result.filters.skills.must_have)
        if parsed_result.filters.skills.nice_to_have:
            enriched += " " + " ".join(parsed_result.filters.skills.nice_to_have[:5])
        if parsed_result.filters.domain:
            # Add domain keywords to semantic search instead of using as hard filter
            domain_keywords = parsed_result.filters.domain.replace("_", " ")
            enriched += " " + domain_keywords
            parsed_result.filters.domain = None  # Clear so it doesn't hit Qdrant filter
        if parsed_result.filters.certifications:
            # Add cert keywords to semantic text too
            enriched += " " + " ".join(parsed_result.filters.certifications)
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
        # Expand skills & companies so smart_rerank() can award expanded-match bonuses
        _all_skills_openai = parsed_result.filters.skills.must_have + parsed_result.filters.skills.nice_to_have
        _skill_exp_openai = expand_skills(_all_skills_openai) if _all_skills_openai else {}
        _exp_skill_list_openai = list({s for related in _skill_exp_openai.values() for s in related})
        from normalizers import expand_company_search as _expand_co
        _exp_company_list_openai = _expand_co(parsed_result.filters.companies.worked_at) if parsed_result.filters.companies.worked_at else []

        smart_filters = {
            "skills": _all_skills_openai,
            "expanded_skills": _exp_skill_list_openai,
            "companies": parsed_result.filters.companies.worked_at,
            "expanded_companies": _exp_company_list_openai,
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
    
    # 3. Execute search with parsed query (scoped to client)
    # Resolve client_id: body field on parsed query takes priority, then header
    from client_auth import get_client_id as _get_cid
    from fastapi import HTTPException
    if not http_request:
        raise HTTPException(status_code=403, detail="Cannot determine client context.")
    client_id = getattr(parsed_result, 'client_id', None) or _get_cid(http_request)
    if not client_id:
        from config import get_config as _gc
        client_id = _gc().default_client_id
    logger.info(f"🔍 Executing Qdrant search: text='{parsed_result.search_text[:60]}...', "
               f"skills={parsed_result.filters.skills.must_have[:3]}, "
               f"limit={parsed_result.options.limit}, client='{client_id}'")
    response = await search_v2(parsed_result, client_id=client_id)
    
    logger.info(f"📊 Qdrant results: {len(response.results)} candidates | "
               f"took={response.took_ms}ms | "
               f"top scores: {[f'{r.score:.3f}' for r in response.results[:5]]}")
    
    # 4. SMART RE-RANKING: Apply comprehensive ranking based on skills, titles, location, etc.
    # This is what makes smart search "smart" - aggressive bonuses for exact matches
    logger.info(f"🎯 Applying smart re-ranking to {len(response.results)} candidates...")
    logger.info(f"   Smart filters: skills={smart_filters.get('skills', [])[:3]}, "
               f"city={smart_filters.get('city')}, titles={smart_filters.get('titles', [])[:2]}")
    reranked_results = smart_rerank(response.results, smart_filters, location_preference)
    if reranked_results:
        logger.info(f"✅ After re-ranking: top_score={reranked_results[0].score:.3f}, "
                   f"candidate='{reranked_results[0].full_name[:30]}'")
        logger.info(f"   Top 5 scores: {[f'{r.score:.3f}' for r in reranked_results[:5]]}")
    else:
        logger.info("✅ After re-ranking: no results")
    
    # 4b. Experience post-filter — soft: keep profiles with unknown experience
    # (they may still be great matches via skills/title/location).
    smart_min = smart_filters.get("min_years")
    smart_max = smart_filters.get("max_years")
    if smart_min is not None or smart_max is not None:
        before_exp = len(reranked_results)
        def _exp_ok(r):
            yrs = r.years_experience
            if yrs is None:
                return True  # Keep profiles with unknown experience
            if smart_min is not None and yrs < smart_min:
                return False
            if smart_max is not None and yrs > smart_max:
                return False
            return True
        reranked_results = [r for r in reranked_results if _exp_ok(r)]
        logger.info(f"Experience post-filter: {before_exp} → {len(reranked_results)}")
    
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
        
        def _location_match_tier(r):
            """Return tier for sorting: 0=exact city, 1=state, 2=country, 3=area/other"""
            if target_city and target_city.lower() == (r.city or '').lower():
                return 0  # Exact city match — highest priority
            if target_city and target_city.lower() in (r.city or '').lower():
                return 0
            if target_state and target_state.lower() == (getattr(r, 'state', '') or '').lower():
                return 1  # State match
            if target_country and target_country.lower() == (r.country or '').lower():
                return 2  # Country match
            return 3  # Fallback (area/linkedin_area substring match)
        
        filtered_results = []
        for r in results:
            if (target_location in (r.city or "").lower()
                or target_location in (getattr(r, 'location', '') or "").lower()
                or target_location in (r.country or "").lower()
                or target_location in (getattr(r, 'state', '') or "").lower()
                or target_location in (getattr(r, 'linkedin_area', '') or "").lower()
                or target_location in (getattr(r, 'area', '') or "").lower()):
                filtered_results.append(r)
        
        # Two-tier sort: group by location tier first, then by score within each tier
        # This ensures all exact-city matches come first (sorted by score),
        # then state matches, then country matches — no interleaving.
        filtered_results.sort(key=lambda r: (_location_match_tier(r), -r.score))
        results = filtered_results
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
        location_preference: "remote" (no location boost), "preferred" (+0.12 city), "must_match" (+0.60 city, post-filter)
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
        # len > 1 so we keep short but important codes like QA, ML, AI, PM, VP, UX
        title_keywords.update(w for w in t.split() if len(w) > 1)
    
    logger.info(f"Re-ranking {len(results)} candidates with: location_preference={location_preference}, "
               f"city='{target_city}', titles={list(target_titles)[:2]}, "
               f"skills={list(target_skills)[:3]}, certs={list(target_certifications)[:2]}, exp={min_years}-{max_years}")
    
    # Re-rank each result
    reranked = []
    for result in results:
        bonus = 0.0
        bonus_details = []
        all_matched_skills = []
        location_matched = False
        title_matched = False
        exp_matched = False
        
        # ── 1. SKILLS MATCH ──────────────────────────────────────────
        profile_skills = set(s.lower() for s in (result.skills or []))
        
        # 1a. Skills matching - EXACT + FUZZY using thefuzz (catches ReactJS ~ React)
        from thefuzz import fuzz, process
        exact_matches = target_skills & profile_skills
        fuzzy_matches = set()
        
        profile_skill_list = list(profile_skills)  # for fuzzy
        for target_skill in target_skills - exact_matches:  # only fuzzy unmatched
            best_match, score = process.extractOne(target_skill, profile_skill_list, scorer=fuzz.ratio)
            if score >= 80:  # React ~ ReactJS, react.js, React Developer
                fuzzy_matches.add(best_match)
                all_matched_skills.append(target_skill)  # credit original target
        
        all_profile_skills_matched = exact_matches | fuzzy_matches
        if target_skills:
            skill_coverage = len(all_profile_skills_matched) / len(target_skills)
            skill_bonus = skill_coverage * 0.12  # Slightly higher for fuzzy rescue
            bonus += skill_bonus
            all_matched_skills.extend(list(all_profile_skills_matched))
            if skill_bonus > 0:
                bonus_details.append(f"skills_fuzzy+{skill_bonus:.2f} ({len(fuzzy_matches)} fuzzy)")
            logger.debug(f"Fuzzy rescued {len(fuzzy_matches)} skills for {result.full_name}")

        
        # 1b. Skills mentioned in headline/description/title/work_history (lower confidence fallback)
        missing_skills = target_skills - exact_matches
        if missing_skills:
            import re as _re
            profile_text = " ".join([
                (result.headline or ""),
                (result.current_title or ""),
                (result.description or ""),
            ] + [
                (wh.get('title') or "") + " " + (wh.get('description') or "")
                for wh in (result.work_history or [])[:5]
            ]).lower()
            text_matched = set()
            for skill in missing_skills:
                # Short skills (≤2 chars like "go", "r") need word-boundary matching
                # to avoid false positives (e.g. "go" matching "good programmer")
                if len(skill) <= 2:
                    if _re.search(r'\b' + _re.escape(skill) + r'\b', profile_text):
                        text_matched.add(skill)
                else:
                    if skill in profile_text:
                        text_matched.add(skill)
            if text_matched:
                text_bonus = (len(text_matched) / len(target_skills)) * 0.06  # Increased from 0.04
                bonus += text_bonus
                all_matched_skills.extend(text_matched)
                bonus_details.append(f"skills_text+{text_bonus:.2f}")
        
        # 1c. Expanded skill matches (related skills — award even if some exact matches exist)
        expanded_matches = expanded_skills & profile_skills
        if expanded_skills and expanded_matches:
            expanded_coverage = len(expanded_matches) / len(expanded_skills)
            exp_bonus = expanded_coverage * 0.04
            bonus += exp_bonus
            all_matched_skills.extend(expanded_matches)
            if exp_bonus > 0:
                bonus_details.append(f"expanded+{exp_bonus:.2f}")
        
# 1d. SKILLS HARD FILTERING - Use config thresholds
        config = get_config()
        
        # Count total skills (array + text matches)
        total_skill_matches = len(all_profile_skills_matched | text_matched)
        
        # Apply tiered penalties based on skill count
        if total_skill_matches == 0:
            bonus += config.no_skills_penalty  # -0.35
            bonus_details.append(f"no_skills{config.no_skills_penalty:+.2f}")
            logger.info(f"Hard no-skills penalty: {result.full_name} | skills=0")
        elif total_skill_matches == 1:
            bonus += -0.20  # Tier 1 penalty
            bonus_details.append("1_skill-0.20")
        elif total_skill_matches < config.min_skills_required:
            bonus += config.min_skills_penalty  # Configurable
            bonus_details.append(f"low_skills{config.min_skills_penalty:+.2f}")
        
        # Skills coverage score for logging
        if target_skills:
            coverage = total_skill_matches / len(target_skills)
            if coverage < config.min_skills_score:
                bonus_details.append(f"coverage={coverage:.1f}<{config.min_skills_score}")
        

        
        # 1e. SOFT EMPTY PROFILE HANDLING — fallback scan + reduced penalty
        has_headline = bool(result.headline and result.headline.strip() and result.headline.strip() != '--')
        has_summary = bool(result.description and len(result.description.strip()) > 20)
        has_skills = bool(result.skills and len(result.skills) > 0)
        
        if not has_skills:
            # FALLBACK: Scan headline/summary for relevance without hard penalty
            profile_fallback_text = f"{result.headline or ''} {result.description or ''} {result.current_title or ''}".lower()
            fallback_matches = sum(1 for skill in target_skills if skill in profile_fallback_text)
            if fallback_matches > 0:
                fallback_bonus = fallback_matches * 0.03  # Small boost for text relevance
                bonus += fallback_bonus
                bonus_details.append(f"empty_rescue+{fallback_bonus:.2f}")
                logger.debug(f"Rescued empty profile {result.full_name} via text: {fallback_matches} hits")
            elif target_skills:  # Still penalize truly empty/irrelevant
                bonus -= 0.05  # Reduced from 0.15
                bonus_details.append("empty_weak-0.05")

        
        # 1f. TITLE/HEADLINE RELEVANCE PENALTY — demote profiles whose title/headline
        # are obviously unrelated to the query (e.g. "Piano Teacher" for a "react" search)
        if target_skills and not all_matched_skills and not title_matched:
            import re as _re2
            profile_blob = ' '.join([
                (result.headline or ''),
                (result.current_title or ''),
                (result.description or ''),
                ' '.join(s for s in (result.skills or [])),
            ]).lower()
            # Check if ANY target skill keyword appears anywhere in profile text
            has_any_relevance = False
            for skill in target_skills:
                if len(skill) <= 2:
                    if _re2.search(r'\b' + _re2.escape(skill) + r'\b', profile_blob):
                        has_any_relevance = True
                        break
                else:
                    if skill in profile_blob:
                        has_any_relevance = True
                        break
            # Also check title keywords from the query
            if not has_any_relevance and title_keywords:
                profile_title_words = set(profile_blob.split())
                if title_keywords & profile_title_words:
                    has_any_relevance = True
            if not has_any_relevance:
                bonus -= 0.12
                bonus_details.append("no_relevance-0.12")
        
        # ── 2. LOCATION MATCH ────────────────────────────────────────
        location_matched = False
        if location_preference != "remote" and (target_city or target_state or target_country):
            profile_location = (result.location or "").lower()
            profile_city = (result.city or "").lower()
            profile_state = (result.state or "").lower()
            profile_country = (result.country or "").lower()
            
            if location_preference == "must_match":
                city_boost, state_boost, country_boost = 0.60, 0.30, 0.15
            else:  # "preferred" — strong boost to lift local candidates
                city_boost, state_boost, country_boost = 0.35, 0.15, 0.08
            
            # City matching: prefer exact match, then check linkedin_area/area, then substring
            if target_city:
                if profile_city == target_city:
                    # Exact city match — highest confidence
                    bonus += city_boost
                    bonus_details.append(f"city+{city_boost:.2f}")
                    location_matched = True
                elif target_city in (getattr(result, 'linkedin_area', '') or '').lower():
                    bonus += city_boost
                    bonus_details.append(f"city_area+{city_boost:.2f}")
                    location_matched = True
                elif target_city in (getattr(result, 'area', '') or '').lower():
                    bonus += city_boost
                    bonus_details.append(f"city_area+{city_boost:.2f}")
                    location_matched = True
                elif target_city in profile_location:
                    # Substring fallback for combined location strings
                    bonus += city_boost
                    bonus_details.append(f"city+{city_boost:.2f}")
                    location_matched = True
            if not location_matched and target_state and (target_state == profile_state or target_state in profile_location):
                bonus += state_boost
                bonus_details.append(f"state+{state_boost:.2f}")
                location_matched = True
            if not location_matched and target_country and (target_country == profile_country or target_country in profile_location):
                bonus += country_boost
                bonus_details.append(f"country+{country_boost:.2f}")
                location_matched = True
        
        # 2b. LOCATION NON-MATCH PENALTY — demote profiles not in requested location
        if not location_matched and (target_city or target_state or target_country):
            if location_preference != "remote":
                bonus -= 0.08
                bonus_details.append("no_location-0.08")
        
        # ── 3. TITLE MATCH (comprehensive) ────────────────────────────
        profile_title = (result.current_title or "").lower()
        profile_headline = (result.headline or "").lower()
        title_matched = False
        
        # 3a. Exact substring match in current_title
        for target_title in target_titles:
            if target_title in profile_title:
                bonus += 0.10
                bonus_details.append("title+0.10")
                title_matched = True
                break
        
        # 3b. Word overlap match (e.g. "software engineer" matches "senior software engineer")
        if not title_matched and title_keywords:
            title_words = set(w for w in profile_title.split() if len(w) > 2)
            overlap = title_keywords & title_words
            if overlap:
                overlap_ratio = len(overlap) / len(title_keywords)
                if overlap_ratio >= 0.5:  # At least half the words match
                    partial_bonus = overlap_ratio * 0.06
                    bonus += partial_bonus
                    bonus_details.append(f"title_partial+{partial_bonus:.2f}")
                    title_matched = True
        
        # 3c. Headline match (candidate's self-description often mentions their role)
        if not title_matched and profile_headline:
            for target_title in target_titles:
                if target_title in profile_headline:
                    bonus += 0.04
                    bonus_details.append("headline_title+0.04")
                    break
        
        # 3d. Work history title match (past roles matching target title)
        if not title_matched and result.work_history:
            for i, work_exp in enumerate(result.work_history[:5]):
                past_title = (work_exp.get('title') or "").lower()
                for target_title in target_titles:
                    if target_title in past_title:
                        wh_bonus = 0.06 if i == 0 else 0.03
                        bonus += wh_bonus
                        bonus_details.append(f"wh_title+{wh_bonus:.2f}")
                        title_matched = True
                        break
                if title_matched:
                    break
        
        # ── 4. COMPANY MATCH ──────────────────────────────────────────
        # 4a. Current company match
        profile_company = (result.current_company or "").lower()
        company_matched = False
        for target_company in all_companies:
            if target_company in profile_company:
                bonus += 0.06
                bonus_details.append("curr_company+0.06")
                company_matched = True
                break
        
        # 4b. Past company match (work history)
        if not company_matched and result.work_history:
            for work_exp in result.work_history[:5]:  # Check top 5 most recent
                past_company = (work_exp.get('company') or "").lower()
                for target_company in all_companies:
                    if target_company in past_company:
                        bonus += 0.04  # Lower than current company
                        bonus_details.append("past_company+0.04")
                        company_matched = True
                        break
                if company_matched:
                    break
        
        # ── 5. CERTIFICATION MATCH ────────────────────────────────────
        matched_certs = []
        if target_certifications:
            # Build full-text blob from all profile text fields
            cert_text_parts = [c.lower() for c in (result.certifications or [])]
            cert_text_parts.append((result.headline or '').lower())
            cert_text_parts.append((result.current_title or '').lower())
            cert_text_parts.append((getattr(result, 'description', '') or '').lower())
            for wh in (getattr(result, 'work_history', None) or []):
                cert_text_parts.append((wh.get('title') or '').lower())
                cert_text_parts.append((wh.get('description') or '').lower())
            cert_blob = ' '.join(cert_text_parts)

            for target_cert in target_certifications:
                aliases = CERT_ALIASES.get(target_cert, [target_cert])
                for alias in aliases:
                    if alias in cert_blob:
                        matched_certs.append(target_cert)
                        break

            if matched_certs:
                cert_coverage = len(set(matched_certs)) / len(target_certifications)
                cert_bonus = cert_coverage * 0.10  # Up to +0.10 for all certs matched
                bonus += cert_bonus
                bonus_details.append(f"certs+{cert_bonus:.2f}")
        
        # ── 6. EXPERIENCE MATCH (continuous proximity scoring) ─────────
        exp_matched = False
        exp = result.years_experience
        if exp is not None and (min_years is not None or max_years is not None):
            effective_min = min_years if min_years is not None else 0
            effective_max = max_years if max_years is not None else 50
            
            if effective_min <= exp <= effective_max:
                # In range — full bonus
                bonus += 0.06
                bonus_details.append("exp+0.06")
                exp_matched = True
            else:
                # Out of range — continuous decay (closer to ideal = higher bonus)
                ideal = (effective_min + effective_max) / 2
                diff = abs(exp - ideal)
                proximity = max(0.0, 1.0 - diff / 8.0)  # Decays over 8 years
                exp_bonus = round(proximity * 0.04, 3)
                if exp_bonus > 0:
                    bonus += exp_bonus
                    bonus_details.append(f"exp~+{exp_bonus:.2f}")
        
        # ── 7. PROFILE QUALITY BONUS ──────────────────────────────────
        # Reward complete profiles (they're more useful to recruiters)
        completeness = result.profile_completeness or 0
        if completeness > 70:
            quality_bonus = 0.02
            bonus += quality_bonus
        
        # ── 8. MULTI-AXIS COMBO BONUS ─────────────────────────────────
        # Candidates matching on multiple dimensions simultaneously are
        # disproportionately likely to be the right person.
        matched_axes = sum([
            bool(exact_matches),           # has required skills
            title_matched,                 # title match
            location_matched,              # location match
            company_matched,               # company match
            exp_matched,                   # experience match
            bool(matched_certs),           # certification match
        ])
        if matched_axes >= 4:
            bonus += 0.06
            bonus_details.append("combo4+0.06")
        elif matched_axes >= 3:
            bonus += 0.04
            bonus_details.append("combo3+0.04")
        elif matched_axes >= 2:
            bonus += 0.02
            bonus_details.append("combo2+0.02")
        
        # ── FINAL SCORE ───────────────────────────────────────────────
        # Base scores are normalized to [0.10, 0.65] so bonuses push
        # strong candidates toward 1.0. Cap at 1.0 for clean semantics.
        raw_score = result.score + bonus
        display_score = min(raw_score, 1.0)
        
        # Log first 5 for debugging
        if len(reranked) < 5:
            logger.info(f"  Candidate: {result.full_name[:25]} | {(result.city or 'no-city')[:15]} | "
                       f"base={result.score:.3f} bonus={bonus:.3f} final={display_score:.3f} | {bonus_details}")
        
        # Store raw_score temporarily for correct sorting; display_score shown to user
        # Create new result with updated score and comprehensive matched skills
        reranked.append(CandidateResultV2(
            forager_id=result.forager_id,
            score=round(raw_score, 6),   # raw for sorting
            full_name=result.full_name,
            current_title=result.current_title,
            current_company=result.current_company,
            location=result.location,
            city=result.city,
            state=result.state,
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
    
    # Sort by raw score (highest first), then cap for display
    reranked.sort(key=lambda x: x.score, reverse=True)
    for r in reranked:
        r.score = round(min(r.score, 1.0), 4)
    
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

@app.get("/api/stats/simple")
async def legacy_get_stats(request: Request):
    """Get basic statistics for the dashboard"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        p = [client_id]

        conn = get_db_connection()
        total = conn.execute("SELECT COUNT(*) FROM processed_profiles WHERE client_id = ?", p).fetchone()[0]
        countries = conn.execute("SELECT COUNT(DISTINCT canonical_country) FROM processed_profiles WHERE canonical_country IS NOT NULL AND canonical_country != '' AND client_id = ?", p).fetchone()[0]
        cities = conn.execute("SELECT COUNT(DISTINCT canonical_city) FROM processed_profiles WHERE canonical_city IS NOT NULL AND canonical_city != '' AND client_id = ?", p).fetchone()[0]
        industries = conn.execute("SELECT COUNT(DISTINCT primary_domain) FROM processed_profiles WHERE primary_domain IS NOT NULL AND primary_domain != '' AND client_id = ?", p).fetchone()[0]
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
    request: Request,
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
        from client_auth import require_client_id as _req
        client_id = _req(request)

        conn = get_db_connection()
        
        # Build dynamic SQL with proper WHERE clauses
        where_clauses = []
        params = []
        joins = []

        # Client isolation - always scope to the authenticated client's data
        where_clauses.append("pp.client_id = ?")
        params.append(client_id)
        
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
                ORDER BY start DESC NULLS LAST
            """).fetchall()
            
            cert_map = {}
            for fid, cert_name in cert_rows:
                if fid not in cert_map:
                    cert_map[fid] = []
                if cert_name:
                    cert_map[fid].append(cert_name)
            
            # Fetch work history (top 5 most recent per person)
            work_rows = conn.execute(f"""
                SELECT forager_id, role_title, organization_name, start_date, end_date
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
async def legacy_get_skills(request: Request, limit: int = 100):
    """Get top skills for filters (scoped to client's candidates)"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        # Extract skills from processed_profiles for this client
        rows = conn.execute("""
            SELECT canonical_skills
            FROM processed_profiles
            WHERE canonical_skills IS NOT NULL AND client_id = ?
        """, [client_id]).fetchall()
        conn.close()
        from collections import Counter
        skill_counter: Counter = Counter()
        for (skills_raw,) in rows:
            if skills_raw:
                skill_list = skills_raw if isinstance(skills_raw, list) else []
                for sk in skill_list:
                    if sk:
                        skill_counter[sk.strip()] += 1
        top = skill_counter.most_common(limit)
        return [{"value": sk, "label": sk, "count": cnt} for sk, cnt in top if sk]
    except Exception as e:
        logger.error(f"Error in legacy_get_skills: {e}")
        return []


@app.get("/api/filters/countries")
async def legacy_get_countries(request: Request):
    """Get all countries"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT DISTINCT canonical_country, COUNT(*) as cnt
            FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != ''
              AND client_id = ?
            GROUP BY canonical_country
            ORDER BY cnt DESC
        """, [client_id]).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_countries: {e}")
        return []


@app.get("/api/filters/cities")
async def legacy_get_cities(request: Request, country: Optional[str] = None):
    """Get cities, optionally filtered by country"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        if country:
            results = conn.execute("""
                SELECT DISTINCT canonical_city, COUNT(*) as cnt
                FROM processed_profiles
                WHERE canonical_country = ?
                  AND canonical_city IS NOT NULL
                  AND canonical_city != ''
                  AND client_id = ?
                GROUP BY canonical_city
                ORDER BY cnt DESC
            """, [country, client_id]).fetchall()
        else:
            results = conn.execute("""
                SELECT DISTINCT canonical_city, COUNT(*) as cnt
                FROM processed_profiles
                WHERE canonical_city IS NOT NULL AND canonical_city != ''
                  AND client_id = ?
                GROUP BY canonical_city
                ORDER BY cnt DESC
                LIMIT 200
            """, [client_id]).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_cities: {e}")
        return []


@app.get("/api/filters/locations")
async def legacy_get_locations(request: Request):
    """Get all unique locations"""
    return await legacy_get_cities(request)


@app.get("/api/filters/industries")
async def legacy_get_industries(request: Request):
    """Get all industries"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT DISTINCT primary_domain, COUNT(*) as cnt
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != ''
              AND client_id = ?
            GROUP BY primary_domain
            ORDER BY cnt DESC
            LIMIT 100
        """, [client_id]).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_industries: {e}")
        return []


@app.get("/api/filters/roles")
async def legacy_get_roles(request: Request):
    """Get all role titles for this client's candidates"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT r.role_title, COUNT(*) as cnt
            FROM roles r
            INNER JOIN processed_profiles pp ON r.forager_id = pp.person_id
            WHERE r.role_title IS NOT NULL AND r.role_title != ''
              AND pp.client_id = ?
            GROUP BY r.role_title
            ORDER BY cnt DESC
            LIMIT 100
        """, [client_id]).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_roles: {e}")
        return []


@app.get("/api/filters/schools")
async def legacy_get_schools(request: Request):
    """Get all schools for this client's candidates"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT e.school_name, COUNT(*) as cnt
            FROM educations e
            INNER JOIN processed_profiles pp ON e.forager_id = pp.person_id
            WHERE e.school_name IS NOT NULL AND e.school_name != ''
              AND pp.client_id = ?
            GROUP BY e.school_name
            ORDER BY cnt DESC
            LIMIT 100
        """, [client_id]).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_schools: {e}")
        return []


@app.get("/api/filters/certifications")
async def legacy_get_certifications(request: Request):
    """Get all certifications for this client's candidates"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT c.certificate_name, COUNT(*) as cnt
            FROM certifications c
            INNER JOIN processed_profiles pp ON c.forager_id = pp.person_id
            WHERE c.certificate_name IS NOT NULL AND c.certificate_name != ''
              AND pp.client_id = ?
            GROUP BY c.certificate_name
            ORDER BY cnt DESC
            LIMIT 100
        """, [client_id]).fetchall()
        conn.close()
        return [{"value": row[0], "label": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_get_certifications: {e}")
        return []


@app.get("/api/analytics/countries")
async def legacy_analytics_countries(request: Request, limit: int = 10):
    """Get candidate count by country for charts"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT canonical_country as country, COUNT(*) as count
            FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != ''
              AND client_id = ?
            GROUP BY canonical_country
            ORDER BY count DESC
            LIMIT ?
        """, [client_id, limit]).fetchall()
        conn.close()
        return [{"country": row[0], "count": row[1]} for row in results if row[0]]
    except Exception as e:
        logger.error(f"Error in legacy_analytics_countries: {e}")
        return []


@app.get("/api/analytics/industries")
async def legacy_analytics_industries(request: Request, limit: int = 10):
    """Get candidate count by industry for charts"""
    try:
        from client_auth import require_client_id as _req
        client_id = _req(request)
        conn = get_db_connection()
        results = conn.execute("""
            SELECT primary_domain, COUNT(*) as count
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL AND primary_domain != ''
              AND client_id = ?
            GROUP BY primary_domain
            ORDER BY count DESC
            LIMIT ?
        """, [client_id, limit]).fetchall()
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
# INTEGRATION ROUTER  (for the TypeScript mediator backend)
# POST /api/integration/search  — authenticated via X-Integration-Key header
# GET  /api/integration/health  — unauthenticated liveness probe
# =============================================================================

try:
    from integration_api import router as integration_router
    app.include_router(integration_router)
    logger.info("Integration API router loaded: /api/integration/*")
except Exception as _e:
    import traceback
    logger.warning("Integration API not loaded: %s\n%s", _e, traceback.format_exc())


# =============================================================================
# SEARCH QUALITY TEST ENDPOINTS
# =============================================================================
# Diagnostic endpoints to verify search ranking, relevance, and location modes.
# Each test runs a real smart-search and checks assertions on the results.
# Call GET /api/v2/test/all to run EVERY test, or individual tests below.

from pydantic import BaseModel as _TestBaseModel

class TestResult(_TestBaseModel):
    """Result of a single test case"""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None

class TestSuiteResult(_TestBaseModel):
    """Result of the full test suite"""
    total: int
    passed: int
    failed: int
    results: List[TestResult]
    took_ms: int


async def _run_smart_search_for_test(
    query: str,
    location_preference: str = "preferred",
    limit: int = 20,
    http_request: Optional[Request] = None,
) -> SmartSearchResponse:
    """Helper: run a smart search internally for testing."""
    req = SmartSearchQuery(query=query, limit=limit, location_preference=location_preference)
    return await smart_search_endpoint(req, http_request)


@app.get("/api/v2/test/relevance", response_model=TestResult)
async def test_relevance(request: Request, query: str = "react developer"):
    """
    TEST: Skill relevance — search for a skill and verify top results actually mention it.
    
    Example: GET /api/v2/test/relevance?query=react+developer
    
    Checks:
    - Top 10 results should mostly mention the skill in their skills/headline/title
    - Irrelevant profiles (Piano Teacher, Stylist, etc.) should NOT be in top 10
    """
    try:
        response = await _run_smart_search_for_test(query, http_request=request)
        top10 = response.results[:10]
        
        # Extract the primary skill keyword from the query
        query_words = set(w.lower() for w in query.split() if len(w) > 2 and w.lower() not in {
            'developer', 'engineer', 'manager', 'senior', 'junior', 'lead', 'with', 'the', 'and'
        })
        
        relevant_count = 0
        irrelevant = []
        for r in top10:
            profile_text = ' '.join([
                (r.headline or ''), (r.current_title or ''),
                ' '.join(r.skills or []), (r.description or '')[:200]
            ]).lower()
            
            has_relevance = any(kw in profile_text for kw in query_words)
            if has_relevance:
                relevant_count += 1
            else:
                irrelevant.append({
                    "name": r.full_name,
                    "title": r.current_title,
                    "headline": (r.headline or "")[:60],
                    "score": r.score,
                    "skills": (r.skills or [])[:5],
                })
        
        passed = relevant_count >= 7  # At least 7/10 should be relevant
        return TestResult(
            test_name=f"Relevance: '{query}'",
            passed=passed,
            message=f"{relevant_count}/10 top results are relevant" + (
                f". Irrelevant profiles found in top 10: {len(irrelevant)}" if irrelevant else ""
            ),
            details={
                "relevant_count": relevant_count,
                "total_checked": len(top10),
                "query_keywords": list(query_words),
                "irrelevant_in_top10": irrelevant,
                "total_results": response.total_matches,
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Relevance: '{query}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/location-must-match", response_model=TestResult)
async def test_location_must_match(request: Request, query: str = "software engineer", city: str = "Tokyo"):
    """
    TEST: Location must_match — verify ALL top K results are from the target city,
    with no interleaving from other cities.
    
    Example: GET /api/v2/test/location-must-match?query=software+engineer&city=Tokyo
    """
    try:
        full_query = f"{query} in {city}"
        response = await _run_smart_search_for_test(full_query, location_preference="must_match", http_request=request)
        top20 = response.results[:20]
        
        city_lower = city.lower()
        matching = []
        non_matching = []
        for r in top20:
            r_city = (r.city or "").lower()
            r_loc = (getattr(r, 'location', '') or "").lower()
            r_area = (getattr(r, 'area', '') or "").lower()
            if city_lower in r_city or city_lower in r_loc or city_lower in r_area:
                matching.append(r.full_name)
            else:
                non_matching.append({
                    "name": r.full_name,
                    "city": r.city,
                    "location": getattr(r, 'location', ''),
                    "score": r.score,
                })
        
        # Check for contiguous grouping: no non-matching should appear before matching
        all_match = len(non_matching) == 0
        passed = all_match
        
        return TestResult(
            test_name=f"Location Must-Match: '{city}'",
            passed=passed,
            message=f"{len(matching)}/{len(top20)} results match city '{city}'" + (
                f". Non-matching profiles found: {len(non_matching)}" if non_matching else " — all matched!"
            ),
            details={
                "matching_count": len(matching),
                "non_matching": non_matching,
                "total_results": response.total_matches,
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Location Must-Match: '{city}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/location-grouping", response_model=TestResult)
async def test_location_grouping(request: Request, query: str = "developer", city: str = "Mumbai"):
    """
    TEST: Location grouping — verify results are contiguously grouped by location tier
    (no interleaving: all city-matched results come first, then state, then country).
    
    Example: GET /api/v2/test/location-grouping?query=developer&city=Mumbai
    """
    try:
        full_query = f"{query} in {city}"
        response = await _run_smart_search_for_test(full_query, location_preference="must_match", http_request=request)
        top20 = response.results[:20]
        
        city_lower = city.lower()
        
        # Assign tier to each result
        tiers = []
        for r in top20:
            r_city = (r.city or "").lower()
            if r_city == city_lower:
                tiers.append(0)
            elif city_lower in r_city:
                tiers.append(0)
            elif city_lower in (r.state or "").lower():
                tiers.append(1)
            elif city_lower in (r.country or "").lower():
                tiers.append(2)
            else:
                tiers.append(3)
        
        # Check that tiers are non-decreasing (contiguous grouping)
        is_contiguous = all(tiers[i] <= tiers[i+1] for i in range(len(tiers)-1)) if len(tiers) > 1 else True
        
        return TestResult(
            test_name=f"Location Grouping: '{city}'",
            passed=is_contiguous,
            message=f"{'Tiers are contiguous ✅' if is_contiguous else 'Tiers are interleaved ❌'}. Tier sequence: {tiers[:20]}",
            details={
                "tier_sequence": tiers,
                "results": [
                    {"name": r.full_name, "city": r.city, "score": r.score, "tier": t}
                    for r, t in zip(top20, tiers)
                ]
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Location Grouping: '{city}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/empty-profiles", response_model=TestResult)
async def test_empty_profiles(request: Request, query: str = "python developer"):
    """
    TEST: Empty profiles — verify profiles with no skills AND no summary
    are NOT in the top 10.
    
    Example: GET /api/v2/test/empty-profiles?query=python+developer
    """
    try:
        response = await _run_smart_search_for_test(query, http_request=request)
        top10 = response.results[:10]
        
        empty_in_top10 = []
        for r in top10:
            has_skills = bool(r.skills and len(r.skills) > 0)
            has_summary = bool(r.description and len((r.description or "").strip()) > 20)
            has_headline = bool(r.headline and r.headline.strip() and r.headline.strip() != '--')
            
            if not has_skills and not has_summary and not has_headline:
                empty_in_top10.append({
                    "name": r.full_name,
                    "score": r.score,
                    "skills_count": len(r.skills or []),
                    "headline": r.headline,
                })
        
        passed = len(empty_in_top10) == 0
        return TestResult(
            test_name=f"Empty Profile Demotion: '{query}'",
            passed=passed,
            message=f"{'No' if passed else len(empty_in_top10)} empty profiles in top 10" + (
                f" — {[e['name'] for e in empty_in_top10]}" if empty_in_top10 else " ✅"
            ),
            details={
                "empty_profiles_in_top10": empty_in_top10,
                "total_results": response.total_matches,
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Empty Profile Demotion: '{query}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/skill-match", response_model=TestResult)
async def test_skill_match(request: Request, query: str = "react"):
    """
    TEST: Skill matching — verify top results have the queried skill in their profile.
    
    Example: GET /api/v2/test/skill-match?query=react
    """
    try:
        response = await _run_smart_search_for_test(query, http_request=request)
        top10 = response.results[:10]
        
        skill_lower = query.lower().strip()
        matched = []
        unmatched = []
        
        for r in top10:
            profile_skills = [s.lower() for s in (r.skills or [])]
            profile_text = ' '.join([
                (r.headline or ''), (r.current_title or ''),
                (r.description or '')[:200],
            ]).lower()
            
            has_skill = (
                skill_lower in profile_skills
                or any(skill_lower in s for s in profile_skills)
                or skill_lower in profile_text
            )
            
            if has_skill:
                matched.append(r.full_name)
            else:
                unmatched.append({
                    "name": r.full_name,
                    "title": r.current_title,
                    "headline": (r.headline or "")[:60],
                    "skills": (r.skills or [])[:5],
                    "score": r.score,
                })
        
        passed = len(matched) >= 7  # At least 7/10 should mention the skill
        return TestResult(
            test_name=f"Skill Match: '{query}'",
            passed=passed,
            message=f"{len(matched)}/10 top results mention '{skill_lower}'" + (
                f". Unmatched: {[u['name'] for u in unmatched]}" if unmatched else " ✅"
            ),
            details={
                "matched_count": len(matched),
                "unmatched": unmatched,
                "total_results": response.total_matches,
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Skill Match: '{query}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/title-match", response_model=TestResult)
async def test_title_match(request: Request, query: str = "software engineer"):
    """
    TEST: Title matching — verify top results have relevant job titles.
    
    Example: GET /api/v2/test/title-match?query=software+engineer
    """
    try:
        response = await _run_smart_search_for_test(query, http_request=request)
        top10 = response.results[:10]
        
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        
        title_matched = 0
        unmatched_titles = []
        for r in top10:
            title = (r.current_title or "").lower()
            headline = (r.headline or "").lower()
            
            title_words = set(title.split() + headline.split())
            overlap = query_words & title_words
            
            if len(overlap) >= len(query_words) * 0.5:  # At least half the words
                title_matched += 1
            else:
                unmatched_titles.append({
                    "name": r.full_name,
                    "title": r.current_title,
                    "headline": (r.headline or "")[:60],
                    "score": r.score,
                })
        
        passed = title_matched >= 5  # At least 5/10 should have matching titles
        return TestResult(
            test_name=f"Title Match: '{query}'",
            passed=passed,
            message=f"{title_matched}/10 top results have matching titles" + (
                f". Unmatched: {len(unmatched_titles)}" if unmatched_titles else " ✅"
            ),
            details={
                "title_matched": title_matched,
                "query_words": list(query_words),
                "unmatched_titles": unmatched_titles[:5],
                "total_results": response.total_matches,
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Title Match: '{query}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/score-sanity", response_model=TestResult)
async def test_score_sanity(request: Request, query: str = "data scientist"):
    """
    TEST: Score sanity — verify scores are properly ordered and within [0, 1].
    
    Example: GET /api/v2/test/score-sanity?query=data+scientist
    """
    try:
        response = await _run_smart_search_for_test(query, http_request=request)
        results = response.results
        
        issues = []
        
        # Check all scores in [0, 1]
        for r in results:
            if r.score < 0 or r.score > 1:
                issues.append(f"{r.full_name}: score={r.score} out of [0,1]")
        
        # Check descending order
        for i in range(len(results) - 1):
            if results[i].score < results[i+1].score:
                issues.append(
                    f"Score not descending at position {i}: "
                    f"{results[i].full_name}={results[i].score:.4f} < "
                    f"{results[i+1].full_name}={results[i+1].score:.4f}"
                )
        
        passed = len(issues) == 0
        return TestResult(
            test_name=f"Score Sanity: '{query}'",
            passed=passed,
            message=f"{'All scores valid ✅' if passed else f'{len(issues)} issues found ❌'}",
            details={
                "issues": issues[:10],
                "score_range": [results[0].score, results[-1].score] if results else [],
                "total_results": len(results),
            }
        )
    except Exception as e:
        return TestResult(test_name=f"Score Sanity: '{query}'", passed=False, message=f"Error: {str(e)}")


@app.get("/api/v2/test/all", response_model=TestSuiteResult)
async def test_all(request: Request):
    """
    🧪 RUN ALL SEARCH QUALITY TESTS
    
    Runs the full test suite and returns a summary.
    Each test exercises a different aspect of search relevance and ranking.
    
    GET /api/v2/test/all
    """
    start_time = time.time()
    results = []
    
    # Test 1: Skill relevance for "react"
    results.append(await test_relevance(request, query="react developer"))
    
    # Test 2: Skill relevance for "python"
    results.append(await test_relevance(request, query="python"))
    
    # Test 3: Skill match for "react"
    results.append(await test_skill_match(request, query="react"))
    
    # Test 4: Empty profiles demoted
    results.append(await test_empty_profiles(request, query="machine learning engineer"))
    
    # Test 5: Title matching
    results.append(await test_title_match(request, query="software engineer"))
    
    # Test 6: Score sanity
    results.append(await test_score_sanity(request, query="data scientist"))
    
    # Test 7: Location must-match (use a common city)
    results.append(await test_location_must_match(request, query="developer", city="Tokyo"))
    
    # Test 8: Location grouping
    results.append(await test_location_grouping(request, query="engineer", city="Mumbai"))
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    took_ms = int((time.time() - start_time) * 1000)
    
    return TestSuiteResult(
        total=len(results),
        passed=passed,
        failed=failed,
        results=results,
        took_ms=took_ms,
    )


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
