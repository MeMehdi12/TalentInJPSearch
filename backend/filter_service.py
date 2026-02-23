"""
Filter Service Module
======================

Production-ready filter services for:
- Faceted search with counts
- Real-time autocomplete
- Filter metadata

Author: Talentin AI Search Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache
import time

import duckdb
from pydantic import BaseModel, Field

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class FacetItem(BaseModel):
    """Single facet value with count"""
    value: str
    label: str
    count: int


class LocationFacets(BaseModel):
    """Location facets grouped by type"""
    cities: List[FacetItem] = Field(default_factory=list)
    countries: List[FacetItem] = Field(default_factory=list)


class ExperienceRange(BaseModel):
    """Experience range facet"""
    range: str
    label: str
    count: int


class FacetsResponse(BaseModel):
    """Response for facets endpoint"""
    skills: List[FacetItem] = Field(default_factory=list)
    locations: LocationFacets = Field(default_factory=LocationFacets)
    companies: List[FacetItem] = Field(default_factory=list)
    experience_ranges: List[ExperienceRange] = Field(default_factory=list)
    domains: List[FacetItem] = Field(default_factory=list)
    took_ms: int = 0


class AutocompleteResponse(BaseModel):
    """Response for autocomplete endpoint"""
    suggestions: List[FacetItem] = Field(default_factory=list)
    took_ms: int = 0


class FilterMetadataResponse(BaseModel):
    """Response for filter metadata endpoint"""
    total_profiles: int = 0
    available_filters: Dict[str, Any] = Field(default_factory=dict)
    took_ms: int = 0


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_connection():
    """Get DuckDB connection"""
    from config import get_config
    config = get_config()
    return duckdb.connect(str(config.duckdb_path), read_only=True)


# =============================================================================
# FACETS SERVICE
# =============================================================================

# Cache for expensive facet queries (5 min TTL)
_facet_cache: Dict[str, tuple] = {}
CACHE_TTL_SECONDS = 300


def _get_cached(key: str) -> Optional[Any]:
    """Get value from cache if not expired"""
    if key in _facet_cache:
        value, timestamp = _facet_cache[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return value
    return None


def _set_cached(key: str, value: Any) -> None:
    """Set value in cache with current timestamp"""
    _facet_cache[key] = (value, time.time())


def get_skill_facets(
    conn,
    current_filters: Optional[Dict] = None,
    limit: int = 50
) -> List[FacetItem]:
    """
    Get top skills with counts, optionally filtered.
    
    Args:
        conn: DuckDB connection
        current_filters: Active filters to apply
        limit: Max skills to return
    
    Returns:
        List of FacetItem with skill counts
    """
    cache_key = f"skills:{hash(str(current_filters))}:{limit}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        # Build WHERE clause based on current filters - USING PARAMETERIZED QUERIES
        where_clauses = []
        params = []
        
        if current_filters:
            if current_filters.get("location", {}).get("city"):
                city = current_filters["location"]["city"].lower()
                where_clauses.append("LOWER(p.canonical_city) = ?")
                params.append(city)
            if current_filters.get("location", {}).get("country"):
                country = current_filters["location"]["country"].lower()
                where_clauses.append("LOWER(p.canonical_country) = ?")
                params.append(country)
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        query = f"""
            SELECT 
                s.skills as skill,
                COUNT(DISTINCT s.forager_id) as count
            FROM skills s
            INNER JOIN processed_profiles p ON s.forager_id = p.person_id
            {where_sql}
            GROUP BY s.skills
            ORDER BY count DESC
            LIMIT ?
        """
        params.append(limit)
        
        results = conn.execute(query, params).fetchall()
        
        facets = [
            FacetItem(
                value=row[0].lower(),
                label=row[0].title() if row[0] else "Unknown",
                count=row[1]
            )
            for row in results
            if row[0]  # Skip null skills
        ]
        
        _set_cached(cache_key, facets)
        return facets
        
    except Exception as e:
        logger.error(f"Error getting skill facets: {e}")
        return []


def get_location_facets(
    conn,
    current_filters: Optional[Dict] = None,
    limit: int = 30
) -> LocationFacets:
    """Get location facets (cities and countries) with counts"""
    cache_key = f"locations:{hash(str(current_filters))}:{limit}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        # Cities
        city_query = f"""
            SELECT 
                canonical_city,
                canonical_country,
                COUNT(*) as count
            FROM processed_profiles
            WHERE canonical_city IS NOT NULL AND canonical_city != ''
            GROUP BY canonical_city, canonical_country
            ORDER BY count DESC
            LIMIT {limit}
        """
        city_results = conn.execute(city_query).fetchall()
        
        cities = [
            FacetItem(
                value=row[0].lower() if row[0] else "",
                label=f"{row[0]}, {row[1]}" if row[1] else row[0],
                count=row[2]
            )
            for row in city_results
            if row[0]
        ]
        
        # Countries
        country_query = f"""
            SELECT 
                canonical_country,
                COUNT(*) as count
            FROM processed_profiles
            WHERE canonical_country IS NOT NULL AND canonical_country != ''
            GROUP BY canonical_country
            ORDER BY count DESC
            LIMIT {limit}
        """
        country_results = conn.execute(country_query).fetchall()
        
        countries = [
            FacetItem(
                value=row[0].lower() if row[0] else "",
                label=row[0],
                count=row[1]
            )
            for row in country_results
            if row[0]
        ]
        
        result = LocationFacets(cities=cities, countries=countries)
        _set_cached(cache_key, result)
        return result
        
    except Exception as e:
        logger.error(f"Error getting location facets: {e}")
        return LocationFacets()


def get_company_facets(
    conn,
    current_filters: Optional[Dict] = None,
    limit: int = 30
) -> List[FacetItem]:
    """Get top companies with counts"""
    cache_key = f"companies:{hash(str(current_filters))}:{limit}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        query = f"""
            SELECT 
                current_role_company,
                COUNT(*) as count
            FROM processed_profiles
            WHERE current_role_company IS NOT NULL 
              AND current_role_company != ''
              AND current_role_company != 'Unknown'
            GROUP BY current_role_company
            ORDER BY count DESC
            LIMIT {limit}
        """
        results = conn.execute(query).fetchall()
        
        facets = [
            FacetItem(
                value=row[0].lower() if row[0] else "",
                label=row[0],
                count=row[1]
            )
            for row in results
            if row[0]
        ]
        
        _set_cached(cache_key, facets)
        return facets
        
    except Exception as e:
        logger.error(f"Error getting company facets: {e}")
        return []


def get_experience_facets(conn) -> List[ExperienceRange]:
    """Get experience ranges with counts"""
    cache_key = "experience_ranges"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        query = """
            SELECT 
                CASE 
                    WHEN years_experience <= 2 THEN '0-2'
                    WHEN years_experience <= 5 THEN '3-5'
                    WHEN years_experience <= 10 THEN '6-10'
                    ELSE '10+'
                END as exp_range,
                COUNT(*) as count
            FROM processed_profiles
            WHERE years_experience IS NOT NULL
            GROUP BY exp_range
            ORDER BY 
                CASE exp_range 
                    WHEN '0-2' THEN 1 
                    WHEN '3-5' THEN 2 
                    WHEN '6-10' THEN 3 
                    ELSE 4 
                END
        """
        results = conn.execute(query).fetchall()
        
        labels = {
            "0-2": "0-2 years (Junior)",
            "3-5": "3-5 years (Mid)",
            "6-10": "6-10 years (Senior)",
            "10+": "10+ years (Expert)"
        }
        
        facets = [
            ExperienceRange(
                range=row[0],
                label=labels.get(row[0], row[0]),
                count=row[1]
            )
            for row in results
        ]
        
        _set_cached(cache_key, facets)
        return facets
        
    except Exception as e:
        logger.error(f"Error getting experience facets: {e}")
        return []


def get_domain_facets(conn, limit: int = 20) -> List[FacetItem]:
    """Get domain facets with counts"""
    cache_key = f"domains:{limit}"
    cached = _get_cached(cache_key)
    if cached:
        return cached
    
    try:
        query = f"""
            SELECT 
                primary_domain,
                COUNT(*) as count
            FROM processed_profiles
            WHERE primary_domain IS NOT NULL
            GROUP BY primary_domain
            ORDER BY count DESC
            LIMIT {limit}
        """
        results = conn.execute(query).fetchall()
        
        facets = [
            FacetItem(
                value=row[0].lower() if row[0] else "",
                label=row[0].replace("_", " ").title() if row[0] else "Unknown",
                count=row[1]
            )
            for row in results
            if row[0]
        ]
        
        _set_cached(cache_key, facets)
        return facets
        
    except Exception as e:
        logger.error(f"Error getting domain facets: {e}")
        return []


def get_all_facets(current_filters: Optional[Dict] = None) -> FacetsResponse:
    """
    Get all facets in one call.
    
    Args:
        current_filters: Active filters to apply (affects counts)
    
    Returns:
        FacetsResponse with all facet types
    """
    start_time = time.time()
    conn = get_db_connection()
    
    try:
        response = FacetsResponse(
            skills=get_skill_facets(conn, current_filters),
            locations=get_location_facets(conn, current_filters),
            companies=get_company_facets(conn, current_filters),
            experience_ranges=get_experience_facets(conn),
            domains=get_domain_facets(conn),
            took_ms=int((time.time() - start_time) * 1000)
        )
        
        logger.info(f"Facets loaded in {response.took_ms}ms")
        return response
        
    finally:
        conn.close()


# =============================================================================
# AUTOCOMPLETE SERVICE
# =============================================================================

def autocomplete_skills(query: str, limit: int = 10) -> AutocompleteResponse:
    """Autocomplete for skills field"""
    start_time = time.time()
    conn = get_db_connection()
    
    try:
        # Use LIKE for prefix matching
        query_pattern = f"{query.lower()}%"
        
        sql = f"""
            SELECT 
                skills as skill,
                COUNT(*) as count
            FROM skills
            WHERE LOWER(skills) LIKE ?
            GROUP BY skills
            ORDER BY count DESC
            LIMIT {limit}
        """
        results = conn.execute(sql, [query_pattern]).fetchall()
        
        suggestions = [
            FacetItem(
                value=row[0].lower(),
                label=row[0],
                count=row[1]
            )
            for row in results
        ]
        
        return AutocompleteResponse(
            suggestions=suggestions,
            took_ms=int((time.time() - start_time) * 1000)
        )
        
    finally:
        conn.close()


def autocomplete_companies(query: str, limit: int = 10) -> AutocompleteResponse:
    """Autocomplete for companies field"""
    start_time = time.time()
    conn = get_db_connection()
    
    try:
        query_pattern = f"%{query.lower()}%"
        
        sql = f"""
            SELECT 
                current_role_company,
                COUNT(*) as count
            FROM processed_profiles
            WHERE LOWER(current_role_company) LIKE ?
              AND current_role_company IS NOT NULL
              AND current_role_company != ''
            GROUP BY current_role_company
            ORDER BY count DESC
            LIMIT {limit}
        """
        results = conn.execute(sql, [query_pattern]).fetchall()
        
        suggestions = [
            FacetItem(
                value=row[0].lower(),
                label=row[0],
                count=row[1]
            )
            for row in results
        ]
        
        return AutocompleteResponse(
            suggestions=suggestions,
            took_ms=int((time.time() - start_time) * 1000)
        )
        
    finally:
        conn.close()


def autocomplete_locations(query: str, limit: int = 10) -> AutocompleteResponse:
    """Autocomplete for locations (cities)"""
    start_time = time.time()
    conn = get_db_connection()
    
    try:
        query_pattern = f"%{query.lower()}%"
        
        sql = f"""
            SELECT 
                canonical_city,
                canonical_country,
                COUNT(*) as count
            FROM processed_profiles
            WHERE LOWER(canonical_city) LIKE ?
              AND canonical_city IS NOT NULL
              AND canonical_city != ''
            GROUP BY canonical_city, canonical_country
            ORDER BY count DESC
            LIMIT {limit}
        """
        results = conn.execute(sql, [query_pattern]).fetchall()
        
        suggestions = [
            FacetItem(
                value=row[0].lower(),
                label=f"{row[0]}, {row[1]}" if row[1] else row[0],
                count=row[2]
            )
            for row in results
        ]
        
        return AutocompleteResponse(
            suggestions=suggestions,
            took_ms=int((time.time() - start_time) * 1000)
        )
        
    finally:
        conn.close()


def autocomplete_titles(query: str, limit: int = 10) -> AutocompleteResponse:
    """Autocomplete for job titles"""
    start_time = time.time()
    conn = get_db_connection()
    
    try:
        query_pattern = f"%{query.lower()}%"
        
        sql = f"""
            SELECT 
                current_role_title,
                COUNT(*) as count
            FROM processed_profiles
            WHERE LOWER(current_role_title) LIKE ?
              AND current_role_title IS NOT NULL
              AND current_role_title != ''
            GROUP BY current_role_title
            ORDER BY count DESC
            LIMIT {limit}
        """
        results = conn.execute(sql, [query_pattern]).fetchall()
        
        suggestions = [
            FacetItem(
                value=row[0].lower(),
                label=row[0],
                count=row[1]
            )
            for row in results
        ]
        
        return AutocompleteResponse(
            suggestions=suggestions,
            took_ms=int((time.time() - start_time) * 1000)
        )
        
    finally:
        conn.close()


def autocomplete(field: str, query: str, limit: int = 10) -> AutocompleteResponse:
    """
    Unified autocomplete handler.
    
    Args:
        field: Field to autocomplete (skills, companies, locations, titles)
        query: Search prefix
        limit: Max suggestions
    
    Returns:
        AutocompleteResponse with suggestions
    """
    handlers = {
        "skills": autocomplete_skills,
        "companies": autocomplete_companies,
        "locations": autocomplete_locations,
        "titles": autocomplete_titles
    }
    
    handler = handlers.get(field.lower())
    if not handler:
        logger.warning(f"Unknown autocomplete field: {field}")
        return AutocompleteResponse()
    
    return handler(query, limit)


# =============================================================================
# FILTER METADATA SERVICE
# =============================================================================

def get_filter_metadata() -> FilterMetadataResponse:
    """
    Get metadata about available filters.
    
    Returns counts, ranges, and top values for all filterable fields.
    """
    start_time = time.time()
    conn = get_db_connection()
    
    try:
        # Total profiles
        total = conn.execute("SELECT COUNT(*) FROM processed_profiles").fetchone()[0]
        
        # Skills metadata
        skills_count = conn.execute(
            "SELECT COUNT(DISTINCT skills) FROM skills"
        ).fetchone()[0]
        
        top_skills = conn.execute("""
            SELECT skills as skill, COUNT(*) as count
            FROM skills
            GROUP BY skills
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()
        
        # Location metadata
        cities_count = conn.execute(
            "SELECT COUNT(DISTINCT canonical_city) FROM processed_profiles WHERE canonical_city IS NOT NULL"
        ).fetchone()[0]
        
        countries_count = conn.execute(
            "SELECT COUNT(DISTINCT canonical_country) FROM processed_profiles WHERE canonical_country IS NOT NULL"
        ).fetchone()[0]
        
        # Company metadata
        companies_count = conn.execute(
            "SELECT COUNT(DISTINCT current_role_company) FROM processed_profiles WHERE current_role_company IS NOT NULL"
        ).fetchone()[0]
        
        # Experience metadata
        exp_stats = conn.execute("""
            SELECT 
                MIN(years_experience),
                MAX(years_experience),
                AVG(years_experience)
            FROM processed_profiles
            WHERE years_experience IS NOT NULL
        """).fetchone()
        
        return FilterMetadataResponse(
            total_profiles=total,
            available_filters={
                "skills": {
                    "total_unique": skills_count,
                    "top_10": [s[0] for s in top_skills]
                },
                "locations": {
                    "total_cities": cities_count,
                    "total_countries": countries_count
                },
                "companies": {
                    "total_unique": companies_count
                },
                "experience": {
                    "min": exp_stats[0] if exp_stats[0] else 0,
                    "max": exp_stats[1] if exp_stats[1] else 50,
                    "avg": round(exp_stats[2], 1) if exp_stats[2] else 0
                }
            },
            took_ms=int((time.time() - start_time) * 1000)
        )
        
    finally:
        conn.close()


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def clear_facet_cache():
    """Clear all cached facet data"""
    global _facet_cache
    _facet_cache = {}
    logger.info("Facet cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return {
        "cached_keys": len(_facet_cache),
        "ttl_seconds": CACHE_TTL_SECONDS
    }
