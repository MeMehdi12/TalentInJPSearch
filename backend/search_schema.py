"""
Search Schema v3 - Clean Pydantic models for Hybrid Search
==========================================================
Defines the input format for LLM output and API response format.

Two Input Formats Supported:
1. ParsedQueryV2 - New structured format from external LLM API
2. HybridSearchQuery - Original semantic/keyword format

Input Sanitization:
- Strips whitespace from all string inputs
- Removes empty strings from lists
- Handles null/None values gracefully
- Case-insensitive (normalized on search side)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from uuid import uuid4


# ============================================================================
# HELPER FUNCTIONS FOR SANITIZATION
# ============================================================================

def sanitize_string(value: Optional[str]) -> Optional[str]:
    """Clean a string input"""
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def sanitize_string_list(values: Optional[List]) -> List[str]:
    """Clean a list of strings - remove empty/None values"""
    if not values:
        return []
    result = []
    for v in values:
        if v is not None:
            cleaned = str(v).strip()
            if cleaned:
                result.append(cleaned)
    return result


# ============================================================================
# V2 INPUT SCHEMA - What your external LLM API should return
# ============================================================================

class SkillFiltersV2(BaseModel):
    """Skill-based filtering with AND/OR/NOT logic"""
    must_have: List[str] = Field(default_factory=list, description="Required skills (AND logic)")
    nice_to_have: List[str] = Field(default_factory=list, description="Preferred skills (boost ranking)")
    exclude: List[str] = Field(default_factory=list, description="Skills to filter out")
    
    @field_validator('must_have', 'nice_to_have', 'exclude', mode='before')
    @classmethod
    def clean_skill_lists(cls, v):
        return sanitize_string_list(v)


class ExperienceFilterV2(BaseModel):
    """Experience range filter"""
    min_years: Optional[int] = Field(None, ge=0, le=50, description="Minimum years of experience")
    max_years: Optional[int] = Field(None, ge=0, le=50, description="Maximum years of experience")
    
    @model_validator(mode='after')
    def validate_range(self):
        if self.min_years is not None and self.max_years is not None:
            if self.min_years > self.max_years:
                # Auto-swap if reversed
                self.min_years, self.max_years = self.max_years, self.min_years
        return self


class LocationFilterV2(BaseModel):
    """Location filter with city/state/country"""
    city: Optional[str] = Field(None, description="City name (e.g., 'Tokyo', 'San Francisco')")
    state: Optional[str] = Field(None, description="State/Prefecture (e.g., 'California', 'Tokyo')")
    country: Optional[str] = Field(None, description="Country (e.g., 'Japan', 'USA')")
    
    @field_validator('city', 'state', 'country', mode='before')
    @classmethod
    def clean_location_strings(cls, v):
        return sanitize_string(v)


class CompanyFilterV2(BaseModel):
    """Company filter - supports current AND past employment"""
    worked_at: List[str] = Field(default_factory=list, description="Companies they worked at (current OR past)")
    current_only: bool = Field(False, description="If true, only match current company")
    exclude: List[str] = Field(default_factory=list, description="Companies to exclude")
    
    @field_validator('worked_at', 'exclude', mode='before')
    @classmethod
    def clean_company_lists(cls, v):
        return sanitize_string_list(v)


class FiltersV2(BaseModel):
    """All filters for v2 search"""
    skills: SkillFiltersV2 = Field(default_factory=SkillFiltersV2)
    experience: ExperienceFilterV2 = Field(default_factory=ExperienceFilterV2)
    location: LocationFilterV2 = Field(default_factory=LocationFilterV2)
    companies: CompanyFilterV2 = Field(default_factory=CompanyFilterV2)
    domain: Optional[str] = Field(None, description="Domain category (e.g., 'technology_software')")
    current_company: Optional[str] = Field(None, description="Current company filter (legacy)")
    job_titles: List[str] = Field(default_factory=list, description="Job titles to match semantically")
    schools: List[str] = Field(default_factory=list, description="Schools/universities to filter by")
    industries: List[str] = Field(default_factory=list, description="Industries to filter by")
    first_name: Optional[str] = Field(None, description="Exact first name to filter")
    last_name: Optional[str] = Field(None, description="Exact last name to filter")
    
    @field_validator('first_name', 'last_name', mode='before')
    @classmethod
    def clean_name_strings(cls, v):
        return sanitize_string(v)
    
    @field_validator('domain', 'current_company', mode='before')
    @classmethod
    def clean_strings(cls, v):
        return sanitize_string(v)
    
    @field_validator('job_titles', 'schools', mode='before')
    @classmethod
    def clean_lists(cls, v):
        return sanitize_string_list(v)


class SearchOptionsV2(BaseModel):
    """Search options"""
    limit: int = Field(50, ge=1, le=2000, description="Number of results to return")
    offset: int = Field(0, ge=0, description="Offset for pagination (number of results to skip)")
    expand_skills: bool = Field(True, description="Expand skills to related ones using skill_relationships")
    location_preference: str = Field("preferred", description="Location mode: 'remote', 'preferred', 'must_match'")
    selected_locations: List[str] = Field(default_factory=list, description="Filter results to these locations only")
    
    @field_validator('selected_locations', mode='before')
    @classmethod
    def clean_location_list(cls, v):
        return sanitize_string_list(v)


class ParsedQueryV2(BaseModel):
    """
    V2 Input Schema - Structured query from external LLM API.
    
    Example:
    {
        "search_text": "senior python backend developer",
        "filters": {
            "skills": {
                "must_have": ["python", "django"],
                "nice_to_have": ["react", "aws"],
                "exclude": ["java"]
            },
            "experience": {"min_years": 5, "max_years": 15},
            "location": {"city": "San Francisco", "state": "California"},
            "companies": {
                "worked_at": ["Google", "Meta", "Apple"],
                "current_only": false
            },
            "domain": "technology_software",
            "job_titles": ["Senior Engineer", "Tech Lead"]
        },
        "options": {"limit": 50, "expand_skills": true}
    }
    """
    search_text: Optional[str] = Field(None, description="Semantic search text (for embedding)")
    filters: FiltersV2 = Field(default_factory=FiltersV2)
    options: SearchOptionsV2 = Field(default_factory=SearchOptionsV2)


# ============================================================================
# HELPER CLASSES FOR HYBRID SEARCH QUERY
# ============================================================================

class LocationFilter(BaseModel):
    """Location preference"""
    city: Optional[str] = Field(None)
    state: Optional[str] = Field(None)
    country: Optional[str] = Field(None)
    
    @field_validator('city', 'state', 'country', mode='before')
    @classmethod
    def clean_strings(cls, v):
        return sanitize_string(v)


class ExperienceRange(BaseModel):
    """Experience range (can be hard filter or ranking signal)"""
    min_years: Optional[int] = Field(None, ge=0, le=50)
    max_years: Optional[int] = Field(None, ge=0, le=50)
    
    @model_validator(mode='after')
    def validate_range(self):
        if self.min_years is not None and self.max_years is not None:
            if self.min_years > self.max_years:
                self.min_years, self.max_years = self.max_years, self.min_years
        return self


class HybridSearchQuery(BaseModel):
    """
    Main input schema for hybrid search.
    Your external LLM should output this format.
    
    SIMPLE DESIGN:
    - search_text: Goes to dense (semantic) search
    - keywords: Goes to sparse (keyword/BM25) search
    - boost_*: Ranking boosts (higher score if matched)
    - exclude_*: Hard filters (remove from results)
    
    Example LLM outputs:
    
    1. "Python developer from Google":
    {
        "search_text": "python software developer engineer",
        "keywords": ["python", "google"],
        "boost_skills": ["python", "software development"],
        "boost_companies": ["google", "alphabet"]
    }
    
    2. "Senior engineer in San Francisco":
    {
        "search_text": "senior software engineer",
        "keywords": ["senior", "engineer", "san francisco"],
        "boost_titles": ["senior", "staff", "principal"],
        "boost_location": {"city": "san francisco", "state": "california"}
    }
    
    3. "MIT grad with ML experience":
    {
        "search_text": "machine learning engineer MIT",
        "keywords": ["mit", "machine learning"],
        "boost_skills": ["machine learning", "deep learning", "python"],
        "boost_schools": ["mit", "massachusetts institute of technology"]
    }
    """
    
    # === CORE SEARCH ===
    search_text: str = Field(
        default="",
        description="Semantic search text for dense embedding. Describe the ideal candidate."
    )
    
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for BM25 sparse search. Include specific names, skills, companies."
    )
    
    # === RANKING BOOSTS (soft signals) ===
    boost_skills: List[str] = Field(
        default_factory=list,
        description="Skills to boost in ranking. Candidates with these skills rank higher."
    )
    
    boost_companies: List[str] = Field(
        default_factory=list,
        description="Companies to boost. Ex-employees rank higher."
    )
    
    boost_schools: List[str] = Field(
        default_factory=list,
        description="Schools/universities to boost. Alumni rank higher."
    )
    
    boost_titles: List[str] = Field(
        default_factory=list,
        description="Job title keywords to boost. Matching titles rank higher."
    )
    
    boost_certifications: List[str] = Field(
        default_factory=list,
        description="Certifications to boost."
    )
    
    boost_location: Optional[LocationFilter] = Field(
        None,
        description="Location preference for ranking boost."
    )
    
    # === EXPERIENCE ===
    experience_range: Optional[ExperienceRange] = Field(
        None,
        description="Ideal experience range (years)."
    )
    
    # === HARD EXCLUDES (filter out) ===
    exclude_companies: List[str] = Field(
        default_factory=list,
        description="Companies to exclude. Candidates from these are filtered out."
    )
    
    exclude_skills: List[str] = Field(
        default_factory=list,
        description="Skills to exclude. Candidates with these are filtered out."
    )
    
    # === RESPONSE ===
    limit: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of results to return."
    )


# ============================================================================
# OUTPUT SCHEMA - What the search API returns
# ============================================================================

class RankingFactors(BaseModel):
    """Breakdown of how a candidate was scored"""
    vector_similarity: float = Field(..., description="Base score from hybrid search (0-1)")
    skills_coverage: float = Field(0.0, description="% of boost_skills matched")
    company_match_bonus: float = Field(0.0, description="1.0 if worked at boost_company, else 0")
    education_match_bonus: float = Field(0.0, description="1.0 if attended boost_school, else 0")
    location_bonus: float = Field(0.0, description="Location proximity (1.0=exact, 0.5=state, 0=other)")
    title_match_bonus: float = Field(0.0, description="1.0 if title matches boost_titles")
    experience_fit_bonus: float = Field(0.0, description="1.0 if in range, partial if close")
    cert_match_bonus: float = Field(0.0, description="1.0 if has boost_certification")
    profile_completeness_bonus: float = Field(0.0, description="Profile quality factor")


class CandidateResult(BaseModel):
    """A single candidate in search results"""
    forager_id: int = Field(..., description="Unique profile identifier")
    score: float = Field(..., description="Final relevance score")
    
    # Ranking breakdown
    ranking_factors: Optional[RankingFactors] = None
    
    # Basic info
    full_name: str = ""
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    
    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    
    # Experience
    years_experience: Optional[float] = None
    domain: Optional[str] = None
    
    # Skills
    skills: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    
    # Optional extended data
    headline: Optional[str] = None
    linkedin_url: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from the search API"""
    total_matches: int = Field(..., description="Total candidates found before limit")
    returned: int = Field(..., description="Number of candidates returned")
    took_ms: int = Field(..., description="Total search time in milliseconds")
    
    results: List[CandidateResult] = Field(default_factory=list)
    
    # Performance breakdown (for debugging)
    timings: Optional[Dict[str, int]] = Field(
        None,
        description="Timing breakdown in ms: dense_encode, sparse_encode, qdrant_search, duckdb_fetch, ranking"
    )
    
    # Debug info
    expanded_skills: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Skills that were expanded for search"
    )


# ============================================================================
# V2 RESPONSE SCHEMA - Clean response format
# ============================================================================

class CandidateResultV2(BaseModel):
    """A single candidate result in V2 response"""
    forager_id: int = Field(..., description="Unique profile identifier")
    score: float = Field(..., description="Final relevance score (0-1)")
    
    # Basic info
    full_name: str = ""
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: str = Field("", description="Formatted location string")
    city: Optional[str] = None
    country: Optional[str] = None
    photo: Optional[str] = None
    headline: Optional[str] = None
    linkedin_url: Optional[str] = None
    linkedin_slug: Optional[str] = None
    industry: Optional[str] = None
    search_name: Optional[str] = None
    description: Optional[str] = None
    is_creator: Optional[bool] = None
    is_influencer: Optional[bool] = None
    
    # Details
    years_experience: Optional[float] = None
    domain: Optional[str] = None
    profile_completeness: Optional[float] = None
    
    # Skills
    skills: List[str] = Field(default_factory=list, description="All skills")
    matched_skills: List[str] = Field(default_factory=list, description="Skills that matched query")
    
    # URLs
    linkedin_url: Optional[str] = None
    
    # Extended Person Data
    address: Optional[str] = None
    linkedin_country: Optional[str] = None
    linkedin_area: Optional[str] = None
    date_updated: Optional[Any] = None  # datetime or str
    primary_locale: Optional[str] = None
    temporary_status: Optional[str] = None
    temporary_emoji_status: Optional[str] = None
    background_picture: Optional[str] = None
    area: Optional[str] = None


class CityCount(BaseModel):
    """City with candidate count"""
    city: str = Field(..., description="City name")
    count: int = Field(..., description="Number of candidates from this city")


class LocationFacet(BaseModel):
    """Location facet for filtering"""
    name: str = Field(..., description="Location name")
    count: int = Field(..., description="Number of candidates from this location")


class SearchFacets(BaseModel):
    """Facets for filtering results"""
    remote_available: bool = Field(False, description="Whether remote flag is enabled")
    remote_count: int = Field(0, description="Total candidates when remote=true")
    locations: List[LocationFacet] = Field(default_factory=list, description="Locations with counts")


class SearchResponseV2(BaseModel):
    """V2 Response format"""
    query_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique query ID")
    total_matches: int = Field(..., description="Total candidates found before limit")
    returned: int = Field(..., description="Number of candidates returned")
    took_ms: int = Field(..., description="Total search time in milliseconds")
    
    results: List[CandidateResultV2] = Field(default_factory=list)
    
    # Facets for filtering
    facets: Optional[SearchFacets] = Field(None, description="Facets for filtering")
    
    # City breakdown for filtering (deprecated - use facets.locations)
    city_breakdown: List[CityCount] = Field(
        default_factory=list,
        description="Cities in results with candidate counts"
    )
    
    # Debug info (optional)
    expanded_skills: Optional[Dict[str, List[str]]] = Field(
        None, 
        description="Skills expanded via skill_relationships"
    )
    timings: Optional[Dict[str, int]] = Field(
        None,
        description="Timing breakdown in ms"
    )


# ============================================================================
# LEGACY COMPATIBILITY - Keep old classes for migration
# ============================================================================

class SkillFilters(BaseModel):
    """Legacy: Skill-based filtering"""
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


class ExperienceFilter(BaseModel):
    """Legacy: Experience level filtering"""
    min_years: Optional[int] = Field(None, ge=0, le=50)
    max_years: Optional[int] = Field(None, ge=0, le=50)


class SearchFilters(BaseModel):
    """Legacy: All available filters"""
    skills: SkillFilters = Field(default_factory=SkillFilters)
    experience: ExperienceFilter = Field(default_factory=ExperienceFilter)
    location: LocationFilter = Field(default_factory=LocationFilter)
    domain: Optional[str] = None
    job_titles: List[str] = Field(default_factory=list)
    current_company: Optional[str] = None


class ParsedSearchQuery(BaseModel):
    """Legacy: Old input schema (for backwards compatibility)"""
    search_text: Optional[str] = None
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(50, ge=1, le=100)
