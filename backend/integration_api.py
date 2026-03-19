"""
Integration API — Endpoint for the TypeScript Mediator Backend
==============================================================

Exposes a single endpoint:
    POST /api/integration/search

This is the ONLY entry-point the mediator backend should call.
It is completely separated from the existing frontend API so
existing behaviour is not affected.

Authentication
--------------
Every request must include:
    X-Integration-Key: <INTEGRATION_API_KEY value from .env>
OR
    Authorization: Bearer <INTEGRATION_API_KEY value from .env>

The key is compared in constant time to prevent timing side-channels.

Response Format
---------------
Returns the same schema as sampleresponse.json so the mediator/frontend
can consume it without any transformation:

    {
        "header": { "status": 200, "message": "Success" },
        "data": {
            "profiles": [ ...profile objects... ],
            "total": 42,
            "returned": 20,
            "page": 1,
            "took_ms": 312
        }
    }
"""

import hmac
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integration", tags=["integration"])


# =============================================================================
# AUTHENTICATION & PER-CLIENT KEY RESOLUTION
# =============================================================================

def _constant_time_equal(a: str, b: str) -> bool:
    """Timing-safe string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


def _load_key_client_map() -> Dict[str, str]:
    """
    Load INTEGRATION_KEY_CLIENT_MAP from env.
    Format: JSON object mapping api-key -> client_id
    Example: {"key-for-acme": "acme", "key-for-beta-corp": "beta"}
    """
    raw = os.getenv("INTEGRATION_KEY_CLIENT_MAP", "{}").strip()
    if not raw:
        return {}
    try:
        import json
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            raise ValueError("Must be a JSON object")
        return {str(k).strip(): str(v).strip() for k, v in mapping.items() if k and v}
    except Exception as exc:
        logger.error("INTEGRATION_KEY_CLIENT_MAP is not valid JSON: %s", exc)
        return {}


def _extract_key(request: Request) -> Optional[str]:
    """Pull the raw API key from headers (X-Integration-Key or Bearer token)."""
    key = request.headers.get("X-Integration-Key", "").strip()
    if key:
        return key
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip() or None
    return None


def _authenticate_and_resolve_client(request: Request, body_client_id: Optional[str]) -> str:
    """
    1. Extract the API key from the request headers.
    2. Validate it against INTEGRATION_KEY_CLIENT_MAP (per-client keys) -OR-
       the single INTEGRATION_API_KEY fallback.
    3. Resolve the client_id in priority order:
         a. Key found in INTEGRATION_KEY_CLIENT_MAP  → use mapped client_id
         b. body.client_id explicitly set             → use that
         c. INTEGRATION_API_KEY matches + DEFAULT_CLIENT_ID set → use default
    Returns the resolved client_id string, or raises HTTP 401/503.
    """
    provided_key = _extract_key(request)
    if not provided_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Send X-Integration-Key or Authorization: Bearer <key>.",
            headers={"WWW-Authenticate": 'Bearer realm="integration"'},
        )

    # --- Try per-client key map first ---
    key_map = _load_key_client_map()
    if key_map:
        # Must be constant-time against each entry to avoid timing oracle.
        # We iterate all entries regardless of early match.
        matched_client: Optional[str] = None
        for valid_key, client_id in key_map.items():
            if _constant_time_equal(provided_key, valid_key):
                matched_client = client_id  # keep iterating
        if matched_client is not None:
            # body.client_id can still override if the mapped key is a shared/admin key
            resolved = (body_client_id or matched_client).strip()
            logger.info("Integration auth OK — key→client: %s", resolved)
            return resolved
        # Key was not in map — fall through to single-key check

    # --- Fallback: single shared INTEGRATION_API_KEY ---
    master_key = os.getenv("INTEGRATION_API_KEY", "").strip()
    if not master_key:
        logger.error("No INTEGRATION_API_KEY or INTEGRATION_KEY_CLIENT_MAP configured")
        raise HTTPException(
            status_code=503,
            detail="Integration endpoint is not configured on this server.",
        )

    if not _constant_time_equal(provided_key, master_key):
        logger.warning("Integration auth failed — bad key")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": 'Bearer realm="integration"'},
        )

    # Key matches — resolve client_id
    resolved = (
        body_client_id
        or os.getenv("DEFAULT_CLIENT_ID", "talentin")
    ).strip()
    if not resolved:
        raise HTTPException(status_code=400, detail="client_id could not be resolved. Set DEFAULT_CLIENT_ID in .env.")

    logger.info("Integration auth OK — master key, client: %s", resolved)
    return resolved


# =============================================================================
# REQUEST MODELS
# =============================================================================

class LocationFilter(BaseModel):
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None


class ExperienceFilter(BaseModel):
    min_years: Optional[int] = Field(None, ge=0, le=60)
    max_years: Optional[int] = Field(None, ge=0, le=60)


class SearchFilters(BaseModel):
    # ── Skills ───────────────────────────────────────────────────────────────
    skills_required: List[str] = Field(default_factory=list, description="Must-have skills (AND — all required)")
    skills_nice_to_have: List[str] = Field(default_factory=list, description="Preferred skills (boost ranking)")
    skills_exclude: List[str] = Field(default_factory=list, description="Remove profiles that have any of these skills")

    # ── Location ─────────────────────────────────────────────────────────────
    location: Optional[LocationFilter] = None

    # ── Experience ───────────────────────────────────────────────────────────
    experience: Optional[ExperienceFilter] = None

    # ── Job & domain ─────────────────────────────────────────────────────────
    job_titles: List[str] = Field(default_factory=list, description="Target job title keywords")
    domain: Optional[str] = Field(None, description="Industry domain (e.g. 'technology_software')")
    industries: List[str] = Field(default_factory=list, description="Industry names to match")

    # ── Companies ────────────────────────────────────────────────────────────
    companies_worked_at: List[str] = Field(default_factory=list, description="Must have worked at (current OR past)")
    companies_current_only: bool = Field(False, description="If true, only match against current employer")
    companies_exclude: List[str] = Field(default_factory=list, description="Exclude profiles from these companies")

    # ── Education & credentials ───────────────────────────────────────────────
    schools: List[str] = Field(default_factory=list, description="Universities / schools to filter by")
    certifications: List[str] = Field(default_factory=list, description="Required certifications (e.g. 'PMP', 'AWS Certified')")

    # ── Person ────────────────────────────────────────────────────────────────
    first_name: Optional[str] = Field(None, description="Exact first-name filter")
    last_name: Optional[str] = Field(None, description="Exact last-name filter")


class IntegrationSearchRequest(BaseModel):
    search_text: str = Field("", description="Semantic description of the ideal candidate (used for vector embedding — no OpenAI call)")
    filters: Optional[SearchFilters] = None
    limit: int = Field(50, ge=1, le=500, description="Max profiles per page")
    page: int = Field(1, ge=1, description="Page number (1-based)")
    expand_skills: bool = Field(True, description="Expand skills via skill-relationship graph (no OpenAI)")
    client_id: Optional[str] = Field(None, description="Tenant scope override — falls back to DEFAULT_CLIENT_ID env var")
    location_preference: str = Field(
        "preferred",
        description="Location handling mode: 'preferred' (semantic boost, no hard filter), "
                    "'must_match' (strict location filter), 'remote' (ignore location entirely)",
    )
    explicit_match: bool = Field(
        False,
        description="When true, enforce strict filtering: all required skills must appear "
                    "in profile skills, and job titles must match current title or headline",
    )


# =============================================================================
# DUCKDB FULL-PROFILE HYDRATION (for integration — no truncation)
# =============================================================================

def _get_db_path() -> str:
    from config import get_config
    return get_config().duckdb_path


def _safe_ids_str(ids: List[int]) -> str:
    """Validate all IDs are integers and return a comma-separated string for SQL IN clause."""
    return ','.join(str(int(i)) for i in ids)


def _hydrate_full_profiles(person_ids: List[int], client_id: str) -> Dict[int, Dict]:
    """
    Fetch complete profile data from DuckDB for the given person IDs.
    Unlike the main search hydration, this does NOT truncate skills or work history.
    """
    if not person_ids:
        return {}

    db_path = _get_db_path()
    conn = duckdb.connect(db_path, read_only=True)

    try:
        ids_str = _safe_ids_str(person_ids)
        client_clause = "AND pp.client_id = ?" if client_id else ""
        client_params = [client_id] if client_id else []

        # ── Base profile data ─────────────────────────────────────────────────
        rows = conn.execute(
            f"""
            SELECT
                pp.person_id,
                pp.full_name,
                pp.canonical_city       AS city,
                pp.canonical_state      AS state,
                pp.canonical_country    AS country,
                pp.primary_domain       AS domain,
                pp.years_experience,
                pp.profile_completeness,
                pp.canonical_skills     AS skills,
                pp.current_role_title   AS current_title,
                pp.current_role_company AS current_company,
                p.photo,
                p.headline,
                p.linkedin_url,
                p.linkedin_slug,
                p.industry,
                p.description,
                p.linkedin_area,
                p.area,
                p.date_updated
            FROM processed_profiles pp
            LEFT JOIN persons p ON pp.person_id = p.forager_id
            WHERE pp.person_id IN ({ids_str}) {client_clause}
            """,
            client_params,
        ).fetchall()

        columns = [
            "person_id", "full_name", "city", "state", "country",
            "domain", "years_experience", "profile_completeness",
            "skills", "current_title", "current_company",
            "photo", "headline", "linkedin_url", "linkedin_slug",
            "industry", "description", "linkedin_area", "area", "date_updated",
        ]

        profiles: Dict[int, Dict] = {}
        for row in rows:
            p = dict(zip(columns, row))
            pid = p["person_id"]
            p.setdefault("skills", [])
            p["work_history"] = []
            p["education"] = []
            p["certifications"] = []
            profiles[pid] = p

        # ── Work history (all roles, no limit) ────────────────────────────────
        role_rows = conn.execute(
            f"""
            SELECT forager_id, role_title, organization_name,
                   start_date, end_date, description
            FROM roles
            WHERE forager_id IN ({ids_str})
            ORDER BY start_date DESC NULLS LAST
            """
        ).fetchall()

        for fid, title, company, start_dt, end_dt, desc in role_rows:
            if fid in profiles:
                profiles[fid]["work_history"].append(
                    {
                        "title": title,
                        "company": company,
                        "start_date": str(start_dt) if start_dt else None,
                        "end_date": str(end_dt) if end_dt else None,
                        "description": desc,
                    }
                )

        # ── Education (all records) ───────────────────────────────────────────
        edu_rows = conn.execute(
            f"""
            SELECT forager_id, school_name, degree, field_of_study,
                   start_date, end_date
            FROM educations
            WHERE forager_id IN ({ids_str})
            ORDER BY start_date DESC NULLS LAST
            """
        ).fetchall()

        for fid, school, degree, field, start_dt, end_dt in edu_rows:
            if fid in profiles:
                profiles[fid]["education"].append(
                    {
                        "school": school,
                        "degree": degree,
                        "field": field,
                        "start_date": str(start_dt) if start_dt else None,
                        "end_date": str(end_dt) if end_dt else None,
                    }
                )

        # ── Certifications ────────────────────────────────────────────────────
        cert_rows = conn.execute(
            f"""
            SELECT forager_id, certificate_name
            FROM certifications
            WHERE forager_id IN ({ids_str})
            """
        ).fetchall()

        for fid, cert_name in cert_rows:
            if fid in profiles and cert_name:
                profiles[fid]["certifications"].append(cert_name)

        return profiles

    finally:
        conn.close()


# =============================================================================
# RESPONSE SHAPE BUILDERS
# =============================================================================

def _fmt_date(raw) -> Optional[str]:
    """Normalise a date-like value to ISO 8601 with a time component."""
    if not raw:
        return None
    s = str(raw)
    # Already has time component
    if "T" in s:
        return s
    # Date only (YYYY-MM-DD)
    if len(s) >= 10:
        return s[:10] + "T00:00:00"
    return s


def _calc_years(start_raw, end_raw) -> float:
    """Calculate years between two date strings (end defaults to today)."""
    try:
        start = datetime.fromisoformat(str(start_raw)[:10])
        end = (
            datetime.fromisoformat(str(end_raw)[:10])
            if end_raw
            else datetime.now(timezone.utc).replace(tzinfo=None)
        )
        return round(max((end - start).days / 365.25, 0), 1)
    except Exception:
        return 0.0


def _years_label(raw: float) -> str:
    if raw < 1:
        return "Less than 1 year"
    if raw < 2:
        return "1 to 2 years"
    if raw < 5:
        return "3 to 5 years"
    if raw < 10:
        return "6 to 10 years"
    return "10+ years"


def _build_employer(role: Dict, is_current: bool) -> Dict:
    """Convert a work-history dict into the sampleresponse employer object."""
    start = role.get("start_date")
    end = role.get("end_date")
    yrs_raw = _calc_years(start, end if not is_current else None)

    obj: Dict[str, Any] = {
        "name": role.get("company") or "",
        "linkedin_id": "",
        "company_id": 0,
        "company_linkedin_id": "",
        "company_website_domain": "",
        "position_id": 0,
        "title": role.get("title") or "",
        "description": role.get("description") or "",
        "location": "",
        "start_date": _fmt_date(start),
        "employer_is_default": False,
        "seniority_level": "",
        "function_category": "",
        "years_at_company": _years_label(yrs_raw),
        "years_at_company_raw": int(yrs_raw),
        "company_headquarters_country": "",
        "company_hq_location": "",
        "company_hq_location_address_components": [],
        "company_headcount_range": "",
        "company_industries": [],
        "company_linkedin_industry": "",
        "company_type": "",
        "company_headcount_latest": 0,
        "company_website": "",
        "company_linkedin_profile_url": "",
        "business_email_verified": False,
    }

    if not is_current:
        obj["end_date"] = _fmt_date(end)

    return obj


def _build_education(edu: Dict) -> Dict:
    """Convert an education dict into the sampleresponse education_background object."""
    return {
        "degree_name": edu.get("degree") or "",
        "institute_name": edu.get("school") or "",
        "institute_linkedin_id": "",
        "institute_linkedin_url": "",
        "institute_logo_url": "",
        "field_of_study": edu.get("field") or "",
        "activities_and_societies": "",
        "start_date": _fmt_date(edu.get("start_date")),
        "end_date": _fmt_date(edu.get("end_date")),
    }


def _build_profile(candidate_id: int, score: float, raw: Dict) -> Dict:
    """
    Map a raw DuckDB profile dict into the sampleresponse.json profile object.
    """
    full_name = raw.get("full_name") or ""
    name_parts = full_name.strip().split(" ", 1)
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    # ── Region / location ─────────────────────────────────────────────────────
    # Prefer the LinkedIn-sourced area string (e.g. "San Francisco Bay Area")
    region = (
        raw.get("linkedin_area")
        or raw.get("area")
        or ", ".join(filter(None, [raw.get("city"), raw.get("state"), raw.get("country")]))
    )

    region_components: List[str] = [
        p for p in [
            raw.get("city"),
            raw.get("state"),
            raw.get("country"),
        ]
        if p
    ]

    # ── Employer splits ───────────────────────────────────────────────────────
    current_company_name = (raw.get("current_company") or "").lower()
    work_history = raw.get("work_history") or []

    current_employers: List[Dict] = []
    past_employers: List[Dict] = []

    for role in work_history:
        end_date = role.get("end_date")
        # A role is current if it has no end date, or the company name matches
        # the known current company (handles data inconsistencies)
        company_match = (role.get("company") or "").lower() == current_company_name
        is_current = (end_date is None) or (current_company_name and company_match and end_date is None)
        emp = _build_employer(role, is_current)
        if is_current:
            current_employers.append(emp)
        else:
            past_employers.append(emp)

    all_employers = current_employers + past_employers

    # ── Education ─────────────────────────────────────────────────────────────
    education_list = [_build_education(e) for e in (raw.get("education") or [])]

    # ── LinkedIn URLs ─────────────────────────────────────────────────────────
    linkedin_url = raw.get("linkedin_url") or ""
    slug = raw.get("linkedin_slug") or ""
    flagship_url = f"https://www.linkedin.com/in/{slug}" if slug else linkedin_url

    # ── Years of experience ───────────────────────────────────────────────────
    yrs_raw = raw.get("years_experience") or 0
    yrs_label = _years_label(float(yrs_raw))

    # ── Timestamps ───────────────────────────────────────────────────────────
    date_updated = raw.get("date_updated")
    if date_updated and hasattr(date_updated, "isoformat"):
        ts = date_updated.isoformat()
    else:
        ts = str(date_updated) if date_updated else None

    skills = raw.get("skills") or []
    if not isinstance(skills, list):
        skills = []

    return {
        "person_id": candidate_id,
        "_score": round(score, 4),  # Useful for the mediator to sort/debug
        "name": full_name,
        "first_name": first_name,
        "last_name": last_name,
        "region": region or "",
        "region_address_components": region_components,
        "headline": raw.get("headline") or "",
        "summary": raw.get("description") or "",
        "skills": skills,
        "languages": [],
        "profile_language": "",
        "linkedin_profile_url": linkedin_url,
        "flagship_profile_url": flagship_url,
        "emails": [],
        "profile_picture_url": raw.get("photo") or "",
        "profile_picture_permalink": raw.get("photo") or "",
        "twitter_handle": "",
        "open_to_cards": [],
        "num_of_connections": 0,
        "education_background": education_list,
        "honors": [],
        "certifications": raw.get("certifications") or [],
        "current_employers": current_employers,
        "past_employers": past_employers,
        "last_updated": ts,
        "recently_changed_jobs": False,
        "years_of_experience": yrs_label,
        "years_of_experience_raw": int(yrs_raw),
        "all_employers": all_employers,
        "updated_at": ts,
        "location_details": {
            "city": raw.get("city") or "",
            "state": raw.get("state") or "",
            "country": raw.get("country") or "",
            "continent": "",
        },
    }


# =============================================================================
# EXPLICIT MATCH FILTER
# =============================================================================

def _apply_explicit_filter(results, filters: "SearchFilters") -> list:
    """
    Strict post-filter for explicit_match mode.
    - ALL skills_required must appear in the profile's skills (fuzzy substring).
    - At least one job_title must match current_title or headline (if titles specified).
    """
    required_skills = [s.lower() for s in (filters.skills_required or [])]
    target_titles = [t.lower() for t in (filters.job_titles or [])]
    filtered = []

    for r in results:
        profile_skills = set(s.lower() for s in (r.skills or []))

        # Every required skill must have a fuzzy match in profile skills
        if required_skills:
            all_matched = True
            for skill in required_skills:
                # Short skills (≤2 chars like "go", "r") need exact matching
                # to avoid false positives (e.g. "r" matching "react")
                if len(skill) <= 2:
                    if not any(skill == ps for ps in profile_skills):
                        all_matched = False
                        break
                else:
                    if not any(skill in ps or ps in skill for ps in profile_skills):
                        all_matched = False
                        break
            if not all_matched:
                continue

        # At least one target title must appear in current_title or headline
        if target_titles:
            title = (r.current_title or "").lower()
            headline = (r.headline or "").lower()
            if not any(t in title or t in headline for t in target_titles):
                continue

        filtered.append(r)

    logger.info("Explicit filter: %d → %d candidates", len(results), len(filtered))
    return filtered


# =============================================================================
# ENDPOINT
# =============================================================================

@router.post("/search")
async def integration_search(body: IntegrationSearchRequest, request: Request):
    """
    Search endpoint for the TypeScript mediator backend.

    Authentication: X-Integration-Key header (or Authorization: Bearer <key>).
    client_id is resolved automatically from the key map — the mediator
    does NOT need to send client_id in the body unless overriding.

    Quality parity: This endpoint applies the same smart_rerank() pass
    used by the frontend smart-search so ranking quality is identical.
    """
    # ── 1. Auth + client_id resolution (one step) ────────────────────────────
    client_id = _authenticate_and_resolve_client(request, body.client_id)

    # ── 2. Convert payload → ParsedQueryV2 ────────────────────────────────────
    # NOTE: This path does NOT call OpenAI/ChatGPT.
    # search_v2 uses SentenceTransformer (local) for embeddings + Qdrant + DuckDB.
    from search_schema import (
        ParsedQueryV2,
        FiltersV2,
        SkillFiltersV2,
        ExperienceFilterV2,
        LocationFilterV2,
        CompanyFilterV2,
        SearchOptionsV2,
    )
    from search_api_v2 import search_v2, smart_rerank

    f = body.filters or SearchFilters()
    loc = f.location or LocationFilter()
    exp = f.experience or ExperienceFilter()

    # ── Enrich search_text for better embedding quality ──────────────────────
    enriched_text = body.search_text or ""
    if f.job_titles:
        enriched_text += " " + " ".join(f.job_titles)
    if f.domain:
        enriched_text += " " + f.domain.replace("_", " ")
    if f.certifications:
        enriched_text += " " + " ".join(f.certifications)

    # ── Skills handling ──────────────────────────────────────────────────────
    # By default, skills are NOT hard Qdrant filters. They're added to the
    # embedding text so semantic search surfaces relevant profiles, and
    # smart_rerank() boosts matches. This avoids the alias/normalization
    # dead-end where "next" returns 0 because no profile stores that exact
    # string, even though Next.js profiles are semantically perfect matches.
    # Only explicit_match mode enforces hard skill filtering (via post-filter).
    all_skill_terms = f.skills_required + f.skills_nice_to_have
    if all_skill_terms:
        enriched_text += " " + " ".join(all_skill_terms)

    # ── Location handling (mirrors smart-search behaviour) ───────────────────
    # Qdrant has inconsistent location data, so by default ("preferred") we
    # clear the hard filter and rely on semantic similarity + smart_rerank boost.
    original_city = loc.city
    original_state = loc.state
    original_country = loc.country

    qdrant_city, qdrant_state, qdrant_country = loc.city, loc.state, loc.country
    if body.location_preference != "must_match":
        # Add location to embedding text for semantic matching
        for part in (loc.city, loc.state, loc.country):
            if part:
                enriched_text += " " + part
        qdrant_city = qdrant_state = qdrant_country = None

    # ── Fetch extra candidates for re-ranking / post-filter headroom ─────────
    fetch_limit = min(body.limit * 5, 500)
    if body.location_preference == "must_match" or body.explicit_match:
        fetch_limit = min(body.limit * 10, 1000)

    parsed_query = ParsedQueryV2(
        search_text=enriched_text.strip() or "professional",
        filters=FiltersV2(
            skills=SkillFiltersV2(
                must_have=[],           # no hard filter — semantic + re-rank handles it
                nice_to_have=[],        # already in enriched_text
                exclude=f.skills_exclude,  # exclusions stay hard
            ),
            experience=ExperienceFilterV2(
                min_years=exp.min_years,
                max_years=exp.max_years,
            ),
            location=LocationFilterV2(
                city=qdrant_city,
                state=qdrant_state,
                country=qdrant_country,
            ),
            companies=CompanyFilterV2(
                worked_at=f.companies_worked_at,
                current_only=f.companies_current_only,
                exclude=f.companies_exclude,
            ),
            job_titles=f.job_titles,
            domain=f.domain,
            schools=f.schools,
            industries=f.industries,
            certifications=f.certifications,
            first_name=f.first_name,
            last_name=f.last_name,
        ),
        options=SearchOptionsV2(
            limit=fetch_limit,
            offset=0,       # paginate after re-ranking, not inside search_v2
            expand_skills=body.expand_skills,
        ),
    )

    # ── 3. Run vector search ─────────────────────────────────────────────────
    try:
        search_result = await search_v2(parsed_query, client_id=client_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Integration search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Search engine error. Please try again.")

    if not search_result.results:
        return {
            "header": {"status": 200, "message": "Success"},
            "data": {
                "profiles": [],
                "total": 0,
                "returned": 0,
                "page": body.page,
                "took_ms": search_result.took_ms,
            },
        }

    # ── 4. Smart re-ranking (same quality as frontend smart-search) ──────────
    # Expand skills & companies so smart_rerank() can award expanded-match bonuses
    from search_api_v2 import expand_skills as _expand_skills
    from normalizers import expand_company_search as _expand_companies
    _all_skills = f.skills_required + f.skills_nice_to_have
    _skill_expansion = _expand_skills(_all_skills) if _all_skills else {}
    _expanded_skill_list = list({s for related in _skill_expansion.values() for s in related})
    _expanded_company_list = _expand_companies(f.companies_worked_at) if f.companies_worked_at else []

    smart_filters = {
        "skills": _all_skills,
        "expanded_skills": _expanded_skill_list,
        "companies": f.companies_worked_at,
        "expanded_companies": _expanded_company_list,
        "certifications": f.certifications,
        "city": original_city,
        "state": original_state,
        "country": original_country,
        "min_years": exp.min_years,
        "max_years": exp.max_years,
        "titles": f.job_titles,
    }
    reranked = smart_rerank(
        search_result.results, smart_filters, body.location_preference
    )

    # ── 4a. Location grouping — group exact-location matches at the top ──────────
    # Applies in both "preferred" and "must_match" modes (not "remote").
    # Within each tier, candidates are sorted by score (highest first).
    # Tier 0 = exact city match, Tier 1 = state match, Tier 2 = country match, Tier 3 = other.
    #
    # RELEVANCE FLOOR: Profiles with score < 0.30 are NOT promoted by location.
    # This prevents irrelevant profiles (e.g. "Inspector" for a "nodejs developer"
    # search) from appearing at the top just because they're in the right city.
    _LOCATION_TIER_MIN_SCORE = 0.30

    if body.location_preference != "remote" and (original_city or original_state or original_country):
        _loc_city = (original_city or "").lower()
        _loc_state = (original_state or "").lower()
        _loc_country = (original_country or "").lower()

        def _integration_location_tier(r):
            """Return tier for grouping: 0=exact city, 1=state, 2=country, 3=other.
            Profiles below the relevance floor stay in tier 3 regardless of location."""
            # Relevance floor: don't promote irrelevant profiles
            if r.score < _LOCATION_TIER_MIN_SCORE:
                return 3

            r_city = (r.city or "").lower()
            r_state = (getattr(r, "state", "") or "").lower()
            r_country = (r.country or "").lower()
            r_location = (getattr(r, "location", "") or "").lower()
            r_area = (getattr(r, "area", "") or "").lower()
            r_linkedin_area = (getattr(r, "linkedin_area", "") or "").lower()

            # Tier 0: exact city match (multiple field checks)
            if _loc_city:
                if _loc_city == r_city or _loc_city in r_city:
                    return 0
                if _loc_city in r_location or _loc_city in r_area or _loc_city in r_linkedin_area:
                    return 0
            # Tier 1: state match
            if _loc_state:
                if _loc_state == r_state or _loc_state in r_location:
                    return 1
            # Tier 2: country match
            if _loc_country:
                if _loc_country == r_country or _loc_country in r_location:
                    return 2
            return 3

        reranked.sort(key=lambda r: (_integration_location_tier(r), -r.score))
        logger.info("Location grouping applied (min_score=%.2f): city='%s', state='%s', country='%s' — "
                     "tier counts: %s",
                     _LOCATION_TIER_MIN_SCORE, _loc_city, _loc_state, _loc_country,
                     {t: sum(1 for r in reranked if _integration_location_tier(r) == t) for t in range(4)})

    # ── 4b. Experience post-filter — catch profiles with null/stale Qdrant data ──
    if exp.min_years is not None or exp.max_years is not None:
        before_exp = len(reranked)
        def _exp_passes(r):
            yrs = r.years_experience
            if yrs is None:
                return False  # Unknown experience doesn't pass strict filter
            if exp.min_years is not None and yrs < exp.min_years:
                return False
            if exp.max_years is not None and yrs > exp.max_years:
                return False
            return True
        reranked = [r for r in reranked if _exp_passes(r)]
        logger.info("Experience post-filter: %d → %d (min=%s, max=%s)",
                    before_exp, len(reranked), exp.min_years, exp.max_years)

    # ── 5. Explicit match: strict post-filter ────────────────────────────────
    if body.explicit_match:
        reranked = _apply_explicit_filter(reranked, f)

    # ── 6. Location must_match post-filter ───────────────────────────────────
    if body.location_preference == "must_match":
        target_loc = (original_city or original_state or original_country or "").lower()
        if target_loc:
            before = len(reranked)
            reranked = [
                r for r in reranked
                if target_loc in (r.city or "").lower()
                or target_loc in (getattr(r, "state", "") or "").lower()
                or target_loc in (getattr(r, "location", "") or "").lower()
                or target_loc in (r.country or "").lower()
                or target_loc in (getattr(r, "linkedin_area", "") or "").lower()
                or target_loc in (getattr(r, "area", "") or "").lower()
            ]
            logger.info("must_match location filter: %d → %d", before, len(reranked))

    # ── 7. Paginate ──────────────────────────────────────────────────────────
    total_matches = len(reranked)
    start_idx = (body.page - 1) * body.limit
    end_idx = start_idx + body.limit
    paged = reranked[start_idx:end_idx]

    if not paged:
        return {
            "header": {"status": 200, "message": "Success"},
            "data": {
                "profiles": [],
                "total": total_matches,
                "returned": 0,
                "page": body.page,
                "took_ms": search_result.took_ms,
            },
        }

    # ── 8. Re-hydrate with FULL profile data (no truncation) ─────────────────
    # search_v2 caps skills at 15 and work_history at 10 for perf reasons.
    # For the integration response we want complete data.
    person_ids = [c.forager_id for c in paged]
    score_map = {c.forager_id: c.score for c in paged}

    try:
        full_profiles = _hydrate_full_profiles(person_ids, client_id)
    except Exception as exc:
        logger.error("Integration hydration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Profile hydration error.")

    # ── 9. Build sampleresponse.json-shaped profiles ──────────────────────────
    profiles: List[Dict] = []
    for pid in person_ids:          # preserve ranked order
        raw = full_profiles.get(pid)
        if raw is None:
            continue
        profiles.append(_build_profile(pid, score_map.get(pid, 0.0), raw))

    return {
        "header": {
            "status": 200,
            "message": "Success",
        },
        "data": {
            "profiles": profiles,
            "total": total_matches,
            "returned": len(profiles),
            "page": body.page,
            "took_ms": search_result.took_ms,
        },
    }


# =============================================================================
# HEALTH CHECK (unauthenticated — lets the mediator probe liveness)
# =============================================================================

@router.get("/health")
async def integration_health():
    """
    Lightweight health check for the mediator backend.
    Does NOT require authentication.
    """
    return {"status": "ok"}

