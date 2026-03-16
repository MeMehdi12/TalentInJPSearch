# Talent Search Integration API — Complete Reference

## Base URL

```
https://jp.talentin.ai
```

> **Note:** Port 8001 is not exposed externally. All requests must go through nginx on port 443 (HTTPS).

## Authentication

Every request must include **one** of these headers:

| Method | Header |
|--------|--------|
| Custom header (preferred) | `X-Integration-Key: <api_key>` |
| Bearer token | `Authorization: Bearer <api_key>` |

Missing or invalid key → `401 Unauthorized`.

---

## Endpoint 1: Health Check

```
GET /api/integration/health
```

**Headers:** None

This endpoint is intentionally unauthenticated and is meant for liveness/readiness checks from the mediator or load balancer.

**Response:**

```json
{ "status": "ok" }
```

---

## Endpoint 2: Search

```
POST /api/integration/search
```

**Headers:**

| Header | Value | Required |
|--------|-------|----------|
| `X-Integration-Key` | `<api_key>` | **YES** |
| `Content-Type` | `application/json` | **YES** |

---

## Request Body — Full Payload (all fields)

```json
{
  "search_text": "senior react developer with AWS experience",
  "filters": {
    "skills_required": ["React", "TypeScript"],
    "skills_nice_to_have": ["Next.js", "GraphQL"],
    "skills_exclude": ["PHP"],
    "location": {
      "city": "San Francisco",
      "state": "California",
      "country": "United States"
    },
    "experience": {
      "min_years": 3,
      "max_years": 10
    },
    "job_titles": ["Software Engineer", "Frontend Developer"],
    "domain": "technology_software",
    "industries": ["Information Technology"],
    "companies_worked_at": ["Google", "Meta"],
    "companies_current_only": false,
    "companies_exclude": ["Oracle"],
    "schools": ["Stanford University"],
    "certifications": ["AWS Certified"],
    "first_name": null,
    "last_name": null
  },
  "limit": 50,
  "page": 1,
  "expand_skills": true,
  "client_id": null,
  "location_preference": "preferred",
  "explicit_match": false
}
```

---

## Request Fields — Detailed Breakdown

### Top-Level Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `search_text` | `string` | **COMPULSORY** (can be `""`) | `""` | Natural language description of ideal candidate. Gets vector-embedded locally (no OpenAI). Use empty string `""` for filter-only searches. |
| `filters` | `object` | Optional | `null` | Filter object. All sub-fields are optional. |
| `limit` | `integer` | Optional | `50` | Results per page. Range: **1–500**. |
| `page` | `integer` | Optional | `1` | Page number (**1-based**). |
| `expand_skills` | `boolean` | Optional | `true` | Auto-expand skills via skill-relationship graph (e.g. "React" also matches "React.js", "ReactJS"). No OpenAI call. |
| `client_id` | `string` | Optional | `null` | Tenant scope override. **Don't send this** — it's auto-resolved from the API key. Only use if explicitly needed. |
| `location_preference` | `string` | Optional | `"preferred"` | Location handling mode — see **Location Modes** below. |
| `explicit_match` | `boolean` | Optional | `false` | Strict mode: ALL `skills_required` must appear in profile skills, and at least one `job_titles` keyword must match current title or headline. Candidates that don't pass are excluded. |

### Location Modes (`location_preference`)

| Mode | Behaviour |
|------|-----------|
| `"preferred"` (default) | Location is used for **semantic similarity + ranking boost** but does NOT hard-filter. Candidates everywhere are returned; local ones are ranked higher (city +0.12, state +0.06, country +0.03). |
| `"must_match"` | Hard post-filter — only candidates whose city/state/country matches the location filter are returned. Strong ranking boost (+0.60 city, +0.30 state, +0.15 country). |
| `"remote"` | Location is completely ignored — no boost, no filter. Useful when location doesn't matter. |

### Filter Fields (`filters.*`)

#### Skills

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.skills_required` | `string[]` | Optional | `[]` | Target skills. Handled via **semantic search + re-ranking** — profiles with these skills rank highest, but you won't get 0 results from typos or naming variations (e.g. `"next"` finds Next.js developers). Use `explicit_match: true` for strict AND enforcement. |
| `filters.skills_nice_to_have` | `string[]` | Optional | `[]` | Preferred skills. Boosts ranking but does NOT exclude candidates missing them. |
| `filters.skills_exclude` | `string[]` | Optional | `[]` | Reject profiles that have **ANY** of these skills. This is a **hard filter** — profiles are removed entirely. |

#### Location

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.location` | `object` | Optional | `null` | Location filter. All sub-fields optional. |
| `filters.location.city` | `string` | Optional | `null` | City name (e.g. `"San Francisco"`) |
| `filters.location.state` | `string` | Optional | `null` | State/province (e.g. `"California"`) |
| `filters.location.country` | `string` | Optional | `null` | Country (e.g. `"United States"`, `"Japan"`) |

#### Experience

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.experience` | `object` | Optional | `null` | Experience range filter. Both sub-fields optional. |
| `filters.experience.min_years` | `integer` | Optional | `null` | Minimum years of experience. Range: 0–60. |
| `filters.experience.max_years` | `integer` | Optional | `null` | Maximum years of experience. Range: 0–60. |

#### Job & Domain

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.job_titles` | `string[]` | Optional | `[]` | Target job title keywords (e.g. `["Software Engineer", "Developer"]`) |
| `filters.domain` | `string` | Optional | `null` | Industry domain (e.g. `"technology_software"`) |
| `filters.industries` | `string[]` | Optional | `[]` | Industry names to match (e.g. `["Information Technology"]`) |

#### Companies

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.companies_worked_at` | `string[]` | Optional | `[]` | Must have worked at these companies (current OR past). |
| `filters.companies_current_only` | `boolean` | Optional | `false` | If `true`, `companies_worked_at` only matches **current** employer. |
| `filters.companies_exclude` | `string[]` | Optional | `[]` | Exclude profiles from these companies. |

#### Education & Credentials

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.schools` | `string[]` | Optional | `[]` | Universities/schools to filter by (e.g. `["Stanford University"]`) |
| `filters.certifications` | `string[]` | Optional | `[]` | Required certifications (e.g. `["PMP", "AWS Certified"]`) |

#### Person Name

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters.first_name` | `string` | Optional | `null` | Exact first name filter. |
| `filters.last_name` | `string` | Optional | `null` | Exact last name filter. |

---

## Example Requests

### Minimal — just semantic search

```json
{
  "search_text": "java backend developer"
}
```

### Filter-only — no semantic text

```json
{
  "search_text": "",
  "filters": {
    "skills_required": ["Python", "Django"],
    "location": { "country": "Japan" }
  },
  "limit": 20,
  "page": 1
}
```

### Combined — semantic + filters

```json
{
  "search_text": "senior react developer with cloud experience",
  "filters": {
    "skills_required": ["React"],
    "skills_nice_to_have": ["AWS", "GCP"],
    "experience": { "min_years": 5 },
    "location": { "country": "United States" }
  },
  "limit": 50,
  "page": 1
}
```

### Pagination — page 2

```json
{
  "search_text": "data scientist",
  "limit": 20,
  "page": 2
}
```

### Explicit match — strict skill + title filtering

```json
{
  "search_text": "python backend developer",
  "filters": {
    "skills_required": ["Python", "Django", "PostgreSQL"],
    "job_titles": ["Backend Developer", "Software Engineer"]
  },
  "explicit_match": true,
  "limit": 20
}
```

### Location must-match — only candidates in Tokyo

```json
{
  "search_text": "react developer",
  "filters": {
    "skills_required": ["React"],
    "location": { "city": "Tokyo", "country": "Japan" }
  },
  "location_preference": "must_match",
  "limit": 30
}
```

### Remote mode — location irrelevant

```json
{
  "search_text": "machine learning engineer",
  "filters": {
    "skills_required": ["PyTorch", "Python"]
  },
  "location_preference": "remote",
  "limit": 50
}
```

---

## Response — Full Structure

```json
{
  "header": {
    "status": 200,
    "message": "Success"
  },
  "data": {
    "total": 2000,
    "returned": 2,
    "page": 1,
    "took_ms": 920,
    "profiles": [
      {
        "person_id": 123456,
        "_score": 0.8721,
        "name": "John Doe",
        "first_name": "John",
        "last_name": "Doe",
        "region": "San Francisco Bay Area",
        "region_address_components": ["San Francisco", "California", "United States"],
        "headline": "Senior Software Engineer at Google",
        "summary": "Experienced developer with 8 years in full-stack development...",
        "skills": ["React", "TypeScript", "Node.js", "Python", "AWS"],
        "languages": [],
        "profile_language": "",
        "linkedin_profile_url": "https://www.linkedin.com/in/johndoe",
        "flagship_profile_url": "https://www.linkedin.com/in/johndoe",
        "emails": [],
        "profile_picture_url": "https://media.licdn.com/dms/image/...",
        "profile_picture_permalink": "https://media.licdn.com/dms/image/...",
        "twitter_handle": "",
        "open_to_cards": [],
        "num_of_connections": 0,
        "education_background": [
          {
            "degree_name": "Bachelor of Science",
            "institute_name": "Stanford University",
            "field_of_study": "Computer Science",
            "start_date": "2012",
            "end_date": "2016"
          }
        ],
        "honors": [],
        "certifications": ["AWS Certified Solutions Architect"],
        "current_employers": [
          {
            "name": "Google",
            "title": "Senior Software Engineer",
            "start_date": "2020-01-01",
            "end_date": null,
            "is_current": true,
            "years_at_company": "4 years",
            "years_at_company_raw": 4
          }
        ],
        "past_employers": [
          {
            "name": "Meta",
            "title": "Software Engineer",
            "start_date": "2017-06-01",
            "end_date": "2019-12-01",
            "is_current": false,
            "years_at_company": "2 years",
            "years_at_company_raw": 2
          }
        ],
        "all_employers": [
          {
            "name": "Google",
            "title": "Senior Software Engineer",
            "start_date": "2020-01-01",
            "end_date": null,
            "is_current": true,
            "years_at_company": "4 years",
            "years_at_company_raw": 4
          },
          {
            "name": "Meta",
            "title": "Software Engineer",
            "start_date": "2017-06-01",
            "end_date": "2019-12-01",
            "is_current": false,
            "years_at_company": "2 years",
            "years_at_company_raw": 2
          }
        ],
        "last_updated": "2024-11-15T08:30:00",
        "updated_at": "2024-11-15T08:30:00",
        "recently_changed_jobs": false,
        "years_of_experience": "8 years",
        "years_of_experience_raw": 8,
        "location_details": {
          "city": "San Francisco",
          "state": "California",
          "country": "United States",
          "continent": ""
        }
      }
    ]
  }
}
```

---

## Response Fields — Detailed Breakdown

### Envelope

| Field | Type | Description |
|-------|------|-------------|
| `header.status` | `integer` | HTTP status code (200) |
| `header.message` | `string` | `"Success"` |
| `data.total` | `integer` | Total matching candidates in the database |
| `data.returned` | `integer` | Number of profiles returned on this page |
| `data.page` | `integer` | Current page number |
| `data.took_ms` | `float` | Server-side processing time in milliseconds |

### Profile Object (`data.profiles[]`)

| Field | Type | Description |
|-------|------|-------------|
| `person_id` | `integer` | Unique candidate ID |
| `_score` | `float` | Relevance score **0.0–1.0** (higher = better match). Three-layer scoring: base vector similarity normalized to [0.10, 0.65], light skill-coverage pass, then comprehensive re-ranking across skills, titles, location, companies, experience, certifications, and multi-axis combo bonus. Scores are sorted by raw accumulated value, then capped at 1.0 for display. |
| `name` | `string` | Full name |
| `first_name` | `string` | Parsed first name |
| `last_name` | `string` | Parsed last name |
| `region` | `string` | Human-readable location (e.g. "San Francisco Bay Area") |
| `region_address_components` | `string[]` | Non-null parts of [city, state, country] |
| `headline` | `string` | LinkedIn headline |
| `summary` | `string` | Profile description/bio |
| `skills` | `string[]` | All skills (no truncation in integration API) |
| `languages` | `string[]` | **Reserved — always `[]` for now** |
| `profile_language` | `string` | **Reserved — always `""` for now** |
| `linkedin_profile_url` | `string` | LinkedIn profile URL |
| `flagship_profile_url` | `string` | LinkedIn public URL (from slug) |
| `emails` | `string[]` | **Reserved — always `[]` for now** |
| `profile_picture_url` | `string` | Profile photo URL |
| `profile_picture_permalink` | `string` | Same as `profile_picture_url` |
| `twitter_handle` | `string` | **Reserved — always `""` for now** |
| `open_to_cards` | `any[]` | **Reserved — always `[]` for now** |
| `num_of_connections` | `integer` | **Reserved — always `0` for now** |
| `education_background` | `Education[]` | List of education entries |
| `honors` | `any[]` | **Reserved — always `[]` for now** |
| `certifications` | `string[]` | List of certification names |
| `current_employers` | `Employer[]` | Current job(s) |
| `past_employers` | `Employer[]` | Previous jobs |
| `all_employers` | `Employer[]` | `current_employers` + `past_employers` combined |
| `last_updated` | `string \| null` | ISO datetime of last profile update |
| `updated_at` | `string \| null` | Same as `last_updated` |
| `recently_changed_jobs` | `boolean` | **Reserved — always `false` for now** |
| `years_of_experience` | `string` | Human-readable (e.g. "5 years", "10+ years") |
| `years_of_experience_raw` | `integer` | Numeric years |
| `location_details` | `object` | Structured location |
| `location_details.city` | `string` | City |
| `location_details.state` | `string` | State/province |
| `location_details.country` | `string` | Country |
| `location_details.continent` | `string` | **Reserved — always `""` for now** |

### Education Object (`education_background[]`)

| Field | Type | Description |
|-------|------|-------------|
| `degree_name` | `string` | e.g. "Bachelor of Science" |
| `institute_name` | `string` | e.g. "Stanford University" |
| `field_of_study` | `string` | e.g. "Computer Science" |
| `start_date` | `string \| null` | Start year/date |
| `end_date` | `string \| null` | End year/date |

### Employer Object (`current_employers[]`, `past_employers[]`, `all_employers[]`)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Company name |
| `title` | `string` | Job title |
| `start_date` | `string \| null` | Start date |
| `end_date` | `string \| null` | End date (`null` = current) |
| `is_current` | `boolean` | `true` if currently employed here |
| `years_at_company` | `string` | Human-readable (e.g. "3 years") |
| `years_at_company_raw` | `integer` | Numeric years |

---

## Error Responses

| Status | Cause | Example |
|--------|-------|---------|
| `401` | Missing or invalid API key | `{"detail": "Invalid integration key"}` |
| `422` | Malformed request body (validation error) | Pydantic error details |
| `500` | Server-side error (search engine or DB) | `{"detail": "Search engine error. Please try again."}` |

---

## Pagination

- `page` is **1-based** (`page: 1` = first page)
- `limit` controls page size (1–500, default 50)
- Total pages: `Math.ceil(data.total / limit)`
- `data.total` = total matching candidates
- `data.returned` = how many came back on this page

---

## Notes

1. **API key** — store in env var on mediator server, never expose to frontend
2. **Cold start** — first request after service restart takes ~15–30s (model loading). Subsequent: 500ms–2s
3. **No OpenAI** — the integration path uses local SentenceTransformer embeddings only
4. **No rate limiting** — implement on mediator side if needed
5. **CORS** — not needed for server-to-server calls
6. **Reserved fields** (marked "always [] / "" / 0 for now") — typed for future compatibility, no breaking changes when populated later
7. **Security** — restrict port 8001 in AWS Security Group to mediator's IP only

---

## How Search Works

The integration API uses **semantic vector search**, not keyword matching. Here's what that means:

### Semantic search (what `search_text` + `skills_required` + `location` do)

Every query is converted into a **768-dimensional vector** by a local SentenceTransformer model (`all-mpnet-base-v2`). This vector captures the *meaning* of the entire query — skills, job titles, locations, domain — all at once. That vector is then compared against pre-computed vectors for all 253K+ profiles in the database.

This is why:
- `"next"` correctly finds Next.js developers (semantically similar)
- `"ML engineer"` finds "Machine Learning Engineer" profiles (same meaning, different words)
- `"python developer Tokyo"` finds Python developers in Tokyo without needing exact field matches

Skills, job titles, domain, certifications, and location are all **injected into the embedding text** so the vector captures the full intent. There are no hardcoded alias dictionaries in this path — the model understands language naturally.

### Hard filters (what still uses exact matching)

These fields are applied as **strict database filters** before vector search:

| Filter | Logic | Why it's strict |
|--------|-------|-----------------|
| `skills_exclude` | NOT — remove any profile with these skills | Exclusion should be absolute |
| `companies_worked_at` | AND — must have worked at these companies | Company names are stored consistently |
| `companies_exclude` | NOT — remove profiles from these companies | Exclusion should be absolute |
| `experience.min_years` / `max_years` | Range filter | Numeric, no ambiguity |
| `explicit_match: true` | Post-filter: ALL `skills_required` in profile + title match | Opt-in strict mode for when you need precision over recall |

### What this means in practice

- **Default mode**: Send skills/location/titles freely — the system understands intent and ranks by relevance. You'll always get results even with informal skill names.
- **`explicit_match: true`**: Strict mode — only returns candidates who literally have ALL required skills AND match a job title. Use when precision matters more than recall.
- **`location_preference: "must_match"`**: Hard location post-filter. Use when the candidate MUST be in a specific place.

---

## Ranking Architecture

Scores are computed in three layers:

| Layer | What it does | Score range |
|-------|-------------|-------------|
| **1. Vector similarity** | Qdrant search using dense semantic vectors (sentence-transformers/all-mpnet-base-v2). Skills, titles, location, domain, and certifications are all injected into the embedding text so the vector captures the full intent. Sparse BM25 vectors exist in the collection but the query-side vocabulary is not yet active — search is effectively dense-only with RRF fusion falling back gracefully. | Normalized to **0.10 – 0.65** |
| **2. Light skill pass** | Word-boundary skill matching bonus based on coverage of `skills_required` | Up to **+0.08** |
| **3. Smart re-rank** | Comprehensive multi-signal re-ranking (see below) | Up to **~0.35** |

### Smart Re-Rank Signals

| Signal | Max bonus | Details |
|--------|-----------|----------|
| Skill exact match | +0.10 | Coverage ratio × 0.10 for skills in profile `skills[]` array |
| Skill text fallback | +0.04 | Skills found in headline/title/description (lower confidence) |
| Expanded skill match | +0.04 | Related skills from skill graph (only if no exact matches) |
| Title exact match | +0.10 | Target title is a substring of current title |
| Title partial match | +0.06 | ≥50% keyword overlap with current title |
| Title in headline | +0.04 | Target title appears in LinkedIn headline |
| Location (preferred) | +0.12 / +0.06 / +0.03 | City / state / country match |
| Location (must_match) | +0.60 / +0.30 / +0.15 | City / state / country match (hard filter also applied) |
| Current company | +0.06 | Target company matches current employer |
| Past company | +0.04 | Target company found in work history |
| Certifications | +0.10 | Coverage ratio × 0.10 across target certs |
| Experience in-range | +0.06 | Years of experience falls within min–max range |
| Experience proximity | +0.04 | Continuous decay for out-of-range (closer → higher) |
| Profile completeness | +0.02 | Profiles with >70% completeness |
| Multi-axis combo (4+) | +0.06 | Matches on 4+ of: skills, title, location, company, experience, certs |
| Multi-axis combo (3) | +0.04 | Matches on exactly 3 axes |
| Multi-axis combo (2) | +0.02 | Matches on exactly 2 axes |

**Theoretical max score: ~1.0** (base 0.65 + bonuses ~0.35). Scores are sorted by raw total, then capped at 1.0 for display.

### Skill Expansion

When `expand_skills: true` (default), each requested skill is expanded via a pre-computed skill-relationship graph (cosine similarity ≥ 0.55). For example, `"React"` also matches `"React.js"`, `"ReactJS"`, `"React Native"`. This runs locally with no external API call.

---

## TypeScript Interfaces

```typescript
// ─── REQUEST ──────────────────────────────────────────

interface SearchRequest {
  search_text: string;               // COMPULSORY (can be "")
  filters?: SearchFilters;
  limit?: number;                    // 1–500, default 50
  page?: number;                     // 1-based, default 1
  expand_skills?: boolean;           // default true
  client_id?: string;                // don't send — auto-resolved from key
  location_preference?: 'preferred' | 'must_match' | 'remote';  // default "preferred"
  explicit_match?: boolean;          // default false — strict skill+title filtering
}

interface SearchFilters {
  skills_required?: string[];
  skills_nice_to_have?: string[];
  skills_exclude?: string[];
  location?: {
    city?: string;
    state?: string;
    country?: string;
  };
  experience?: {
    min_years?: number;
    max_years?: number;
  };
  job_titles?: string[];
  domain?: string;
  industries?: string[];
  companies_worked_at?: string[];
  companies_current_only?: boolean;
  companies_exclude?: string[];
  schools?: string[];
  certifications?: string[];
  first_name?: string;
  last_name?: string;
}

// ─── RESPONSE ─────────────────────────────────────────

interface SearchResponse {
  header: {
    status: number;
    message: string;
  };
  data: {
    profiles: Profile[];
    total: number;
    returned: number;
    page: number;
    took_ms: number;
  };
}

interface Profile {
  person_id: number;
  _score: number;
  name: string;
  first_name: string;
  last_name: string;
  region: string;
  region_address_components: string[];
  headline: string;
  summary: string;
  skills: string[];
  languages: string[];
  profile_language: string;
  linkedin_profile_url: string;
  flagship_profile_url: string;
  emails: string[];
  profile_picture_url: string;
  profile_picture_permalink: string;
  twitter_handle: string;
  open_to_cards: any[];
  num_of_connections: number;
  education_background: Education[];
  honors: any[];
  certifications: string[];
  current_employers: Employer[];
  past_employers: Employer[];
  all_employers: Employer[];
  last_updated: string | null;
  updated_at: string | null;
  recently_changed_jobs: boolean;
  years_of_experience: string;
  years_of_experience_raw: number;
  location_details: {
    city: string;
    state: string;
    country: string;
    continent: string;
  };
}

interface Education {
  degree_name: string;
  institute_name: string;
  field_of_study: string;
  start_date: string | null;
  end_date: string | null;
}

interface Employer {
  name: string;
  title: string;
  start_date: string | null;
  end_date: string | null;
  is_current: boolean;
  years_at_company: string;
  years_at_company_raw: number;
}
```
