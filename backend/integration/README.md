# Integration API — Mediator Contract

## Endpoint

```
POST /api/integration/search
GET  /api/integration/health   (no auth required — liveness probe)
```

Base URL: `https://<your-python-backend-ec2-ip-or-domain>:<port>`

---

## Authentication

Every request to `/api/integration/search` must include:

```
X-Integration-Key: <INTEGRATION_API_KEY>
```

Or alternatively:

```
Authorization: Bearer <INTEGRATION_API_KEY>
```

The key is set in `.env` on the Python server as `INTEGRATION_API_KEY`.

---

## Request Schema

```typescript
interface LocationFilter {
  country?: string;   // e.g. "United States"
  state?: string;     // e.g. "California"
  city?: string;      // e.g. "San Francisco"
}

interface ExperienceFilter {
  min_years?: number;
  max_years?: number;
}

interface SearchFilters {
  skills_required?: string[];       // AND — must have ALL of these
  skills_nice_to_have?: string[];   // Boosts ranking, not required
  skills_exclude?: string[];        // Filter out profiles with these skills
  location?: LocationFilter;
  experience?: ExperienceFilter;
  job_titles?: string[];            // e.g. ["Senior Product Designer"]
  companies_worked_at?: string[];   // Current OR past
  companies_exclude?: string[];
  certifications?: string[];        // e.g. ["PMP", "AWS Certified"]
}

interface IntegrationSearchRequest {
  search_text: string;              // Free-form semantic query
  filters?: SearchFilters;
  limit?: number;                   // Default 50, max 500
  page?: number;                    // 1-based, default 1
  client_id?: string;               // Override tenant scope (optional)
}
```

### Example Request Body

```json
{
  "search_text": "Senior Product Designer with Figma and UX research experience",
  "filters": {
    "skills_required": ["Figma", "UX Research"],
    "skills_nice_to_have": ["Prototyping", "Design Systems"],
    "location": {
      "country": "United States",
      "state": "California"
    },
    "experience": {
      "min_years": 3,
      "max_years": 10
    },
    "job_titles": ["Product Designer", "Senior Product Designer", "Lead Product Designer"]
  },
  "limit": 50,
  "page": 1
}
```

---

## Response Schema

Matches `sampleresponse.json` exactly.

```typescript
interface EmployerEntry {
  name: string;
  linkedin_id: string;
  company_id: number;
  company_linkedin_id: string;
  company_website_domain: string;
  position_id: number;
  title: string;
  description: string;
  location: string;
  start_date: string | null;
  end_date?: string | null;        // only on past_employers
  employer_is_default: boolean;
  seniority_level: string;
  function_category: string;
  years_at_company: string;
  years_at_company_raw: number;
  company_headquarters_country: string;
  company_hq_location: string;
  company_hq_location_address_components: string[];
  company_headcount_range: string;
  company_industries: string[];
  company_linkedin_industry: string;
  company_type: string;
  company_headcount_latest: number;
  company_website: string;
  company_linkedin_profile_url: string;
  business_email_verified: boolean;
}

interface EducationEntry {
  degree_name: string;
  institute_name: string;
  institute_linkedin_id: string;
  institute_linkedin_url: string;
  institute_logo_url: string;
  field_of_study: string;
  activities_and_societies: string;
  start_date: string | null;
  end_date: string | null;
}

interface LocationDetails {
  city: string;
  state: string;
  country: string;
  continent: string;
}

interface Profile {
  person_id: number;
  _score: number;                        // Talentin relevance score 0-1
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
  education_background: EducationEntry[];
  honors: any[];
  certifications: string[];
  current_employers: EmployerEntry[];
  past_employers: EmployerEntry[];
  last_updated: string | null;
  recently_changed_jobs: boolean;
  years_of_experience: string;
  years_of_experience_raw: number;
  all_employers: EmployerEntry[];
  updated_at: string | null;
  location_details: LocationDetails;
}

interface IntegrationSearchResponse {
  header: {
    status: number;    // 200 on success
    message: string;   // "Success"
  };
  data: {
    profiles: Profile[];
    total: number;     // Total matching (before pagination)
    returned: number;  // Profiles in this response
    page: number;
    took_ms: number;
  };
}
```

---

## TypeScript Mediator Example (Node.js / fetch)

```typescript
const SEARCH_SERVER_URL = process.env.SEARCH_SERVER_URL!;
const INTEGRATION_API_KEY = process.env.INTEGRATION_API_KEY!;

async function searchCandidates(
  searchText: string,
  filters?: SearchFilters,
  page = 1,
  limit = 50
): Promise<IntegrationSearchResponse> {
  const res = await fetch(`${SEARCH_SERVER_URL}/api/integration/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Integration-Key": INTEGRATION_API_KEY,
    },
    body: JSON.stringify({ search_text: searchText, filters, page, limit }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`Search failed [${res.status}]: ${err.detail ?? res.statusText}`);
  }

  return res.json() as Promise<IntegrationSearchResponse>;
}
```

---

## Security Notes

- The Python server should be in a **private subnet** or have its **security group** set to only accept traffic from the mediator backend's IP.
- The `INTEGRATION_API_KEY` should be a random 32+ byte hex string (never reuse across environments).
- Rotate the key by updating `.env` on the Python server and the mediator's environment variables simultaneously.
- CORS is irrelevant for server-to-server calls (it only applies to browsers). Only set `MEDIATOR_ORIGIN` if a browser ever calls this server directly.
