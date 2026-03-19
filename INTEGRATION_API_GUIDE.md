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
 
**Headers:**
 
| Header | Value | Required |
|--------|-------|----------|
| `X-Integration-Key` | `<api_key>` | **YES** |
 
**Response:**
 
```json
{ "status": "ok" }
```
 
---
 
## Endpoint 2: Search
 
```
POST /api/v2/search
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
    "skills": {
      "must_have": ["React", "TypeScript"],
      "nice_to_have": ["Next.js", "GraphQL"],
      "exclude": ["PHP"]
    },
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
    "companies": {
      "worked_at": ["Google", "Meta"],
      "current_only": false,
      "exclude": ["Oracle"]
    },
    "schools": ["Stanford University"],
    "certifications": ["AWS Certified"],
    "first_name": null,
    "last_name": null
  },
  "options": {
    "limit": 50,
    "offset": 0,
    "expand_skills": true
  },
  "client_id": null
}
```
 
---
 
## Request Fields — Detailed Breakdown
 
### Top-Level Fields
 
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `search_text` | `string` | **YES** (can be `""`) | `""` | Semantic search text. Vector-embedded for similarity matching. |
| `filters` | `object` | Optional | `{}` | All filters (skills, location, etc.). |
| `options.limit` | `int` | Optional | `50` | Results count (1-500). |
| `options.offset` | `int` | Optional | `0` | Pagination offset. |
| `options.expand_skills` | `bool` | Optional | `true` | Expand skills via graph (React → React.js). |
| `client_id` | `string` | Optional | API key mapped | Tenant scope. |
 
### Filters Breakdown
 
#### Skills (`filters.skills`)
 
| Field | Type | Description |
|-------|------|-------------|
| `must_have` | `string[]` | **Required** (AND logic). **Hard filter**. |
| `nice_to_have` | `string[]` | Ranking boost only. |
| `exclude` | `string[]` | **Reject** profiles with these. |
 
#### Location (`filters.location`)
 
| Field | Type | Description |
|-------|------|-------------|
| `city` | `string` | `"Los Angeles"` |
| `state` | `string` | `"California"` |
| `country` | `string` | `"United States"` |
 
#### Other Filters
 
| Field | Type | Description |
|-------|------|-------------|
| `experience.min_years` | `int` | Min experience (hard filter). |
| `job_titles` | `string[]` | Semantic title matching. |
| `companies.worked_at` | `string[]` | Past/current companies. |
| `certifications` | `string[]` | Certification matching. |
 
---
 
## Example Test Query (React Developer + LA)
 
```bash
curl -X POST https://jp.talentin.ai/api/v2/search \\
  -H 'X-Integration-Key: YOUR_KEY' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "search_text": "react developer",
    "filters": {
      "location": {"city": "Los Angeles"},
      "skills": {"must_have": ["react"]},
      "experience": {"min_years": 3}
    },
    "options": {"limit": 20}
  }'
```
 
**Expected**: **No IT Technician** (skills don't match "react").
 
---
 
## Response Format (Same as before)
 
```json
{
  "header": { "status": 200, "message": "Success" },
  "data": {
    "total": 25,
    "returned": 20,
    "took_ms": 850,
    "profiles": [ ... ]  // React devs in LA only!
  }
}
```
 
**Dashboard Test**: Save as `test_react_la.json` → Import to frontend.

**DONE** — Full API Reference + Test Query provided!

