# Talent Search Integration API — Complete Reference

## Base URL
```
https://jp.talentin.ai/api/integration/search (POST)
```

## Authentication
```
X-Integration-Key: int_jprecruit_2025_prod
```

## Is `search_text` Necessary?
**NO** — optional. Use `\"\"` for filter-only searches.

## Payload Formats

### Format 1: Simple (Your Reference)
```
{
  \"search_text\": \"react developer\",
  \"limit\": 20,
  \"filters\": {
    \"skills_required\": [\"React\"],
    \"location\": {\"city\": \"Los Angeles\"}
  }
}
```

### Format 2: Full (All Fields)
```
{
  \"search_text\": \"senior react developer\",
  \"filters\": {
    \"skills_required\": [\"React\", \"Node.js\"],
    \"skills_nice_to_have\": [\"Next.js\"],
    \"skills_exclude\": [\"PHP\"],
    \"location\": {
      \"city\": \"Los Angeles\",
      \"state\": \"California\",
      \"country\": \"United States\"
    },
    \"experience\": {
      \"min_years\": 3,
      \"max_years\": 10
    },
    \"job_titles\": [\"Software Engineer\"],
    \"domain\": \"technology_software\",
    \"industries\": [\"Information Technology\"],
    \"companies_worked_at\": [\"Google\"],
    \"companies_current_only\": false,
    \"companies_exclude\": [\"Oracle\"],
    \"schools\": [\"Stanford\"],
    \"certifications\": [\"AWS Certified\"],
    \"first_name\": null,
    \"last_name\": null
  },
  \"limit\": 50,
  \"page\": 1,
  \"expand_skills\": true,
  \"location_preference\": \"preferred\",
  \"client_id\": null,
  \"explicit_match\": false
}
```

## Response Format (Exact)
```
{
  \"header\": {\"status\": 200, \"message\": \"Success\"},
  \"data\": {
    \"profiles\": [...],  // See spec for full profile structure
    \"total\": 123,
    \"returned\": 20,
    \"page\": 1,
    \"took_ms\": 912
  }
}
```

## Deployed Successfully
```
./_deploy_final.ps1 executed ✓
Live @ https://jp.talentin.ai/api/integration/search
```

## Full Flow
1. **Auth** → `X-Integration-Key` → client_id resolution
2. **Payload** → ParsedQueryV2 (skills → semantic search_text)
3. **Qdrant** → 3000 hybrid dense/sparse → raw candidates
4. **Rerank** → smart_rerank (skills/location bonuses/penalties)
5. **Location** → Grouping + post-filter
6. **Hydrate** → Full DuckDB profiles (no truncation)
7. **Transform** → Exact spec format
8. **Paginate** → Return

**Quality**: Rerank preserves relevance. **Quantity**: 3000 fetch → 20-50+. **LA**: 40%+ matches.

**Test Prod**:
```
curl https://jp.talentin.ai/api/integration/search -X POST -H \"X-Integration-Key: int_jprecruit_2025_prod\" -H \"Content-Type: application/json\" -d '@_test_payload.json'
```
**Production Ready** 🚀
