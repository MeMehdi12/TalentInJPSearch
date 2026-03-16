# Talentin Pipeline

Modular data pipeline to manage clients and their candidate data across DuckDB and Qdrant.

---

## Files

| File | Purpose | When to run |
|---|---|---|
| `run_pipeline.py` | CLI orchestrator — single entry point | every time |
| `migrate_db.py` | Adds `client_id` column to DuckDB | **one-time** after first deployment |
| `backfill_qdrant.py` | Stamps `client_id` on existing Qdrant points | **one-time** after first deployment |
| `ingest_client.py` | Imports a client's candidate file → DuckDB + Qdrant | **every new dataset** |
| `clients.json` | Client registry (domains, display names) | update when adding clients |

---

## First-time Setup (existing database)

Run this once after deploying the multi-tenant code changes:

```bash
cd "D:\Talentin JPSEARCH DEPLOY"
python pipeline/run_pipeline.py full-setup --default-client talentin
```

This will:
1. Add `client_id` column to `processed_profiles` and backfill `talentin`
2. Stamp `client_id=talentin` on all Qdrant points that have no `client_id`

Dry-run first to check what will happen:
```bash
python pipeline/run_pipeline.py full-setup --dry-run
```

---

## Adding a New Client

**Step 1 — Register the client:**
```bash
python pipeline/run_pipeline.py add-client \
  --client-id acme \
  --display-name "Acme Corp" \
  --domains acme.com,acme.net
```

**Step 2 — Update frontend env** (so users who log in with `@acme.com` get client_id `acme`):  
In `frontend/.env` (or `.env.production`):
```
VITE_CLIENT_MAP={"acme.com":"acme","acme.net":"acme","talentin.ai":"talentin"}
```

**Step 3 — Ingest their candidate data:**
```bash
python pipeline/run_pipeline.py ingest \
  --client acme \
  --file data/acme_candidates.jsonl
```

---

## Input File Format

The ingest command accepts `.jsonl`, `.json` (array), or `.csv`.

**JSONL (recommended)** — one JSON object per line:
```json
{"person_id":"p001","full_name":"Jane Smith","skills":["python","django"],"current_title":"Engineer","current_company":"Acme","city":"Dubai","country":"UAE","years_experience":7}
{"person_id":"p002","full_name":"Ali Hassan","skills":["react","typescript"],"current_title":"Frontend Dev","current_company":"Acme","city":"Abu Dhabi","country":"UAE","years_experience":4}
```

**Full schema** (all optional except `full_name`):
```json
{
  "person_id":          "unique_string_or_int",
  "full_name":          "Jane Smith",
  "headline":           "Senior Python Developer @ Acme",
  "city":               "Dubai",
  "state":              "Dubai",
  "country":            "UAE",
  "skills":             ["python", "django", "postgres"],
  "years_experience":   7,
  "current_title":      "Senior Developer",
  "current_company":    "Acme Corp",
  "linkedin_url":       "https://linkedin.com/in/janesmith",
  "linkedin_slug":      "janesmith",
  "photo":              null,
  "industry":           "Technology",
  "domain":             "software_engineering",
  "profile_completeness": 85,
  "description":        "Bio text here",
  "area":               "UAE",
  "work_history": [
    {"title":"Dev","company":"OldCo","start_date":"2019-01","end_date":"2022-06"}
  ],
  "education": [
    {"school":"MIT","degree":"BS","field":"Computer Science","start_date":"2015","end_date":"2019"}
  ],
  "certifications": [
    {"name":"AWS SAA","authority":"Amazon","issued_date":null}
  ]
}
```

---

## Re-running After Data Updates

When you receive an updated dataset for an existing client, just re-run ingest — it uses upsert so existing records are overwritten cleanly:

```bash
python pipeline/run_pipeline.py ingest --client acme --file data/acme_v2.jsonl
```

To fix only DuckDB (skip re-embedding):
```bash
python pipeline/run_pipeline.py ingest --client acme --file data/acme_v2.jsonl --skip-qdrant
```

To fix only Qdrant vectors (skip DuckDB):
```bash
python pipeline/run_pipeline.py ingest --client acme --file data/acme_v2.jsonl --skip-duckdb
```

---

## Check Status

```bash
python pipeline/run_pipeline.py status
```

Output:
```
════════════════════════════════════════════════
  Talentin Pipeline — Database Status
════════════════════════════════════════════════

Client                     Candidates
─────────────────────────  ──────────
talentin                       12,450
acme                            3,200
──────────────────────────────────────
TOTAL                          15,650

Qdrant collection   : profiles_hybrid
  Total points      : 15,650
```

---

## Environment Variables

All settings are read from `backend/.env`:

| Variable | Purpose |
|---|---|
| `DUCKDB_PATH` | Path to talent_search.duckdb |
| `QDRANT_URL` | Qdrant URL or local path |
| `QDRANT_API_KEY` | Qdrant API key (cloud only) |
| `QDRANT_COLLECTION` | Hybrid collection name |
| `EMBEDDING_MODEL` | HuggingFace model name |
| `JWT_SECRET` | Secret for verifying JWT tokens |
| `DEFAULT_CLIENT_ID` | Fallback client when none resolved |
| `API_KEY_CLIENT_MAP` | JSON mapping API keys to client IDs |
