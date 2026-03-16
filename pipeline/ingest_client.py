"""
ingest_client.py  —  Step 3 (repeatable) of the Talentin Pipeline
===================================================================

Imports candidate data for a specific client into DuckDB and Qdrant.
Run this every time you receive a new dataset for any client.

INPUT FORMAT (JSONL — one JSON object per line, or a single JSON array):

  {
    "person_id":       "unique_id",          // required; string or int
    "full_name":       "Jane Smith",         // required
    "skills":          ["python","django"],   // list of strings
    "current_title":   "Senior Developer",
    "current_company": "Acme Corp",
    "city":            "San Francisco",
    "state":           "CA",
    "country":         "United States",
    "years_experience": 7,
    "headline":        "Building APIs @ Acme",
    "linkedin_url":    "https://linkedin.com/in/janesmith",
    "linkedin_slug":   "janesmith",
    "photo":           null,
    "industry":        "Technology",
    "domain":          "software_engineering",
    "profile_completeness": 85,
    "description":     "Free-text bio",
    "area":            "San Francisco Bay Area",
    "work_history": [
      { "title": "Dev", "company": "OldCorp",
        "start_date": "2019-01", "end_date": "2022-06", "description": "" }
    ],
    "education": [
      { "school": "MIT", "degree": "BS",
        "field": "Computer Science", "start_date": "2015", "end_date": "2019" }
    ],
    "certifications": [
      { "name": "AWS SAA", "authority": "Amazon", "issued_date": null, "url": null }
    ]
  }

CSV is also supported: columns map to the same field names above.
  - skills must be a JSON array string: '["python","django"]'
  - work_history / education / certifications will be skipped for CSV
    (use JSONL for full structured data)

Usage:
    python pipeline/ingest_client.py --client acme --file data/acme_candidates.jsonl
    python pipeline/ingest_client.py --client acme --file data/acme.csv
    python pipeline/ingest_client.py --client acme --file data/acme.json   # JSON array
    python pipeline/ingest_client.py --client acme --file data/ --ext jsonl  # whole folder

Options:
    --client          client_id to tag all records with (required)
    --file            path to data file or folder (required)
    --ext             file extension when --file is a folder (default: jsonl)
    --batch-size      Qdrant upsert batch size (default: 100)
    --skip-qdrant     Write to DuckDB only (useful when re-running to fix DuckDB)
    --skip-duckdb     Write to Qdrant only
    --overwrite       Replace existing records for this client (default: upsert/update)
    --dry-run         Validate input + print stats without writing anything
"""

import sys
import json
import argparse
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import duckdb
from dotenv import load_dotenv

load_dotenv(ROOT / "backend" / ".env")

from config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_client")


# ────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────────────────────────

def iter_records(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Yield one candidate dict per candidate regardless of input format."""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        import csv
        with open(file_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                # Parse JSON fields if they look like arrays
                for field in ("skills", "work_history", "education", "certifications"):
                    raw = row.get(field, "")
                    if raw and raw.strip().startswith("["):
                        try:
                            row[field] = json.loads(raw)
                        except json.JSONDecodeError:
                            row[field] = []
                    elif field in row and not isinstance(row[field], list):
                        row[field] = []
                yield dict(row)

    elif suffix in (".jsonl", ".ndjson"):
        with open(file_path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning(f"  Skipping line {line_no} in {file_path.name}: {e}")

    elif suffix == ".json":
        with open(file_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            # Could be {candidates: [...]} or a single record
            if "candidates" in data:
                yield from data["candidates"]
            else:
                yield data
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Use .jsonl, .json, or .csv")


def collect_files(path: Path, ext: str) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob(f"*.{ext.lstrip('.')}"))


# ────────────────────────────────────────────────────────────────────────────
# NORMALISATION  (lightweight — backend normalizers used for complex cases)
# ────────────────────────────────────────────────────────────────────────────

def _str(v: Any) -> Optional[str]:
    return str(v).strip() if v is not None and str(v).strip() else None


def _int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if x]
    if isinstance(v, str):
        # Try JSON first
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass
        # Comma-separated fallback
        return [x.strip() for x in v.split(",") if x.strip()]
    return []


def normalize_record(raw: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Normalise a raw candidate dict into a flat, typed record."""
    # Support both flat ("city") and nested ("location.city") schemas
    loc = raw.get("location") or {}
    if isinstance(loc, str):
        loc = {}

    rec = {
        "person_id":          _str(raw.get("person_id") or raw.get("id") or raw.get("forager_id")),
        "full_name":          _str(raw.get("full_name") or raw.get("name")),
        "city":               _str(raw.get("city") or loc.get("city")),
        "state":              _str(raw.get("state") or loc.get("state")),
        "country":            _str(raw.get("country") or loc.get("country")),
        "domain":             _str(raw.get("domain") or raw.get("primary_domain")),
        "years_experience":   _int(raw.get("years_experience") or raw.get("years_exp")),
        "profile_completeness": _int(raw.get("profile_completeness", 50)),
        "skills":             _list(raw.get("skills") or raw.get("canonical_skills") or []),
        "current_title":      _str(raw.get("current_title") or raw.get("title")),
        "current_company":    _str(raw.get("current_company") or raw.get("company")),
        # persons table extras
        "headline":           _str(raw.get("headline")),
        "linkedin_url":       _str(raw.get("linkedin_url")),
        "linkedin_slug":      _str(raw.get("linkedin_slug")),
        "photo":              _str(raw.get("photo")),
        "industry":           _str(raw.get("industry")),
        "description":        _str(raw.get("description")),
        "area":               _str(raw.get("area")),
        # structured sub-records
        "work_history":       raw.get("work_history") or [],
        "education":          raw.get("education") or [],
        "certifications":     raw.get("certifications") or [],
        "client_id":          client_id,
    }

    # Auto-generate a person_id if missing
    if not rec["person_id"]:
        slug = re.sub(r"[^a-z0-9]", "", (rec["full_name"] or "x").lower())[:12]
        rec["person_id"] = f"{client_id}_{slug}_{uuid.uuid4().hex[:8]}"

    return rec


# ────────────────────────────────────────────────────────────────────────────
# DUCKDB WRITER
# ────────────────────────────────────────────────────────────────────────────

def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure all required tables and the client_id column exist."""
    # processed_profiles ─ canonical view
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_profiles (
            person_id          VARCHAR PRIMARY KEY,
            full_name          VARCHAR,
            canonical_city     VARCHAR,
            canonical_state    VARCHAR,
            canonical_country  VARCHAR,
            primary_domain     VARCHAR,
            years_experience   INTEGER,
            profile_completeness INTEGER,
            canonical_skills   VARCHAR[],
            current_role_title   VARCHAR,
            current_role_company VARCHAR,
            client_id          VARCHAR
        )
    """)

    # persons ─ raw LinkedIn-style data
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            forager_id      VARCHAR PRIMARY KEY,
            headline        VARCHAR,
            linkedin_url    VARCHAR,
            linkedin_slug   VARCHAR,
            photo           VARCHAR,
            industry        VARCHAR,
            description     VARCHAR,
            area            VARCHAR,
            is_creator      BOOLEAN,
            is_influencer   BOOLEAN,
            search_name     VARCHAR,
            address         VARCHAR,
            linkedin_country VARCHAR,
            linkedin_area   VARCHAR,
            date_updated    TIMESTAMP,
            primary_locale  VARCHAR,
            temporary_status VARCHAR,
            temporary_emoji_status VARCHAR,
            background_picture VARCHAR
        )
    """)

    # roles
    conn.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id              VARCHAR DEFAULT gen_random_uuid(),
            forager_id      VARCHAR,
            role_title      VARCHAR,
            organization_name VARCHAR,
            start_date      VARCHAR,
            end_date        VARCHAR,
            description     VARCHAR
        )
    """)

    # educations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS educations (
            id              VARCHAR DEFAULT gen_random_uuid(),
            forager_id      VARCHAR,
            school_name     VARCHAR,
            degree          VARCHAR,
            field_of_study  VARCHAR,
            start_date      VARCHAR,
            end_date        VARCHAR
        )
    """)

    # certifications
    conn.execute("""
        CREATE TABLE IF NOT EXISTS certifications (
            id              VARCHAR DEFAULT gen_random_uuid(),
            forager_id      VARCHAR,
            certificate_name VARCHAR,
            authority       VARCHAR,
            issued_date     VARCHAR,
            certificate_url VARCHAR,
            "start"         VARCHAR,
            "end"           VARCHAR
        )
    """)

    # Ensure client_id column exists on processed_profiles (safe for old DBs)
    from migrate_db import column_exists
    if not column_exists(conn, "processed_profiles", "client_id"):
        conn.execute("ALTER TABLE processed_profiles ADD COLUMN client_id VARCHAR")
        log.info("Added client_id column to processed_profiles")


def upsert_record_to_duckdb(
    conn: duckdb.DuckDBPyConnection, rec: Dict[str, Any]
) -> None:
    pid = rec["person_id"]

    # ── processed_profiles ───────────────────────────────────────────────
    conn.execute(
        """
        INSERT OR REPLACE INTO processed_profiles
            (person_id, full_name, canonical_city, canonical_state, canonical_country,
             primary_domain, years_experience, profile_completeness, canonical_skills,
             current_role_title, current_role_company, client_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            pid,
            rec["full_name"],
            rec["city"],
            rec["state"],
            rec["country"],
            rec["domain"],
            rec["years_experience"],
            rec["profile_completeness"],
            rec["skills"],  # DuckDB handles VARCHAR[] natively
            rec["current_title"],
            rec["current_company"],
            rec["client_id"],
        ],
    )

    # ── persons ──────────────────────────────────────────────────────────
    conn.execute(
        """
        INSERT OR REPLACE INTO persons
            (forager_id, headline, linkedin_url, linkedin_slug, photo,
             industry, description, area)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            pid,
            rec["headline"],
            rec["linkedin_url"],
            rec["linkedin_slug"],
            rec["photo"],
            rec["industry"],
            rec["description"],
            rec["area"],
        ],
    )

    # ── roles ─────────────────────────────────────────────────────────────
    # Delete old roles then re-insert  (clean overwrite per person)
    conn.execute("DELETE FROM roles WHERE forager_id = ?", [pid])
    for wh in (rec["work_history"] or []):
        conn.execute(
            """
            INSERT INTO roles (forager_id, role_title, organization_name,
                               start_date, end_date, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                pid,
                _str(wh.get("title") or wh.get("role_title")),
                _str(wh.get("company") or wh.get("organization_name")),
                _str(wh.get("start_date")),
                _str(wh.get("end_date")),
                _str(wh.get("description")),
            ],
        )

    # ── educations ───────────────────────────────────────────────────────
    conn.execute("DELETE FROM educations WHERE forager_id = ?", [pid])
    for edu in (rec["education"] or []):
        conn.execute(
            """
            INSERT INTO educations (forager_id, school_name, degree,
                                    field_of_study, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                pid,
                _str(edu.get("school") or edu.get("school_name")),
                _str(edu.get("degree")),
                _str(edu.get("field") or edu.get("field_of_study")),
                _str(edu.get("start_date")),
                _str(edu.get("end_date")),
            ],
        )

    # ── certifications ───────────────────────────────────────────────────
    conn.execute("DELETE FROM certifications WHERE forager_id = ?", [pid])
    for cert in (rec["certifications"] or []):
        conn.execute(
            """
            INSERT INTO certifications (forager_id, certificate_name, authority,
                                        issued_date, certificate_url)
            VALUES (?, ?, ?, ?, ?)
            """,            [
                pid,
                _str(cert.get("name") or cert.get("certificate_name")),
                _str(cert.get("authority")),
                _str(cert.get("issued_date")),
                _str(cert.get("url") or cert.get("certificate_url")),
            ],
        )


# ────────────────────────────────────────────────────────────────────────────
# EMBEDDING + QDRANT UPSERT
# ────────────────────────────────────────────────────────────────────────────

def build_embedding_text(rec: Dict[str, Any]) -> str:
    """
    Build a rich natural-language text for embedding.
    Mirrors the format used by 20_embed_profiles.py so search quality is consistent
    with the original Talentin dataset.
    Includes: headline, identity, experience summary, full job history,
              all companies, all titles, education, ALL skills, domain.
    """
    parts = []

    # 1. Headline (often the highest-signal field)
    if rec.get("headline"):
        parts.append(rec["headline"].strip())

    # 2. Identity + location
    name = rec.get("full_name") or "Professional"
    domain = rec.get("domain") or "general"
    industry = rec.get("industry") or domain.replace("_", " ")

    loc_parts = [p for p in [rec.get("city"), rec.get("state")] if p]
    if not loc_parts and rec.get("country"):
        loc_parts = [rec["country"]]
    location = ", ".join(loc_parts) or rec.get("country") or "undisclosed"

    parts.append(f"{name} is a {industry} professional based in {location}.")

    # 3. Experience seniority sentence
    years = rec.get("years_experience") or 0
    try:
        years = float(years)
    except (ValueError, TypeError):
        years = 0
    if years > 0:
        if years >= 15:
            level = "senior executive with"
        elif years >= 10:
            level = "senior professional with"
        elif years >= 5:
            level = "experienced professional with"
        else:
            level = "professional with"
        parts.append(f"{level} {int(years)} years of experience.")

    # 4. Full work history (title + company, most recent first)
    wh_list = rec.get("work_history") or []
    if wh_list:
        job_history_parts = []
        for wh in wh_list[:10]:
            title = wh.get("title") or wh.get("role_title") or ""
            company = wh.get("company") or wh.get("organization_name") or ""
            if title and company:
                job_history_parts.append(f"{title} at {company}")
            elif title:
                job_history_parts.append(title)
        if job_history_parts:
            jh_text = " | ".join(job_history_parts)[:600]
            parts.append(f"Work experience: {jh_text}.")

    # 5. All unique companies (for "worked at X" searches)
    all_companies = []
    if rec.get("current_company"):
        all_companies.append(rec["current_company"])
    for wh in wh_list[:10]:
        c = wh.get("company") or wh.get("organization_name")
        if c and c not in all_companies:
            all_companies.append(c)
    if all_companies:
        parts.append(f"Companies worked at: {', '.join(all_companies[:400//4])}.")

    # 6. All unique titles (for role-specific searches)
    all_titles = []
    if rec.get("current_title"):
        all_titles.append(rec["current_title"])
    for wh in wh_list[:10]:
        t = wh.get("title") or wh.get("role_title")
        if t and t not in all_titles:
            all_titles.append(t)
    if all_titles:
        parts.append(f"Roles held: {', '.join(all_titles[:400//4])}.")

    # 7. Education (for school/degree searches)
    edu_list = rec.get("education") or []
    if edu_list:
        edu_parts = []
        for edu in edu_list[:5]:
            school = edu.get("school") or edu.get("school_name") or ""
            degree = edu.get("degree") or ""
            field  = edu.get("field") or edu.get("field_of_study") or ""
            entry  = " - ".join(filter(None, [school, degree, field]))
            if entry:
                edu_parts.append(entry)
        if edu_parts:
            parts.append(f"Education: {' | '.join(edu_parts)[:300]}.")

    # 8. ALL skills (no limit — model can handle it)
    skills = rec.get("skills") or []
    if skills:
        parts.append(f"Technical skills and expertise: {', '.join(str(s) for s in skills)}.")

    # 9. Domain/industry context
    if rec.get("domain"):
        parts.append(f"Primary domain: {rec['domain'].replace('_', ' ')}.")

    # 10. Free-text description / bio
    if rec.get("description"):
        parts.append(rec["description"][:300])

    return " ".join(parts)


def _load_model(config):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)} — using CUDA")
    else:
        log.info("No GPU detected — using CPU")
    log.info(f"Loading embedding model: {config.embedding_model} …")
    return SentenceTransformer(config.embedding_model, device=device)


def _load_sparse(config):
    """Load BM25 sparse encoder from pre-built cache (if available)."""
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from sparse_encoder import get_sparse_encoder
        enc = get_sparse_encoder()
        log.info("Sparse encoder loaded from cache ✓")
        return enc
    except Exception as e:
        log.warning(f"Sparse encoder not available ({e}) — dense-only mode")
        return None


def upsert_records_to_qdrant(
    records: List[Dict[str, Any]],
    model,
    sparse_enc,
    config,
    collection: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, SparseVector, NamedVector, NamedSparseVector

    if config.is_cloud:
        qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=60)
    else:
        qdrant = QdrantClient(path=config.qdrant_url)

    is_hybrid = collection == config.qdrant_collection

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        texts = [build_embedding_text(r) for r in batch]

        # Dense embeddings — large batch size benefits GPU
        import torch
        encode_batch = 256 if torch.cuda.is_available() else 64
        dense_vecs = model.encode(
            texts,
            batch_size=encode_batch,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        points = []
        for rec, dense_vec, text in zip(batch, dense_vecs, texts):
            pid = rec["person_id"]
            # Use a stable integer ID for Qdrant (hash of person_id)
            qdrant_id = abs(hash(str(pid))) % (2**63)

            # Payload
            all_companies = [rec.get("current_company")] if rec.get("current_company") else []
            for wh in (rec.get("work_history") or [])[:5]:
                if wh.get("company"):
                    all_companies.append(wh["company"])

            payload = {
                "forager_id":     pid,
                "client_id":      rec["client_id"],
                "skills":         rec.get("skills") or [],
                "current_company": rec.get("current_company"),
                "companies":      all_companies,
                "years_experience": rec.get("years_experience"),
            }

            if is_hybrid and sparse_enc is not None:
                # Hybrid point: named vectors
                sparse_idx, sparse_vals = sparse_enc.encode(text)
                points.append(
                    PointStruct(
                        id=qdrant_id,
                        vector={
                            "dense":  dense_vec,
                            "sparse": {"indices": sparse_idx, "values": sparse_vals},
                        },
                        payload=payload,
                    )
                )
            else:
                # Dense-only point
                points.append(
                    PointStruct(id=qdrant_id, vector=dense_vec, payload=payload)
                )

        if not dry_run:
            qdrant.upsert(collection_name=collection, points=points)
        log.info(
            f"  {'[DRY-RUN] ' if dry_run else ''}Qdrant upsert: "
            f"{batch_start + len(batch):>6} / {len(records)} points"
        )


# ────────────────────────────────────────────────────────────────────────────
# MAIN INGESTION ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────────────

def ingest(
    client_id: str,
    files: List[Path],
    batch_size: int = 100,
    skip_qdrant: bool = False,
    skip_duckdb: bool = False,
    dry_run: bool = False,
    offset: int = 0,
) -> None:
    config = get_config()

    log.info("=" * 62)
    log.info(f"  Talentin Ingest Pipeline  —  client: '{client_id}'")
    log.info("=" * 62)
    log.info(f"  DuckDB : {config.duckdb_path}")
    log.info(f"  Qdrant : {config.qdrant_url}")
    log.info(f"  Files  : {len(files)} file(s)")
    log.info(f"  Dry-run: {dry_run}")
    log.info("=" * 62)

    # ── Load + validate all records first ────────────────────────────────
    all_records: List[Dict[str, Any]] = []
    errors = 0

    for f in files:
        log.info(f"Loading: {f.name}")
        try:
            for raw in iter_records(f):
                rec = normalize_record(raw, client_id)
                if not rec["full_name"]:
                    log.warning(f"  Skipping record with no full_name: {raw}")
                    errors += 1
                    continue
                all_records.append(rec)
        except Exception as e:
            log.error(f"  Failed to load {f.name}: {e}")
            errors += 1

    log.info(f"Loaded {len(all_records):,} valid records ({errors} errors)")

    if offset > 0:
        log.info(f"Resuming from offset {offset:,} — skipping first {offset:,} records")
        all_records = all_records[offset:]
        log.info(f"Records remaining to process: {len(all_records):,}")

    if not all_records:
        log.error("No valid records to ingest. Exiting.")
        return

    if dry_run:
        log.info("[DRY-RUN] Skipping all writes.")
        # Show sample
        sample = all_records[0]
        log.info(f"  Sample record:")
        for k, v in sample.items():
            if k not in ("work_history", "education", "certifications"):
                log.info(f"    {k:25s}: {v}")
        return

    # ── DuckDB ────────────────────────────────────────────────────────────
    if not skip_duckdb:
        log.info(f"Writing {len(all_records):,} records to DuckDB…")
        conn = duckdb.connect(config.duckdb_path, read_only=False)
        try:
            ensure_schema(conn)
            for i, rec in enumerate(all_records, 1):
                upsert_record_to_duckdb(conn, rec)
                if i % 500 == 0 or i == len(all_records):
                    log.info(f"  DuckDB: {i:>6} / {len(all_records):,}")
            conn.commit()
        finally:
            conn.close()
        log.info("✅  DuckDB write complete.")

    # ── Qdrant ────────────────────────────────────────────────────────────
    if not skip_qdrant:
        model = _load_model(config)
        sparse_enc = _load_sparse(config)

        # Determine collection name
        from qdrant_client import QdrantClient
        if config.is_cloud:
            qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
        else:
            qdrant = QdrantClient(path=config.qdrant_url)

        collections = [c.name for c in qdrant.get_collections().collections]
        if config.qdrant_collection in collections:
            collection = config.qdrant_collection
        elif config.qdrant_collection_dense in collections:
            collection = config.qdrant_collection_dense
            log.warning(f"Hybrid collection not found — using dense: {collection}")
        else:
            log.error("No Qdrant collection found. Run your existing indexing script first.")
            return

        log.info(f"Upserting {len(all_records):,} vectors into Qdrant collection '{collection}'…")
        upsert_records_to_qdrant(
            all_records, model, sparse_enc, config, collection, batch_size, dry_run
        )
        log.info("✅  Qdrant upsert complete.")

    log.info("🎉  Ingestion finished.")
    log.info(f"     Client  : {client_id}")
    log.info(f"     Records : {len(all_records):,}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import candidate data for a client into DuckDB + Qdrant"
    )
    parser.add_argument("--client", required=True, help="client_id (e.g. acme)")
    parser.add_argument(
        "--file", required=True,
        help="Path to .jsonl / .json / .csv file, or a folder of such files"
    )
    parser.add_argument(
        "--ext", default="jsonl",
        help="File extension when --file is a folder (default: jsonl)"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Qdrant upsert batch size")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-duckdb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    path = Path(args.file)
    if not path.exists():
        log.error(f"Path not found: {path}")
        sys.exit(1)

    files = collect_files(path, args.ext)
    if not files:
        log.error(f"No {args.ext} files found in {path}")
        sys.exit(1)

    ingest(
        client_id=args.client,
        files=files,
        batch_size=args.batch_size,
        skip_qdrant=args.skip_qdrant,
        skip_duckdb=args.skip_duckdb,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
