"""
convert_loxo.py  —  Convert Loxo ATS export → standard ingest JSONL
=====================================================================

Reads the Loxo candidate export (CSV without extension, or .csv)
and writes a clean .jsonl file ready for:

    python pipeline/run_pipeline.py ingest --client 01 --file data/loxo.jsonl

Usage:
    python pipeline/convert_loxo.py --input "D:/Talentin Japanese AI Search/Loxo/loxo-data" --output data/loxo.jsonl
    python pipeline/convert_loxo.py --input path/to/loxo-data --output data/loxo.jsonl --client-id 01
"""

import sys
import csv
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_loxo")

# Loxo exports can have very wide rows and huge jsondata fields
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field


def parse_skills(raw) -> list:
    """Parse comma-separated skillsets string into a clean list."""
    if not raw or str(raw).strip() in ("", "NULL", "None"):
        return []
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def safe_str(v) -> str | None:
    if v is None or str(v).strip() in ("", "NULL", "None"):
        return None
    return str(v).strip()


def convert_row(row: dict) -> dict | None:
    """Convert one CSV row into a standard ingest record."""

    # ── Parse the embedded jsondata blob ──────────────────────────────
    jd = {}
    raw_json = row.get("jsondata", "")
    if raw_json and raw_json.strip() not in ("", "NULL", "None"):
        try:
            jd = json.loads(raw_json)
        except json.JSONDecodeError:
            # Loxo sometimes double-escapes — try replacing "" with "
            try:
                jd = json.loads(raw_json.replace('""', '"'))
            except json.JSONDecodeError:
                pass  # Fall back to flat columns

    # ── person_id ─────────────────────────────────────────────────────
    person_id = (
        safe_str(jd.get("id"))
        or safe_str(row.get("atscandidateid"))
        or safe_str(row.get("candidateid"))
    )
    if not person_id:
        return None  # Can't identify this record

    # ── Name ──────────────────────────────────────────────────────────
    full_name = (
        safe_str(jd.get("name"))
        or safe_str(row.get("candidatename"))
    )
    if not full_name:
        return None  # Skip unnamed profiles

    # ── Location: prefer flat ATS columns (already cleaned) over jsondata ──
    city    = safe_str(row.get("ats_city"))    or safe_str(jd.get("city"))
    state   = safe_str(row.get("ats_state"))   or safe_str(jd.get("state"))
    country = safe_str(row.get("ats_country")) or safe_str(jd.get("country"))
    area    = safe_str(jd.get("location"))     # e.g. "New York City Metropolitan Area"

    # ── Current role ──────────────────────────────────────────────────
    current_title   = safe_str(jd.get("current_title"))
    current_company = safe_str(jd.get("current_company"))

    # ── Skills ────────────────────────────────────────────────────────
    # Prefer ats_skills (cleaned), fallback to jsondata skillsets
    skills_raw = (
        safe_str(row.get("ats_skills"))
        or safe_str(row.get("ats_primary_skills"))
        or safe_str(jd.get("skillsets"))
        or ""
    )
    skills = parse_skills(skills_raw)

    # ── Experience ────────────────────────────────────────────────────
    years_experience = None
    raw_exp = safe_str(row.get("ats_experience"))
    if raw_exp:
        try:
            years_experience = int(float(raw_exp))
        except (ValueError, TypeError):
            pass

    # ── LinkedIn + photo ──────────────────────────────────────────────
    linkedin_url = (
        safe_str(row.get("ats_linkedin_url"))
        or safe_str(jd.get("linkedin_url"))
    )
    photo = (
        safe_str(jd.get("profile_picture_original_url"))
        or safe_str(row.get("imageurl"))
    )

    # ── Emails / phones from jsondata arrays ──────────────────────────
    emails = [e.get("value") for e in (jd.get("emails") or []) if e.get("value")]
    phones = [p.get("value") for p in (jd.get("phones") or []) if p.get("value")]

    # ── Candidate jobs they were submitted for (use as work context) ──
    # We'll put these into work_history so embedding picks them up
    work_history = []
    for cj in (jd.get("candidate_jobs") or []):
        title = safe_str(cj.get("title"))
        if title:
            work_history.append({
                "title":       title,
                "company":     current_company or "",
                "start_date":  None,
                "end_date":    None,
                "description": "Applied for this role via Loxo ATS",
            })

    # Current role as first work history entry if not already captured
    if current_title or current_company:
        work_history.insert(0, {
            "title":      current_title,
            "company":    current_company,
            "start_date": None,
            "end_date":   None,
            "description": "",
        })

    return {
        "person_id":       person_id,
        "full_name":       full_name,
        "city":            city,
        "state":           state,
        "country":         country,
        "area":            area,
        "current_title":   current_title,
        "current_company": current_company,
        "skills":          skills,
        "years_experience": years_experience,
        "linkedin_url":    linkedin_url,
        "photo":           photo,
        "emails":          emails,
        "phones":          phones,
        "work_history":    work_history,
        "education":       [],
        "certifications":  [],
        "industry":        safe_str(jd.get("industry")),
        "description":     None,
    }


def convert(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = errors = 0

    with (
        open(input_path, encoding="utf-8-sig", newline="") as fh_in,
        open(output_path, "w", encoding="utf-8") as fh_out,
    ):
        reader = csv.DictReader(fh_in)
        log.info(f"Columns detected: {reader.fieldnames}")

        for i, row in enumerate(reader, 1):
            try:
                rec = convert_row(row)
                if rec is None:
                    skipped += 1
                    continue
                fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                log.warning(f"Row {i}: {e}")
                errors += 1

            if i % 500 == 0:
                log.info(f"  Processed {i:,} rows — {written:,} written, {skipped} skipped")

    log.info("=" * 50)
    log.info(f"  Total rows    : {written + skipped + errors:,}")
    log.info(f"  Written       : {written:,}")
    log.info(f"  Skipped       : {skipped}  (no id or name)")
    log.info(f"  Errors        : {errors}")
    log.info(f"  Output        : {output_path}")
    log.info("=" * 50)
    return written


def main():
    parser = argparse.ArgumentParser(description="Convert Loxo CSV export to ingest JSONL")
    parser.add_argument("--input",  required=True, help="Path to loxo-data file (CSV)")
    parser.add_argument("--output", default="data/loxo.jsonl", help="Output .jsonl path")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        log.error(f"Input not found: {inp}")
        sys.exit(1)

    log.info(f"Converting: {inp}  →  {out}")
    written = convert(inp, out)

    if written == 0:
        log.error("No records written — check input file.")
        sys.exit(1)

    log.info(f"\n✅  Done. Next steps:")
    log.info(f"   python pipeline/run_pipeline.py add-client --client-id 01 --display-name \"Loxo\" --domains loxo.co")
    log.info(f"   python pipeline/run_pipeline.py ingest --client 01 --file {out}")


if __name__ == "__main__":
    main()
