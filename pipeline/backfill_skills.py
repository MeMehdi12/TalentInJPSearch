"""
backfill_skills.py  -  Extract skills from titles/headlines for empty-skill profiles
=====================================================================================

Root-cause: 144,114 profiles (57%) in DuckDB have empty canonical_skills because
the original LinkedIn scrape did not include a skills section for these people.
However, 83K+ of them DO have job titles and headlines. This script extracts
basic skills from those text fields using a keyword mapping approach (no LLM needed).

Usage:
    cd "D:\\Talentin JPSEARCH DEPLOY"
    python pipeline/backfill_skills.py --dry-run       # preview changes
    python pipeline/backfill_skills.py                  # apply to DuckDB
    python pipeline/backfill_skills.py --limit 1000     # test on first 1000
    python pipeline/backfill_skills.py --db Database/talent_search.duckdb  # explicit path
"""

import sys
import re
import argparse
import logging
from pathlib import Path
from typing import List, Set, Dict

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
log = logging.getLogger("backfill_skills")

# ────────────────────────────────────────────────────────────────────────────
# TITLE → SKILL MAPPING
# Maps common job title keywords to skills that are reasonable to infer.
# Conservative: only adds skills that are very likely given the title.
# ────────────────────────────────────────────────────────────────────────────

TITLE_SKILL_MAP: Dict[str, List[str]] = {
    # Engineering / Software
    "software engineer": ["Software Engineering", "Software Development"],
    "software developer": ["Software Development", "Software Engineering"],
    "full stack": ["Full Stack Development"],
    "fullstack": ["Full Stack Development"],
    "frontend": ["Frontend Development"],
    "front-end": ["Frontend Development"],
    "front end": ["Frontend Development"],
    "backend": ["Backend Development"],
    "back-end": ["Backend Development"],
    "back end": ["Backend Development"],
    "devops": ["DevOps", "CI/CD"],
    "sre": ["Site Reliability Engineering", "DevOps"],
    "data engineer": ["Data Engineering", "SQL", "ETL"],
    "data scientist": ["Data Science", "Machine Learning", "Python", "Statistics"],
    "data analyst": ["Data Analysis", "SQL", "Excel"],
    "machine learning": ["Machine Learning", "Python"],
    "ml engineer": ["Machine Learning", "Python"],
    "ai engineer": ["Artificial Intelligence", "Machine Learning"],
    "cloud engineer": ["Cloud Computing"],
    "cloud architect": ["Cloud Architecture", "Cloud Computing"],
    "security engineer": ["Information Security", "Cybersecurity"],
    "qa engineer": ["Quality Assurance", "Testing"],
    "test engineer": ["Quality Assurance", "Testing"],
    "mobile developer": ["Mobile Development"],
    "ios developer": ["iOS Development", "Swift"],
    "android developer": ["Android Development", "Kotlin"],
    "embedded": ["Embedded Systems"],
    "firmware": ["Firmware Development", "Embedded Systems"],

    # Design
    "ux designer": ["UX Design", "User Experience", "User Research"],
    "ui designer": ["UI Design", "User Interface Design"],
    "ux/ui": ["UX Design", "UI Design", "User Experience"],
    "ui/ux": ["UX Design", "UI Design", "User Experience"],
    "product designer": ["Product Design", "UX Design"],
    "graphic designer": ["Graphic Design", "Visual Design"],
    "visual designer": ["Visual Design", "Graphic Design"],
    "interaction designer": ["Interaction Design", "UX Design"],
    "web designer": ["Web Design"],
    "creative director": ["Creative Direction", "Design Leadership"],
    "art director": ["Art Direction", "Visual Design"],

    # Product / Management
    "product manager": ["Product Management", "Product Strategy"],
    "program manager": ["Program Management", "Project Management"],
    "project manager": ["Project Management"],
    "scrum master": ["Scrum", "Agile", "Project Management"],
    "engineering manager": ["Engineering Management", "Technical Leadership"],
    "technical lead": ["Technical Leadership"],
    "tech lead": ["Technical Leadership"],
    "cto": ["Technical Leadership", "Engineering Management", "Strategy"],
    "vp of engineering": ["Engineering Management", "Technical Leadership"],
    "director of engineering": ["Engineering Management", "Technical Leadership"],
    "director of product": ["Product Management", "Product Strategy"],

    # Marketing / Sales
    "marketing manager": ["Marketing", "Marketing Strategy"],
    "digital marketing": ["Digital Marketing"],
    "content marketing": ["Content Marketing", "Content Strategy"],
    "seo": ["SEO", "Search Engine Optimization"],
    "growth": ["Growth Marketing"],
    "brand manager": ["Brand Management", "Marketing"],
    "account executive": ["Sales", "Account Management", "Business Development"],
    "sales director": ["Sales", "Sales Management", "Business Development"],
    "account manager": ["Account Management", "Client Relations"],
    "business development": ["Business Development", "Sales"],

    # Finance / Accounting
    "accountant": ["Accounting", "Financial Reporting"],
    "financial analyst": ["Financial Analysis", "Finance"],
    "controller": ["Accounting", "Financial Reporting", "Compliance"],
    "cfo": ["Finance", "Financial Strategy", "Accounting"],
    "auditor": ["Auditing", "Compliance"],
    "tax": ["Tax", "Accounting"],

    # Healthcare
    "registered nurse": ["Nursing", "Patient Care", "Healthcare"],
    "nurse practitioner": ["Nursing", "Patient Care", "Healthcare"],
    "physician": ["Medicine", "Patient Care", "Healthcare"],
    "doctor": ["Medicine", "Patient Care", "Healthcare"],
    "pharmacist": ["Pharmacy", "Healthcare"],
    "therapist": ["Therapy", "Healthcare"],
    "counselor": ["Counseling", "Mental Health"],
    "psychologist": ["Psychology", "Mental Health"],

    # Education
    "teacher": ["Teaching", "Education"],
    "professor": ["Teaching", "Research", "Education"],
    "instructor": ["Teaching", "Education"],
    "tutor": ["Tutoring", "Education"],
    "curriculum": ["Curriculum Development", "Education"],

    # Operations / HR
    "operations manager": ["Operations Management"],
    "hr manager": ["Human Resources", "HR Management"],
    "recruiter": ["Recruiting", "Talent Acquisition"],
    "talent acquisition": ["Talent Acquisition", "Recruiting"],

    # Legal
    "attorney": ["Law", "Legal"],
    "lawyer": ["Law", "Legal"],
    "paralegal": ["Legal", "Legal Research"],
    "compliance": ["Compliance", "Regulatory"],

    # Consulting
    "consultant": ["Consulting"],
    "management consultant": ["Management Consulting", "Strategy"],
    "strategy": ["Strategy", "Strategic Planning"],

    # Specific technologies (when mentioned in title)
    "python": ["Python"],
    "java developer": ["Java"],
    "javascript": ["JavaScript"],
    "react": ["React"],
    "angular": ["Angular"],
    "vue": ["Vue.js"],
    "node": ["Node.js"],
    "ruby": ["Ruby"],
    "golang": ["Go"],
    ".net": [".NET", "C#"],
    "c++": ["C++"],
    "rust": ["Rust"],
    "php": ["PHP"],
    "salesforce": ["Salesforce"],
    "sap": ["SAP"],
    "oracle": ["Oracle"],
    "aws": ["AWS", "Cloud Computing"],
    "azure": ["Azure", "Cloud Computing"],
    "gcp": ["Google Cloud Platform", "Cloud Computing"],
    "kubernetes": ["Kubernetes", "Container Orchestration"],
    "docker": ["Docker", "Containerization"],
    "terraform": ["Terraform", "Infrastructure as Code"],
    "blockchain": ["Blockchain"],
    "smart contract": ["Smart Contracts", "Blockchain"],
}

# Seniority keywords → "Leadership" or "Management" skills
SENIORITY_KEYWORDS = {
    "senior", "sr.", "sr", "lead", "principal", "staff",
    "director", "vp", "vice president", "head of", "chief",
    "manager", "managing"
}


def extract_skills_from_text(title: str, headline: str) -> List[str]:
    """Extract skills from a job title and headline using keyword matching."""
    skills: Set[str] = set()
    text = f"{title} {headline}".lower().strip()

    if not text or len(text) < 3:
        return []

    # Match against title→skill mapping
    for keyword, mapped_skills in TITLE_SKILL_MAP.items():
        if keyword in text:
            skills.update(mapped_skills)

    # Add leadership skill for senior roles
    text_words = set(re.split(r"[\s,/|·\-]+", text))
    if text_words & SENIORITY_KEYWORDS and len(skills) > 0:
        skills.add("Leadership")

    return sorted(skills)


def run_backfill(dry_run: bool = False, limit: int = 0, db_path: str = None) -> None:
    if db_path:
        duckdb_path = db_path
    else:
        config = get_config()
        duckdb_path = config.duckdb_path
    conn = duckdb.connect(duckdb_path, read_only=dry_run)

    # Count eligible profiles
    count = conn.execute("""
        SELECT COUNT(*)
        FROM processed_profiles pp
        LEFT JOIN persons p ON pp.person_id = p.forager_id
        WHERE (pp.canonical_skills IS NULL OR len(pp.canonical_skills) = 0)
          AND (
            (pp.current_role_title IS NOT NULL AND len(pp.current_role_title) > 2)
            OR (p.headline IS NOT NULL AND len(p.headline) > 5)
          )
    """).fetchone()[0]
    log.info(f"Eligible profiles (empty skills + has title/headline): {count:,}")

    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    rows = conn.execute(f"""
        SELECT pp.person_id,
               pp.current_role_title,
               p.headline
        FROM processed_profiles pp
        LEFT JOIN persons p ON pp.person_id = p.forager_id
        WHERE (pp.canonical_skills IS NULL OR len(pp.canonical_skills) = 0)
          AND (
            (pp.current_role_title IS NOT NULL AND len(pp.current_role_title) > 2)
            OR (p.headline IS NOT NULL AND len(p.headline) > 5)
          )
        {limit_clause}
    """).fetchall()

    updated = 0
    skipped = 0

    for pid, title, headline in rows:
        skills = extract_skills_from_text(title or "", headline or "")
        if not skills:
            skipped += 1
            continue

        if dry_run:
            if updated < 10:
                log.info(f"  [DRY-RUN] {pid} | {title} → {skills}")
            updated += 1
            continue

        conn.execute(
            "UPDATE processed_profiles SET canonical_skills = ? WHERE person_id = ?",
            [skills, pid]
        )
        updated += 1

        if updated % 5000 == 0:
            log.info(f"  Updated {updated:,} / {len(rows):,}")

    if not dry_run and updated > 0:
        conn.commit()

    log.info(f"{'[DRY-RUN] ' if dry_run else ''}Done: {updated:,} profiles updated, {skipped:,} skipped (no matching keywords)")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill skills from titles/headlines")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows to process (0=all)")
    parser.add_argument("--db", default=None, help="Explicit path to DuckDB file")
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run, limit=args.limit, db_path=args.db)


if __name__ == "__main__":
    main()
