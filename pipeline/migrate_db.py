"""
migrate_db.py  —  Step 1 of the Talentin Pipeline
===================================================

Adds the `client_id` column to the DuckDB `processed_profiles` table and
backfills every existing row with the DEFAULT_CLIENT_ID (default: "talentin").

Run this ONCE after pulling the multi-tenant update.

Usage:
    cd "D:\\Talentin JPSEARCH DEPLOY"
    python pipeline/migrate_db.py

    # Override default client for existing data:
    python pipeline/migrate_db.py --default-client acme

    # Dry-run (shows what would happen, no writes):
    python pipeline/migrate_db.py --dry-run
"""

import sys
import argparse
import logging
from pathlib import Path

# ── path bootstrap so we can import backend modules ───────────────────────
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
log = logging.getLogger("migrate_db")


# ── helpers ────────────────────────────────────────────────────────────────

def column_exists(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND column_name = ?",
        [table, column],
    ).fetchall()
    return len(rows) > 0


def run_migration(default_client: str, dry_run: bool = False) -> None:
    config = get_config()

    log.info(f"DuckDB path  : {config.duckdb_path}")
    log.info(f"Default client_id: '{default_client}'")
    log.info(f"Dry-run      : {dry_run}")

    conn = duckdb.connect(config.duckdb_path, read_only=dry_run)

    try:
        # ── Check current state ────────────────────────────────────────────
        exists = column_exists(conn, "processed_profiles", "client_id")
        total = conn.execute("SELECT COUNT(*) FROM processed_profiles").fetchone()[0]

        log.info(f"processed_profiles rows: {total:,}")
        log.info(f"client_id column exists: {exists}")

        if exists:
            # Already migrated — just show counts per client
            stats = conn.execute(
                "SELECT client_id, COUNT(*) FROM processed_profiles GROUP BY client_id ORDER BY 2 DESC"
            ).fetchall()
            log.info("Current client_id distribution:")
            for client, count in stats:
                log.info(f"  {client or '(NULL)':30s}  {count:>8,}")

            nulls = conn.execute(
                "SELECT COUNT(*) FROM processed_profiles WHERE client_id IS NULL"
            ).fetchone()[0]
            if nulls:
                log.warning(f"  {nulls:,} rows still have NULL client_id — backfilling…")
                if not dry_run:
                    conn.execute(
                        "UPDATE processed_profiles SET client_id = ? WHERE client_id IS NULL",
                        [default_client],
                    )
                    log.info(f"  Backfilled {nulls:,} rows with '{default_client}'")
            else:
                log.info("✅  Migration already complete, nothing to do.")
            return

        # ── Add the column ─────────────────────────────────────────────────
        log.info("Adding client_id column…")
        if not dry_run:
            conn.execute(
                "ALTER TABLE processed_profiles ADD COLUMN client_id VARCHAR DEFAULT NULL"
            )
            log.info("  Column added.")

            # Backfill existing rows
            conn.execute(
                "UPDATE processed_profiles SET client_id = ? WHERE client_id IS NULL",
                [default_client],
            )
            log.info(f"  Backfilled {total:,} rows with client_id = '{default_client}'")

            # Add NOT NULL constraint (DuckDB doesn't support ALTER COLUMN directly,
            # so we just ensure the default is always set by application code)
            log.info("  Note: enforce NOT NULL by always supplying client_id in INSERT statements.")
        else:
            log.info(f"  [DRY-RUN] Would add column and backfill {total:,} rows → '{default_client}'")

        log.info("✅  Migration complete.")

    finally:
        conn.close()


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add client_id to DuckDB processed_profiles table"
    )
    parser.add_argument(
        "--default-client",
        default="00",
        help="client_id to assign to all existing rows (default: 00)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing anything",
    )
    args = parser.parse_args()
    run_migration(args.default_client, args.dry_run)


if __name__ == "__main__":
    main()
