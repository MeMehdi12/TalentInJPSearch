"""
backfill_qdrant.py  —  Step 2 of the Talentin Pipeline
========================================================

Adds the `client_id` payload field to every existing point in the Qdrant
collection that is missing it.  Reads person_id → client_id from DuckDB so
the two stores stay in sync.

Run this ONCE after migrate_db.py completes.

Usage:
    python pipeline/backfill_qdrant.py

    # Override default client for unmatched points:
    python pipeline/backfill_qdrant.py --default-client acme

    # Process only a batch size at a time (useful for very large collections):
    python pipeline/backfill_qdrant.py --batch-size 500

    # Dry-run:
    python pipeline/backfill_qdrant.py --dry-run
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

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
log = logging.getLogger("backfill_qdrant")


def run_backfill(
    default_client: str = "00",
    batch_size: int = 500,
    dry_run: bool = False,
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, IsNullCondition, PayloadField

    config = get_config()

    log.info(f"Qdrant URL    : {config.qdrant_url}")
    log.info(f"Collection    : {config.qdrant_collection}")
    log.info(f"DuckDB path   : {config.duckdb_path}")
    log.info(f"Default client: '{default_client}'")
    log.info(f"Batch size    : {batch_size}")
    log.info(f"Dry-run       : {dry_run}")

    # ── 1. Build person_id → client_id map from DuckDB ───────────────────
    duck = duckdb.connect(config.duckdb_path, read_only=True)
    rows = duck.execute(
        "SELECT person_id, client_id FROM processed_profiles WHERE client_id IS NOT NULL"
    ).fetchall()
    duck.close()

    id_to_client = {str(pid): cid for pid, cid in rows}
    log.info(f"Loaded {len(id_to_client):,} person_id → client_id mappings from DuckDB")

    # ── 2. Connect to Qdrant ──────────────────────────────────────────────
    if config.is_cloud:
        qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=60)
    else:
        qdrant = QdrantClient(path=config.qdrant_url)

    collection = config.qdrant_collection
    collections = [c.name for c in qdrant.get_collections().collections]
    if collection not in collections:
        # Try the dense-only fallback
        collection = config.qdrant_collection_dense
        if collection not in collections:
            log.error(f"Neither '{config.qdrant_collection}' nor '{config.qdrant_collection_dense}' found in Qdrant.")
            sys.exit(1)
        log.warning(f"Hybrid collection not found, using dense collection: {collection}")

    total_points = qdrant.get_collection(collection).points_count
    log.info(f"Total points in '{collection}': {total_points:,}")

    # ── 3. Ensure client_id payload index exists ──────────────────────────
    # Qdrant requires a keyword index on client_id before it can be used
    # as a filter in set_payload or search queries.
    from qdrant_client.models import (
        Filter, IsNullCondition, PayloadField,
        PayloadSchemaType, TextIndexParams,
    )
    try:
        existing = qdrant.get_collection(collection).payload_schema or {}
        if "client_id" not in existing:
            log.info("Creating keyword index on 'client_id'…")
            qdrant.create_payload_index(
                collection_name=collection,
                field_name="client_id",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
            log.info("  Index created.")
        else:
            log.info("'client_id' index already exists.")
    except Exception as e:
        log.warning(f"Could not create index (may already exist): {e}")

    missing_filter = Filter(
        must=[IsNullCondition(is_null=PayloadField(key="client_id"))]
    )

    if dry_run:
        # Count how many would be updated
        count_result = qdrant.count(
            collection_name=collection,
            count_filter=missing_filter,
            exact=True,
        )
        log.info(f"[DRY-RUN] Would set client_id='{default_client}' on {count_result.count:,} points")
        return

    log.info(f"Setting client_id='{default_client}' on all points missing it (single bulk call)…")
    qdrant.set_payload(
        collection_name=collection,
        payload={"client_id": default_client},
        points=missing_filter,
        wait=True,
    )

    # Verify
    remaining = qdrant.count(
        collection_name=collection,
        count_filter=missing_filter,
        exact=True,
    ).count

    updated = total_points - remaining
    log.info(f"✅  Backfill complete — {updated:,} points updated, {remaining} still missing client_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add client_id payload to existing Qdrant points"
    )
    parser.add_argument("--default-client", default="00")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_backfill(args.default_client, args.batch_size, args.dry_run)


if __name__ == "__main__":
    main()
