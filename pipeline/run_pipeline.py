"""
run_pipeline.py  —  Talentin Pipeline Orchestrator
====================================================

Single entry point to run any step of the Talentin data pipeline.

Usage:
    python pipeline/run_pipeline.py <command> [options]

Commands
--------
migrate
    One-time: adds client_id column to DuckDB and backfills existing rows.
    Options:
        --default-client  client_id for existing rows  (default: 00)
        --dry-run

backfill-qdrant
    One-time: stamps client_id=<default-client> onto Qdrant points that
    have no client_id payload.
    Options:
        --default-client  (default: 00)
        --batch-size      (default: 500)
        --dry-run

ingest
    Main (repeatable): imports a new client's candidate file into DuckDB + Qdrant.
    Options:
        --client          client_id  (required)
        --file            path to .jsonl / .json / .csv, or a folder
        --ext             extension when --file is a folder  (default: jsonl)
        --batch-size      (default: 100)
        --skip-qdrant
        --skip-duckdb
        --dry-run

add-client
    Adds a new entry to pipeline/clients.json.
    Options:
        --client-id       (required)
        --display-name    human-readable name
        --domains         comma-separated email domains e.g. acme.com,acme.net
        --emails          comma-separated exact email addresses

status
    Shows candidate counts per client in DuckDB.

full-setup
    Convenience: runs migrate then backfill-qdrant in sequence.
    Options:
        --default-client  (default: 00)
        --dry-run

upload-db
    SCP the local DuckDB file to the VPS after you've finished updating it.
    Options:
        --host   EC2 hostname or IP  (default: reads VPS_HOST env var)
        --user   SSH user            (default: ubuntu)
        --remote-path  destination path on server
                       (default: /var/www/talentin/database/talent_search.duckdb)
        --key    path to SSH private key file (optional)
        --restart  also restart the backend service after upload

deploy
    Convenience: ingest + upload-db + optional restart in one command.
    Options: same as ingest + upload-db combined

Examples
--------
    # First-time setup after deploying multi-tenant changes
    python pipeline/run_pipeline.py full-setup --default-client 00

    # Add a new client and then ingest their data
    python pipeline/run_pipeline.py add-client --client-id acme --display-name "Acme Corp" --domains acme.com
    python pipeline/run_pipeline.py ingest --client acme --file data/acme_candidates.jsonl

    # Upload updated DuckDB to VPS
    python pipeline/run_pipeline.py upload-db
    python pipeline/run_pipeline.py upload-db --restart   # also restarts backend

    # Full deploy: ingest + upload + restart in one shot
    python pipeline/run_pipeline.py deploy --client acme --file data/acme.jsonl --restart

    # Check what is in the DB
    python pipeline/run_pipeline.py status
"""

import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "pipeline"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


# ────────────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ────────────────────────────────────────────────────────────────────────────

def cmd_migrate(args) -> None:
    from migrate_db import run_migration
    run_migration(default_client=args.default_client, dry_run=args.dry_run)


def cmd_backfill_qdrant(args) -> None:
    from backfill_qdrant import run_backfill
    run_backfill(
        default_client=args.default_client,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


def cmd_ingest(args) -> None:
    from ingest_client import ingest, collect_files
    path = Path(args.file)
    if not path.exists():
        log.error(f"Path not found: {path}")
        sys.exit(1)
    files = collect_files(path, args.ext)
    if not files:
        log.error(f"No .{args.ext} files found in {path}")
        sys.exit(1)
    ingest(
        client_id=args.client,
        files=files,
        batch_size=args.batch_size,
        skip_qdrant=args.skip_qdrant,
        skip_duckdb=args.skip_duckdb,
        dry_run=args.dry_run,
        offset=getattr(args, 'offset', 0),
    )


def cmd_add_client(args) -> None:
    """Add a new entry to clients.json."""
    import json

    clients_path = Path(__file__).parent / "clients.json"
    if not clients_path.exists():
        data = {"clients": []}
    else:
        with open(clients_path) as fh:
            data = json.load(fh)

    existing_ids = {c["client_id"] for c in data.get("clients", [])}
    if args.client_id in existing_ids:
        log.warning(f"client_id '{args.client_id}' already exists in clients.json — updating.")
        data["clients"] = [c for c in data["clients"] if c["client_id"] != args.client_id]

    new_entry = {
        "client_id":    args.client_id,
        "display_name": args.display_name or args.client_id,
        "email_domains": [d.strip() for d in (args.domains or "").split(",") if d.strip()],
        "email_addresses": [e.strip() for e in (args.emails or "").split(",") if e.strip()],
        "active": True,
        "notes": "",
    }
    data["clients"].append(new_entry)

    with open(clients_path, "w") as fh:
        json.dump(data, fh, indent=2)

    log.info(f"✅  Added client '{args.client_id}' to {clients_path}")
    log.info("    Next steps:")
    log.info(f"      1. Update VITE_CLIENT_MAP in frontend/.env (add domain → '{args.client_id}')")
    log.info(f"      2. Run:  python pipeline/run_pipeline.py ingest --client {args.client_id} --file <data_file>")


def cmd_status(args) -> None:
    """Print candidate counts per client from DuckDB."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / "backend" / ".env")

    try:
        from config import get_config
        config = get_config()
    except Exception as e:
        log.error(f"Could not load config: {e}")
        sys.exit(1)

    import duckdb
    try:
        conn = duckdb.connect(config.duckdb_path, read_only=True)
    except Exception as e:
        log.error(f"Cannot open DuckDB at {config.duckdb_path}: {e}")
        sys.exit(1)

    print()
    print("═" * 48)
    print("  Talentin Pipeline — Database Status")
    print("═" * 48)

    # Client breakdown
    try:
        rows = conn.execute(
            "SELECT COALESCE(client_id, '<null>') AS client, COUNT(*) AS n "
            "FROM processed_profiles GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        print(f"\n{'Client':<25}  {'Candidates':>10}")
        print(f"{'─'*25}  {'─'*10}")
        total = 0
        for client, n in rows:
            print(f"{client:<25}  {n:>10,}")
            total += n
        print(f"{'─'*25}  {'─'*10}")
        print(f"{'TOTAL':<25}  {total:>10,}")
    except Exception as e:
        log.warning(f"processed_profiles query failed: {e}")

    # Tables
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"\nTables: {', '.join(t[0] for t in tables)}")
    except Exception:
        pass

    conn.close()

    # Qdrant point count
    try:
        from qdrant_client import QdrantClient
        if config.is_cloud:
            q = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=15)
        else:
            q = QdrantClient(path=config.qdrant_url)

        col = q.get_collection(config.qdrant_collection)
        print(f"\nQdrant collection   : {config.qdrant_collection}")
        print(f"  Total points      : {col.points_count:,}")  # type: ignore
    except Exception as e:
        print(f"\nQdrant status: unavailable ({e})")

    print()


# VPS connection defaults
VPS_HOST_DEFAULT = "ec2-3-12-221-73.us-east-2.compute.amazonaws.com"
VPS_USER_DEFAULT = "ubuntu"
VPS_REMOTE_PATH_DEFAULT = "/var/www/talentin/database/talent_search.duckdb"
VPS_SERVICE_NAME = "talentin-backend"


def _get_local_duckdb() -> Path:
    """Resolve local DuckDB path from config."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / "backend" / ".env")
    from config import get_config
    return Path(get_config().duckdb_path)


def cmd_upload_db(args) -> None:
    """SCP the local DuckDB file to the VPS."""
    import subprocess
    import os

    host        = args.host or os.getenv("VPS_HOST", VPS_HOST_DEFAULT)
    user        = args.user
    remote_path = args.remote_path
    key_opt     = ["-i", args.key] if args.key else []

    local_db = _get_local_duckdb()
    if not local_db.exists():
        log.error(f"Local DuckDB not found: {local_db}")
        sys.exit(1)

    size_mb = local_db.stat().st_size / (1024 * 1024)
    log.info(f"Uploading {local_db.name}  ({size_mb:.0f} MB)  →  {user}@{host}:{remote_path}")

    scp_cmd = ["scp"] + key_opt + ["-o", "StrictHostKeyChecking=no",
                                     str(local_db), f"{user}@{host}:{remote_path}"]
    result = subprocess.run(scp_cmd)
    if result.returncode != 0:
        log.error("SCP failed.")
        sys.exit(result.returncode)

    log.info("✅  DuckDB uploaded.")

    if getattr(args, "restart", False):
        log.info(f"Restarting {VPS_SERVICE_NAME} on {host}…")
        ssh_cmd = ["ssh"] + key_opt + ["-o", "StrictHostKeyChecking=no",
                   f"{user}@{host}",
                   f"sudo systemctl restart {VPS_SERVICE_NAME}"]
        result = subprocess.run(ssh_cmd)
        if result.returncode == 0:
            log.info("✅  Backend restarted.")
        else:
            log.warning("Restart command returned non-zero — check server manually.")


def cmd_deploy(args) -> None:
    """Ingest data locally then upload the updated DuckDB to VPS."""
    log.info("=== Step 1/2: Ingest ===")
    cmd_ingest(args)

    log.info("=== Step 2/2: Upload DuckDB ===")
    cmd_upload_db(args)

    log.info("=== Deploy complete ===")


def cmd_full_setup(args) -> None:
    """Run migrate then backfill-qdrant."""
    log.info("=== Step 1/2: DuckDB Migration ===")
    from migrate_db import run_migration
    run_migration(default_client=args.default_client, dry_run=args.dry_run)

    log.info("=== Step 2/2: Qdrant Backfill ===")
    from backfill_qdrant import run_backfill
    run_backfill(
        default_client=args.default_client,
        batch_size=500,
        dry_run=args.dry_run,
    )
    log.info("=== Full setup complete ===")


# ────────────────────────────────────────────────────────────────────────────
# CLI SETUP
# ────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Talentin Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # migrate
    p_migrate = sub.add_parser("migrate", help="One-time DuckDB migration")
    p_migrate.add_argument("--default-client", default="00")
    p_migrate.add_argument("--dry-run", action="store_true")
    p_migrate.set_defaults(func=cmd_migrate)

    # backfill-qdrant
    p_bq = sub.add_parser("backfill-qdrant", help="One-time Qdrant payload backfill")
    p_bq.add_argument("--default-client", default="00")
    p_bq.add_argument("--batch-size", type=int, default=500)
    p_bq.add_argument("--dry-run", action="store_true")
    p_bq.set_defaults(func=cmd_backfill_qdrant)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Import candidate data for a client")
    p_ingest.add_argument("--client", required=True)
    p_ingest.add_argument("--file", required=True)
    p_ingest.add_argument("--ext", default="jsonl")
    p_ingest.add_argument("--batch-size", type=int, default=100)
    p_ingest.add_argument("--skip-qdrant", action="store_true")
    p_ingest.add_argument("--skip-duckdb", action="store_true")
    p_ingest.add_argument("--offset", type=int, default=0, help="Skip first N records (resume after crash)")
    p_ingest.add_argument("--dry-run", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    # add-client
    p_ac = sub.add_parser("add-client", help="Register a new client in clients.json")
    p_ac.add_argument("--client-id", required=True)
    p_ac.add_argument("--display-name", default=None)
    p_ac.add_argument("--domains", default="", help="Comma-separated email domains")
    p_ac.add_argument("--emails", default="", help="Comma-separated email addresses")
    p_ac.set_defaults(func=cmd_add_client)

    # status
    p_status = sub.add_parser("status", help="Show DB/Qdrant stats per client")
    p_status.set_defaults(func=cmd_status)

    # full-setup
    p_fs = sub.add_parser("full-setup", help="Run migrate + backfill-qdrant")
    p_fs.add_argument("--default-client", default="00")
    p_fs.add_argument("--dry-run", action="store_true")
    p_fs.set_defaults(func=cmd_full_setup)

    # upload-db
    def _add_vps_args(p):
        p.add_argument("--host", default=None, help=f"VPS hostname (default: {VPS_HOST_DEFAULT})")
        p.add_argument("--user", default=VPS_USER_DEFAULT)
        p.add_argument("--remote-path", default=VPS_REMOTE_PATH_DEFAULT)
        p.add_argument("--key", default=None, help="Path to SSH private key")
        p.add_argument("--restart", action="store_true", help="Restart backend service after upload")

    p_up = sub.add_parser("upload-db", help="SCP local DuckDB to VPS")
    _add_vps_args(p_up)
    p_up.set_defaults(func=cmd_upload_db)

    # deploy  (ingest + upload-db)
    p_dep = sub.add_parser("deploy", help="Ingest client data then upload DB to VPS")
    p_dep.add_argument("--client", required=True)
    p_dep.add_argument("--file", required=True)
    p_dep.add_argument("--ext", default="jsonl")
    p_dep.add_argument("--batch-size", type=int, default=100)
    p_dep.add_argument("--skip-qdrant", action="store_true")
    p_dep.add_argument("--skip-duckdb", action="store_true")
    p_dep.add_argument("--dry-run", action="store_true")
    _add_vps_args(p_dep)
    p_dep.set_defaults(func=cmd_deploy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
