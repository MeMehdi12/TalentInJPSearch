"""Diagnostic: check client_id distribution in DuckDB"""
import duckdb, os

db_path = "/var/www/talentin/database/talent_search.duckdb"
conn = duckdb.connect(db_path, read_only=True)

print("=== DuckDB client_id distribution (processed_profiles) ===")
rows = conn.execute("SELECT client_id, COUNT(*) as cnt FROM processed_profiles GROUP BY client_id ORDER BY cnt DESC LIMIT 20").fetchall()
for r in rows:
    print(f"  client_id={r[0]!r}  count={r[1]}")

total = conn.execute("SELECT COUNT(*) FROM processed_profiles").fetchone()[0]
print(f"\nTotal profiles: {total}")
conn.close()
