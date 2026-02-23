#!/usr/bin/env python3
"""Quick diagnostic script for smart search."""
import requests, json, sys

BASE = "http://localhost:8001"

# 1. Health check
print("=" * 60)
print("1. HEALTH CHECK")
r = requests.get(f"{BASE}/api/v2/health")
print(f"   Status: {r.status_code} -> {r.json()}")

# 2. Smart search test
print("\n" + "=" * 60)
print("2. SMART SEARCH TEST: 'python developer in tokyo'")
r = requests.post(f"{BASE}/api/v2/smart-search", json={
    "query": "python developer in tokyo",
    "limit": 5,
    "location_preference": "preferred"
})

if r.status_code != 200:
    print(f"   ERROR {r.status_code}: {r.text[:500]}")
    sys.exit(1)

data = r.json()
print(f"   Total matches: {data.get('total_matches', 'N/A')}")
print(f"   Query understanding: {json.dumps(data.get('query_understanding', {}), indent=4)}")
print(f"   Took: {data.get('took_ms', '?')}ms")

for i, res in enumerate(data.get("results", [])[:5]):
    print(f"\n   [{i+1}] {res.get('full_name', '?')}")
    print(f"       Score: {res.get('score', 0):.4f}")
    print(f"       City: {res.get('city', '?')}, Country: {res.get('country', '?')}")
    print(f"       Title: {res.get('current_title', '?')}")
    print(f"       Skills: {(res.get('skills') or [])[:8]}")
    print(f"       Matched: {res.get('matched_skills', [])}")

# 3. Second test - different query
print("\n" + "=" * 60)
print("3. SMART SEARCH TEST: 'react frontend engineer'")
r = requests.post(f"{BASE}/api/v2/smart-search", json={
    "query": "react frontend engineer",
    "limit": 5,
    "location_preference": "preferred"
})
data = r.json()
print(f"   Total matches: {data.get('total_matches', 'N/A')}")
for i, res in enumerate(data.get("results", [])[:5]):
    print(f"   [{i+1}] {res.get('full_name', '?')} | Score: {res.get('score', 0):.4f} | Title: {res.get('current_title', '?')} | Skills match: {res.get('matched_skills', [])[:5]}")

print("\n" + "=" * 60)
print("DONE")
