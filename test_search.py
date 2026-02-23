#!/usr/bin/env python3
"""Deeper diagnostic — check Qdrant collection and test queries."""
import requests, json

BASE = "http://localhost:8001"

# Test 1: Broad query (no location) — should return many
print("=" * 60)
print("TEST 1: 'python developer' (no location)")
r = requests.post(f"{BASE}/api/v2/smart-search", json={"query": "python developer", "limit": 5})
d = r.json()
print(f"  Total: {d.get('total_matches')}")
for i, res in enumerate(d.get("results", [])[:5]):
    print(f"  [{i+1}] {res.get('full_name','?')} | {res.get('score',0):.3f} | {res.get('city','?')} | Title: {res.get('current_title','?')} | Matched: {res.get('matched_skills',[])[:4]}")

# Test 2: Location only 
print("\n" + "=" * 60)
print("TEST 2: 'software engineer in tokyo' (location + generic role)")
r = requests.post(f"{BASE}/api/v2/smart-search", json={"query": "software engineer in tokyo", "limit": 5})
d = r.json()
print(f"  Total: {d.get('total_matches')}")
for i, res in enumerate(d.get("results", [])[:5]):
    print(f"  [{i+1}] {res.get('full_name','?')} | {res.get('score',0):.3f} | {res.get('city','?')} | Title: {res.get('current_title','?')} | Matched: {res.get('matched_skills',[])[:4]}")

# Test 3: Python developer in tokyo
print("\n" + "=" * 60)
print("TEST 3: 'python developer in tokyo' (specific)")
r = requests.post(f"{BASE}/api/v2/smart-search", json={"query": "python developer in tokyo", "limit": 5})
d = r.json()
print(f"  Total: {d.get('total_matches')}")
print(f"  Query Understanding: {json.dumps(d.get('query_understanding',{}))}")
for i, res in enumerate(d.get("results", [])[:5]):
    print(f"  [{i+1}] {res.get('full_name','?')} | {res.get('score',0):.3f} | {res.get('city','?')} | Title: {res.get('current_title','?')}")
    print(f"       Skills: {(res.get('skills') or [])[:6]}")
    print(f"       Matched: {res.get('matched_skills',[])}")

# Test 4: Java developer in bangalore
print("\n" + "=" * 60)
print("TEST 4: 'java developer in bangalore'")
r = requests.post(f"{BASE}/api/v2/smart-search", json={"query": "java developer in bangalore", "limit": 5})
d = r.json()
print(f"  Total: {d.get('total_matches')}")
for i, res in enumerate(d.get("results", [])[:5]):
    print(f"  [{i+1}] {res.get('full_name','?')} | {res.get('score',0):.3f} | {res.get('city','?')} | Title: {res.get('current_title','?')} | Matched: {res.get('matched_skills',[])[:4]}")

# Test 5: Check server logs for enriched search text
print("\n" + "=" * 60)
print("TEST 5: Check backend logs")
import subprocess
out = subprocess.run(["sudo", "journalctl", "-u", "talentin-backend", "--no-pager", "-n", "30", "--output=cat"], capture_output=True, text=True)
for line in out.stdout.split("\n"):
    if "Enriched search text" in line or "Search text for embedding" in line or "Qdrant filter" in line or "Initial Qdrant results" in line:
        print(f"  LOG: {line.strip()}")

print("\nDONE")
