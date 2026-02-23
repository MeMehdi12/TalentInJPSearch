#!/usr/bin/env python3
"""Check actual Qdrant payload fields for a sample point."""
import requests, json

# Use Qdrant Cloud REST API directly
# First check collection info via the backend's config
r = requests.get("http://localhost:8001/api/v2/health")
print(f"Health: {r.json()}")

# Try to get collection info from Qdrant
# Check server logs for Qdrant URL
import subprocess
out = subprocess.run(
    ["sudo", "journalctl", "-u", "talentin-backend", "--no-pager", "-n", "50", "--output=cat"],
    capture_output=True, text=True
)
for line in out.stdout.split("\n"):
    if "qdrant" in line.lower() or "collection" in line.lower() or "Available" in line.lower():
        print(f"LOG: {line.strip()[:150]}")

# Scroll 1 point from Qdrant via the app's internal connection
# We need to use the app's Qdrant client - let's query with no filters
r = requests.post("http://localhost:8001/api/v2/search", json={
    "search_text": "engineer",
    "filters": {},
    "options": {"limit": 1}
})
d = r.json()
print(f"\nSearch results count: {d.get('total_matches', 'N/A')}")
if d.get("results"):
    res = d["results"][0]
    print(f"First result city={res.get('city')}, country={res.get('country')}, location={res.get('location')}")
    print(f"Skills (first 5): {(res.get('skills') or [])[:5]}")
