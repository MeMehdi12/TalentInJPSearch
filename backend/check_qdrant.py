import requests, json, sys

BASE = "https://jp.talentin.ai"
KEY = "4716149828801193f46a220c8a74938eefc32aa0bd272a28ddc602104bd9eb11"
HDRS = {"X-Integration-Key": KEY, "Content-Type": "application/json"}

tests = [
    ("Basic search", {"search_text": "software engineer", "limit": 5}),
    ("Empty search_text with filters", {"search_text": "", "filters": {"skills_required": ["Python"]}, "limit": 5}),
    ("Filter-only skills", {"search_text": "", "filters": {"skills_required": ["React", "TypeScript"]}, "limit": 5}),
    ("Location must_match", {"search_text": "developer", "filters": {"location": {"country": "Japan"}}, "location_preference": "must_match", "limit": 5}),
    ("Explicit match", {"search_text": "python developer", "filters": {"skills_required": ["Python"], "job_titles": ["Developer"]}, "explicit_match": True, "limit": 5}),
]

for name, payload in tests:
    try:
        r = requests.post(f"{BASE}/api/integration/search", json=payload, headers=HDRS, timeout=60, verify=False)
        d = r.json()
        total = d.get("data", {}).get("total", "?")
        returned = d.get("data", {}).get("returned", "?")
        print(f"{name:40s} → status={r.status_code} total={total} returned={returned}", flush=True)
    except Exception as e:
        print(f"{name:40s} → ERROR: {e}", flush=True)
