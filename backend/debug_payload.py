import requests, json, sys

BASE = "https://jp.talentin.ai"
KEY = "4716149828801193f46a220c8a74938eefc32aa0bd272a28ddc602104bd9eb11"
HDRS = {"X-Integration-Key": KEY, "Content-Type": "application/json"}

payload = {
    "search_text": "python developer",
    "limit": 20,
    "page": 1,
    "filters": {
        "location": {"city": "Tokyo"},
        "skills_required": ["next"],
        "experience": {"min_years": 3}
    }
}

print(f"Payload: {json.dumps(payload, indent=2)}", flush=True)
r = requests.post(f"{BASE}/api/integration/search", json=payload, headers=HDRS, timeout=60, verify=False)
print(f"Status: {r.status_code}", flush=True)
d = r.json()
print(f"Total: {d.get('data',{}).get('total', '?')}", flush=True)
print(f"Returned: {d.get('data',{}).get('returned', '?')}", flush=True)
print(f"Took: {d.get('data',{}).get('took_ms', '?')}ms", flush=True)

# Now test removing filters one at a time to find which causes 0
print("\n--- Debugging: remove filters one by one ---", flush=True)

# No filters at all
p1 = {"search_text": "python developer", "limit": 5}
r1 = requests.post(f"{BASE}/api/integration/search", json=p1, headers=HDRS, timeout=60, verify=False)
d1 = r1.json()
print(f"No filters:           total={d1.get('data',{}).get('total','?')} returned={d1.get('data',{}).get('returned','?')}", flush=True)

# Only skills_required
p2 = {"search_text": "python developer", "limit": 5, "filters": {"skills_required": ["next"]}}
r2 = requests.post(f"{BASE}/api/integration/search", json=p2, headers=HDRS, timeout=60, verify=False)
d2 = r2.json()
print(f"Only skills=[next]:   total={d2.get('data',{}).get('total','?')} returned={d2.get('data',{}).get('returned','?')}", flush=True)

# Only location
p3 = {"search_text": "python developer", "limit": 5, "filters": {"location": {"city": "Tokyo"}}}
r3 = requests.post(f"{BASE}/api/integration/search", json=p3, headers=HDRS, timeout=60, verify=False)
d3 = r3.json()
print(f"Only location=Tokyo:  total={d3.get('data',{}).get('total','?')} returned={d3.get('data',{}).get('returned','?')}", flush=True)

# Only experience
p4 = {"search_text": "python developer", "limit": 5, "filters": {"experience": {"min_years": 3}}}
r4 = requests.post(f"{BASE}/api/integration/search", json=p4, headers=HDRS, timeout=60, verify=False)
d4 = r4.json()
print(f"Only exp>=3:          total={d4.get('data',{}).get('total','?')} returned={d4.get('data',{}).get('returned','?')}", flush=True)

# Skills + experience (no location)
p5 = {"search_text": "python developer", "limit": 5, "filters": {"skills_required": ["next"], "experience": {"min_years": 3}}}
r5 = requests.post(f"{BASE}/api/integration/search", json=p5, headers=HDRS, timeout=60, verify=False)
d5 = r5.json()
print(f"skills+exp (no loc):  total={d5.get('data',{}).get('total','?')} returned={d5.get('data',{}).get('returned','?')}", flush=True)

# Skills + location (no exp)
p6 = {"search_text": "python developer", "limit": 5, "filters": {"skills_required": ["next"], "location": {"city": "Tokyo"}}}
r6 = requests.post(f"{BASE}/api/integration/search", json=p6, headers=HDRS, timeout=60, verify=False)
d6 = r6.json()
print(f"skills+loc (no exp):  total={d6.get('data',{}).get('total','?')} returned={d6.get('data',{}).get('returned','?')}", flush=True)

# All three combined
p7 = {"search_text": "python developer", "limit": 5, "filters": {"skills_required": ["next"], "location": {"city": "Tokyo"}, "experience": {"min_years": 3}}}
r7 = requests.post(f"{BASE}/api/integration/search", json=p7, headers=HDRS, timeout=60, verify=False)
d7 = r7.json()
print(f"All three combined:   total={d7.get('data',{}).get('total','?')} returned={d7.get('data',{}).get('returned','?')}", flush=True)

# Try "Next.js" instead of "next"
p8 = {"search_text": "python developer", "limit": 5, "filters": {"skills_required": ["Next.js"]}}
r8 = requests.post(f"{BASE}/api/integration/search", json=p8, headers=HDRS, timeout=60, verify=False)
d8 = r8.json()
print(f"skills=[Next.js]:     total={d8.get('data',{}).get('total','?')} returned={d8.get('data',{}).get('returned','?')}", flush=True)
