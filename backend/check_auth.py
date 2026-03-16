import requests, json, urllib3
urllib3.disable_warnings()

BASE = "https://jp.talentin.ai"
GOOD_KEY = "4716149828801193f46a220c8a74938eefc32aa0bd272a28ddc602104bd9eb11"
PAYLOAD = {"search_text": "python developer", "limit": 3}

# Test 1: No key at all
r1 = requests.post(f"{BASE}/api/integration/search", json=PAYLOAD, headers={"Content-Type": "application/json"}, timeout=30, verify=False)
print(f"No key:        status={r1.status_code} body={r1.text[:120]}", flush=True)

# Test 2: Wrong key
r2 = requests.post(f"{BASE}/api/integration/search", json=PAYLOAD, headers={"X-Integration-Key": "wrong_key_123", "Content-Type": "application/json"}, timeout=30, verify=False)
print(f"Wrong key:     status={r2.status_code} body={r2.text[:120]}", flush=True)

# Test 3: Correct key via X-Integration-Key
r3 = requests.post(f"{BASE}/api/integration/search", json=PAYLOAD, headers={"X-Integration-Key": GOOD_KEY, "Content-Type": "application/json"}, timeout=30, verify=False)
d3 = r3.json()
print(f"Correct key:   status={r3.status_code} total={d3.get('data',{}).get('total','?')} returned={d3.get('data',{}).get('returned','?')}", flush=True)

# Test 4: Correct key via Bearer token
r4 = requests.post(f"{BASE}/api/integration/search", json=PAYLOAD, headers={"Authorization": f"Bearer {GOOD_KEY}", "Content-Type": "application/json"}, timeout=30, verify=False)
d4 = r4.json()
print(f"Bearer token:  status={r4.status_code} total={d4.get('data',{}).get('total','?')} returned={d4.get('data',{}).get('returned','?')}", flush=True)

# Test 5: Health check (no auth required)
r5 = requests.get(f"{BASE}/api/integration/health", timeout=10, verify=False)
print(f"Health (no key): status={r5.status_code} body={r5.text}", flush=True)
