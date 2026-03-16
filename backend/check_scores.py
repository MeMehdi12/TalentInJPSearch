import requests, json, sys, urllib3
urllib3.disable_warnings()
try:
    r = requests.post('http://127.0.0.1:8001/api/integration/search',
        json={'search_text':'Python Software Engineer','location':'United States','location_preference':'preferred','limit':8},
        headers={'X-Integration-Key':'int_jprecruit_2025_prod','Content-Type':'application/json'}, timeout=30)
    print(f"status={r.status_code}", flush=True)
    data = r.json()
    for c in data.get('candidates',[]):
        print(f"{c['score']:8.4f} | {c['name'][:30]}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True, file=sys.stderr)
