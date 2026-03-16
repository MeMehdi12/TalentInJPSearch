"""Quick verification of deployed fixes."""
import json, urllib.request

URL = "https://jp.talentin.ai/api/integration/search"
KEY = "4716149828801193f46a220c8a74938eefc32aa0bd272a28ddc602104bd9eb11"

payload = {
    "search_text": "python developer",
    "filters": {
        "location": {"city": "Tokyo"},
        "skills_required": ["next"],
        "experience": {"min_years": 3}
    }
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "X-Integration-Key": KEY},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    d = json.loads(resp.read())

print(f"status={d['header']['status']}")
print(f"total={d['data']['total']}, returned={d['data']['returned']}, took={d['data']['took_ms']}ms")
for p in d["data"]["profiles"][:5]:
    loc = p.get("location_details", {})
    print(f"  {p['name']:30s}  score={p['_score']}  city={loc.get('city','?')}")
