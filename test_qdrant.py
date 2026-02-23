#!/usr/bin/env python3
"""Check Qdrant collection stats and test search pipeline."""
import sys, os
sys.path.insert(0, "/var/www/talentin/backend")
os.chdir("/var/www/talentin/backend")

from config import Config
c = Config.load()
print(f"Qdrant URL: {c.qdrant_url}")
print(f"Collection: {c.qdrant_collection}")
print(f"Embedding Model: {c.embedding_model}")
print(f"Embedding Dim: {c.embedding_dim}")

from qdrant_client import QdrantClient
q = QdrantClient(url=c.qdrant_url, api_key=c.qdrant_api_key)
info = q.get_collection(c.qdrant_collection)
print(f"\nQdrant Collection Stats:")
print(f"  Points: {info.points_count}")
print(f"  Vectors: {info.vectors_count}")
print(f"  Status: {info.status}")

# Check a sample point to see what fields exist
from qdrant_client.models import ScrollRequest
pts = q.scroll(c.qdrant_collection, limit=1, with_payload=True, with_vectors=False)
if pts and pts[0]:
    p = pts[0][0]
    print(f"\nSample point payload keys: {list(p.payload.keys())}")
    print(f"  full_name: {p.payload.get('full_name','?')}")
    print(f"  skills: {(p.payload.get('skills') or [])[:5]}")
    print(f"  city: {p.payload.get('city','?')}")
    print(f"  search_text (first 200 chars): {(p.payload.get('search_text','') or '')[:200]}")

# Count how many have city=tokyo
from qdrant_client.models import Filter, FieldCondition, MatchValue
tokyo_count = q.count(c.qdrant_collection, count_filter=Filter(must=[FieldCondition(key="city", match=MatchValue(value="tokyo"))]))
print(f"\nCandidates in Tokyo: {tokyo_count.count}")

# Count how many have python in skills
python_count = q.count(c.qdrant_collection, count_filter=Filter(must=[FieldCondition(key="skills", match=MatchValue(value="python"))]))
print(f"Candidates with Python skill: {python_count.count}")

# Count both
both = q.count(c.qdrant_collection, count_filter=Filter(must=[
    FieldCondition(key="city", match=MatchValue(value="tokyo")),
    FieldCondition(key="skills", match=MatchValue(value="python"))
]))
print(f"Python devs in Tokyo: {both.count}")

print("\nDONE")
