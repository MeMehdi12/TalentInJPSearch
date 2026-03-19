"""Diagnostic: check client_id distribution in Qdrant"""
from qdrant_client import QdrantClient
from collections import Counter
import os
import json

client = QdrantClient("http://localhost:6333")
collection_name = "talentin_profiles_hybrid"

print(f"Checking collection: {collection_name}")
try:
    info = client.get_collection(collection_name)
    print(f"Total points: {info.points_count}")
    
    # We can't do a GROUP BY in Qdrant easily, so we'll scroll through a sample
    # and count the client_id occurrences. Scanning 10,000 points.
    client_ids = Counter()
    
    records, next_page_offset = client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=["client_id"],
        with_vectors=False
    )
    
    null_count = 0
    for record in records:
        if record.payload and "client_id" in record.payload:
            client_ids[record.payload["client_id"]] += 1
        else:
            null_count += 1
            
    print("\n=== Qdrant client_id distribution (Sample of 10,000) ===")
    for k, v in client_ids.most_common():
        print(f"  client_id={k!r} count={v}")
    print(f"  (null/missing): count={null_count}")

except Exception as e:
    print(f"Error checking Qdrant: {e}")
