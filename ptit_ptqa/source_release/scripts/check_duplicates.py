import os, glob
import pandas as pd

shards = sorted(glob.glob('data/indices/**/chunks', recursive=True))
shard_signatures = {}
for s in shards:
    rel = os.path.relpath(os.path.dirname(s), 'data/indices').replace('\\', '/')
    parts = sorted(glob.glob(os.path.join(s, 'part_*.parquet')))
    if parts:
        df = pd.read_parquet(parts[0], columns=['doc_id'])
        sig = (len(parts), len(df), tuple(df['doc_id'].iloc[:5].tolist()))
        shard_signatures[rel] = sig

seen = {}
duplicates = []
for rel, sig in shard_signatures.items():
    if sig in seen:
        print(f"DUPLICATE DETECTED: '{rel}' == '{seen[sig]}'")
        duplicates.append((rel, seen[sig]))
    else:
        seen[sig] = rel

print(f"\nTotal shards: {len(shards)}")
print(f"Unique shards: {len(seen)}")
print(f"Duplicates: {len(duplicates)}")
