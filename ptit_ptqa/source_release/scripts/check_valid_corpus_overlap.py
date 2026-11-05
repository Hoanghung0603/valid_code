import os
import glob
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

links = pd.read_parquet('data/raw/btc_provided/links_corpus_vi_en_zh.parquet')
valid_ids = set(links['id'])
print(f"Total valid BTC links: {len(valid_ids):,}")

INDEX_BASE = 'data/indices'
chunk_marker_dirs = sorted(glob.glob(os.path.join(INDEX_BASE, '**', 'chunks'), recursive=True))

unique_shard_dirs = []
seen_signatures = {}
for d in chunk_marker_dirs:
    s_dir = os.path.dirname(d)
    p_files = sorted(glob.glob(os.path.join(d, 'part_*.parquet')))
    if not p_files:
        continue
    df_sig = pd.read_parquet(p_files[0], columns=['doc_id'])
    sig = (len(p_files), len(df_sig), tuple(df_sig['doc_id'].iloc[:5].tolist()))
    if sig not in seen_signatures:
        seen_signatures[sig] = s_dir
        unique_shard_dirs.append(s_dir)

all_indexed_docs = set()
valid_indexed_docs = set()
removed_indexed_docs = set()

shard_stats = []
for s in unique_shard_dirs:
    rel_name = os.path.relpath(s, INDEX_BASE).replace('\\', '/')
    shard_docs = set()
    for p_file in glob.glob(f'{s}/chunks/part_*.parquet'):
        c_df = pd.read_parquet(p_file, columns=['doc_id'])
        shard_docs.update(c_df['doc_id'])
    
    val = shard_docs.intersection(valid_ids)
    rem = shard_docs - valid_ids
    all_indexed_docs.update(shard_docs)
    valid_indexed_docs.update(val)
    removed_indexed_docs.update(rem)
    shard_stats.append({
        'shard': rel_name,
        'total': len(shard_docs),
        'valid': len(val),
        'removed': len(rem),
        'valid_pct': len(val) / max(1, len(shard_docs)) * 100
    })

print(f"\nStats across {len(unique_shard_dirs)} unique shards:")
for st in shard_stats:
    name = st['shard']
    tot = st['total']
    val = st['valid']
    rem = st['removed']
    pct = st['valid_pct']
    print(f"  {name:<35} Total: {tot:>6} | Valid: {val:>6} ({pct:.1f}%) | Removed: {rem:>6}")

print("\n" + "=" * 50)
print(f"Total unique indexed docs: {len(all_indexed_docs):,}")
print(f"Valid docs matching new 1.19M corpus: {len(valid_indexed_docs):,} ({len(valid_indexed_docs)/len(all_indexed_docs)*100:.1f}%)")
print(f"Removed docs (dropped by BTC): {len(removed_indexed_docs):,} ({len(removed_indexed_docs)/len(all_indexed_docs)*100:.1f}%)")
print("=" * 50)
