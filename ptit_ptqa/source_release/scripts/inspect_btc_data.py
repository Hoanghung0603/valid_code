import sys
import os
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=== INSPECTING query.parquet ===")
q_new = pd.read_parquet('data/raw/btc_provided/query.parquet')
print('New query shape:', q_new.shape)
print('New query columns:', q_new.columns.tolist())

if os.path.exists('data/raw/btc_provided/query_old_1200.parquet'):
    q_old = pd.read_parquet('data/raw/btc_provided/query_old_1200.parquet')
    print('Old query shape:', q_old.shape)
    is_same = q_new.equals(q_old)
    print('Is new query exactly identical to old query?:', is_same)
    id_col = 'id' if 'id' in q_new.columns else 'query_id'
    old_ids = set(q_old[id_col])
    new_ids = set(q_new[id_col])
    print(f'Old ID count: {len(old_ids)}, New ID count: {len(new_ids)}')
    print(f'Added IDs: {len(new_ids - old_ids)}, Removed IDs: {len(old_ids - new_ids)}')
    if not is_same:
        diff_count = 0
        for i in range(min(len(q_new), len(q_old))):
            if q_new.iloc[i]['query'] != q_old.iloc[i]['query'] or q_new.iloc[i][id_col] != q_old.iloc[i][id_col]:
                diff_count += 1
                if diff_count <= 5:
                    print(f'Diff #{diff_count}:')
                    print(f'  Old ID {q_old.iloc[i][id_col]}: {q_old.iloc[i]["query"][:80]}')
                    print(f'  New ID {q_new.iloc[i][id_col]}: {q_new.iloc[i]["query"][:80]}')
        print('Total differences:', diff_count)

print("\n=== INSPECTING links_corpus_vi_en_zh.parquet ===")
links_path = 'data/raw/btc_provided/links_corpus_vi_en_zh.parquet'
if os.path.exists(links_path):
    links = pd.read_parquet(links_path)
    print('Links shape:', links.shape)
    print('Links columns:', links.columns.tolist())
    print('Sample 2 rows:')
    for idx, row in links.head(2).iterrows():
        print(dict(row))
    for col in links.columns:
        if col not in ['url', 'title', 'id', 'text']:
            print(f'\n{col} value counts:')
            print(links[col].value_counts().head(10))
