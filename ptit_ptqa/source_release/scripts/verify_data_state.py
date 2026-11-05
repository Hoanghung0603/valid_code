import pandas as pd
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df_emb = pd.read_parquet('data/embedded_samples.parquet', columns=['id'])
df_unemb = pd.read_parquet('data/unembedded_samples.parquet', columns=['id'])
df_rem = pd.read_parquet('data/remaining_urls.parquet', columns=['id'])

n_emb = len(df_emb)
u_emb = df_emb['id'].nunique()
n_unemb = len(df_unemb)
u_unemb = df_unemb['id'].nunique()
n_rem = len(df_rem)
u_rem = df_rem['id'].nunique()

overlap_emb_unemb = set(df_emb['id']).intersection(set(df_unemb['id']))
overlap_rem_unemb = set(df_rem['id']).intersection(set(df_unemb['id']))
overlap_rem_emb = set(df_rem['id']).intersection(set(df_emb['id']))

print("=" * 65)
print("📊 BÁO CÁO TOÀN DIỆN KHO DỮ LIỆU:")
print("=" * 65)
print(f"1. embedded_samples.parquet:   {n_emb:,} dòng (Unique IDs: {u_emb:,})")
print(f"2. unembedded_samples.parquet: {n_unemb:,} dòng (Unique IDs: {u_unemb:,})")
print(f"3. remaining_urls.parquet:    {n_rem:,} URLs (Unique IDs: {u_rem:,})")
print(f"4. Trùng lặp (embedded & unembedded):    {len(overlap_emb_unemb)} mẫu")
print(f"5. Trùng lặp (remaining_urls & unembedded): {len(overlap_rem_unemb)} mẫu")
print(f"6. Trùng lặp (remaining_urls & embedded):   {len(overlap_rem_emb)} mẫu")
print(f"7. Tổng corpus (embedded + unembedded):  {n_emb + n_unemb:,} mẫu")
print("=" * 65)

assert n_emb == u_emb == 1061774, f"embedded_samples không khớp 1,061,774! Thực tế: {n_emb}"
assert n_unemb == u_unemb == 111506, f"unembedded_samples không khớp 111,506! Thực tế: {n_unemb}"
assert n_rem == u_rem == 14591, f"remaining_urls không khớp 14,591! Thực tế: {n_rem}"
assert len(overlap_emb_unemb) == 0, "Có trùng lặp giữa embedded và unembedded!"
assert len(overlap_rem_unemb) == 0, "Có trùng lặp giữa remaining và unembedded!"
assert len(overlap_rem_emb) == 0, "Có trùng lặp giữa remaining và embedded!"
print("✅ TẤT CẢ KIỂM TRA ĐỀU HOÀN TOÀN CHÍNH XÁC (100% PASS)!")
