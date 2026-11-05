import os
import sys
import glob
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

UNIFIED_PATH = "data/corpus_urls_1m.parquet"
INDEX_BASE = "data/indices"

OUT_REMAINING = "data/remaining_urls.parquet"
OUT_EMBEDDED = "data/embedded_samples.parquet"
OUT_UNEMBEDDED = "data/unembedded_samples.parquet"

def main():
    print("=" * 70, flush=True)
    print("🚀 BẮT ĐẦU PHÂN TÁCH KHO DỮ LIỆU BTC THÀNH 3 TỆP PARQUET (STREAMING MODE)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    if not os.path.exists(UNIFIED_PATH):
        print(f"❌ Không tìm thấy {UNIFIED_PATH}! Cần chạy build_unified_urls.py trước.", flush=True)
        return

    # 1. Đọc tệp thống nhất corpus_urls_1m.parquet
    print(f"\n1. Đang nạp danh mục chuẩn từ {UNIFIED_PATH}...", flush=True)
    df_meta = pd.read_parquet(UNIFIED_PATH)
    total_urls = len(df_meta)
    print(f"   -> Đã nạp {total_urls:,} URLs.", flush=True)

    id_to_url = dict(zip(df_meta['id'], df_meta['url']))
    embedded_ids = set(df_meta[df_meta['embedded']]['id'])
    unembedded_ids = set(df_meta[df_meta['crawled'] & (~df_meta['embedded'])]['id'])
    remaining_ids = set(df_meta[~df_meta['crawled']]['id'])

    print(f"   • Embedded IDs (sẵn sàng trong Index):   {len(embedded_ids):,}", flush=True)
    print(f"   • Unembedded IDs (đã cào, chờ nhúng):    {len(unembedded_ids):,}", flush=True)
    print(f"   • Remaining IDs (chưa cào):               {len(remaining_ids):,}", flush=True)

    # =========================================================================
    # TỆP 1: data/remaining_urls.parquet (id, url)
    # =========================================================================
    print(f"\n2. [1/3] Đang xuất {OUT_REMAINING}...", flush=True)
    df_rem = df_meta[~df_meta['crawled']][['id', 'url']].copy()
    df_rem.to_parquet(OUT_REMAINING, index=False, compression="snappy")
    sz_rem = os.path.getsize(OUT_REMAINING) / (1024 * 1024)
    print(f"   ✅ Đã tạo {OUT_REMAINING}: {len(df_rem):,} URLs ({sz_rem:.2f} MB)", flush=True)
    del df_rem

    # =========================================================================
    # TỆP 2: data/embedded_samples.parquet (id, url, text) - Stream từ Index Text
    # =========================================================================
    print(f"\n3. [2/3] Đang stream xuất {OUT_EMBEDDED} từ 20 khối chỉ mục...", flush=True)
    t_emb_start = time.time()
    
    # Định nghĩa schema chuẩn cho file output
    schema_3col = pa.schema([
        ("id", pa.int64()),
        ("url", pa.string()),
        ("text", pa.string())
    ])

    chunk_marker_dirs = sorted(glob.glob(os.path.join(INDEX_BASE, "**", "chunks"), recursive=True))
    unique_shard_dirs = []
    seen_signatures = {}
    for d in chunk_marker_dirs:
        s_dir = os.path.dirname(d)
        p_files = sorted(glob.glob(os.path.join(d, "part_*.parquet")))
        if not p_files: continue
        df_sig = pd.read_parquet(p_files[0], columns=["doc_id"])
        sig = (len(p_files), len(df_sig), tuple(df_sig["doc_id"].iloc[:5].tolist()))
        if sig not in seen_signatures:
            seen_signatures[sig] = s_dir
            unique_shard_dirs.append(s_dir)

    written_emb_ids = set()
    with pq.ParquetWriter(OUT_EMBEDDED, schema_3col, compression="snappy") as writer:
        for s_idx, s in enumerate(unique_shard_dirs):
            dname = os.path.relpath(s, INDEX_BASE).replace("\\", "/")
            shard_records = []
            for p_file in glob.glob(f"{s}/text/part_*.parquet"):
                tab = pq.read_table(p_file, columns=["doc_id", "text"])
                pyd = tab.to_pydict()
                for did, txt in zip(pyd["doc_id"], pyd["text"]):
                    did_int = int(did)
                    if did_int in embedded_ids and did_int not in written_emb_ids:
                        written_emb_ids.add(did_int)
                        shard_records.append({
                            "id": did_int,
                            "url": id_to_url.get(did_int, ""),
                            "text": txt or ""
                        })
            if shard_records:
                batch_df = pd.DataFrame(shard_records)
                batch_table = pa.Table.from_pandas(batch_df, schema=schema_3col, preserve_index=False)
                writer.write_table(batch_table)
            print(f"   • [{s_idx+1}/{len(unique_shard_dirs)}] {dname:<35}: Đã ghi lũy kế {len(written_emb_ids):,} docs", flush=True)

    sz_emb = os.path.getsize(OUT_EMBEDDED) / (1024 * 1024)
    print(f"   ✅ Đã tạo {OUT_EMBEDDED}: {len(written_emb_ids):,} samples ({sz_emb:.2f} MB, {time.time() - t_emb_start:.1f}s)", flush=True)

    # =========================================================================
    # TỆP 3: data/unembedded_samples.parquet (id, url, text) - Stream từ Raw Files
    # =========================================================================
    print(f"\n4. [3/3] Đang xuất {OUT_UNEMBEDDED} từ các file raw văn bản...", flush=True)
    t_unemb_start = time.time()

    # Kiểm tra nếu file unembedded_samples.parquet đã tồn tại và đủ số lượng
    need_build_unemb = True
    written_unemb_ids = set()
    if os.path.exists(OUT_UNEMBEDDED):
        existing_rows = pq.ParquetFile(OUT_UNEMBEDDED).metadata.num_rows
        if existing_rows == len(unembedded_ids):
            print(f"   ⚡ Tệp {OUT_UNEMBEDDED} đã tồn tại và khớp chính xác {existing_rows:,} samples! Bỏ qua quét lại.", flush=True)
            need_build_unemb = False
            written_unemb_ids = unembedded_ids

    raw_sources = [
        "data/raw/final_corpus_text/tier_1_resume.parquet",
        "data/raw/final_corpus_text/tier_other.parquet",
        "data/raw/final_corpus_text/tier_1.parquet",
        "data/raw/final_corpus_text/tier_2.parquet"
    ]

    if need_build_unemb:
        with pq.ParquetWriter(OUT_UNEMBEDDED, schema_3col, compression="snappy") as writer:
            for rf in raw_sources:
                if not os.path.exists(rf): continue
                t_rf = time.time()
                pf = pq.ParquetFile(rf)
                id_col = "id" if "id" in pf.schema.names else ("doc_id" if "doc_id" in pf.schema.names else None)
                if not id_col or "text" not in pf.schema.names: continue

                rf_added = 0
                for batch in pf.iter_batches(batch_size=20000, columns=[id_col, "text"]):
                    pyd = batch.to_pydict()
                    batch_records = []
                    for cid, txt in zip(pyd[id_col], pyd["text"]):
                        cid_int = int(cid)
                        if cid_int in unembedded_ids and cid_int not in written_unemb_ids:
                            written_unemb_ids.add(cid_int)
                            batch_records.append({
                                "id": cid_int,
                                "url": id_to_url.get(cid_int, ""),
                                "text": txt or ""
                            })
                    if batch_records:
                        rf_added += len(batch_records)
                        b_df = pd.DataFrame(batch_records)
                        b_tab = pa.Table.from_pandas(b_df, schema=schema_3col, preserve_index=False)
                        writer.write_table(b_tab)

                print(f"   • {os.path.basename(rf):<32}: +{rf_added:>7,} docs (Lũy kế: {len(written_unemb_ids):,}, {time.time() - t_rf:.1f}s)", flush=True)

        # Quét local_recovered nếu còn thiếu
        if len(written_unemb_ids) < len(unembedded_ids):
            local_part_files = glob.glob("data/raw/corpus_text/local_recovered/*.parquet")
            local_added = 0
            for lpf in local_part_files:
                schema = pq.read_schema(lpf)
                id_col = "id" if "id" in schema.names else ("doc_id" if "doc_id" in schema.names else None)
                if id_col and "text" in schema.names:
                    tab = pq.read_table(lpf, columns=[id_col, "text"])
                    pyd = tab.to_pydict()
                    batch_records = []
                    for cid, txt in zip(pyd[id_col], pyd["text"]):
                        cid_int = int(cid)
                        if cid_int in unembedded_ids and cid_int not in written_unemb_ids:
                            written_unemb_ids.add(cid_int)
                            batch_records.append({
                                "id": cid_int,
                                "url": id_to_url.get(cid_int, ""),
                                "text": txt or ""
                            })
                    if batch_records:
                        local_added += len(batch_records)
                        b_df = pd.DataFrame(batch_records)
                        b_tab = pa.Table.from_pandas(b_df, schema=schema_3col, preserve_index=False)
                        writer.write_table(b_tab)
            if local_added > 0:
                print(f"   • local_recovered (591 files)     : +{local_added:>7,} docs (Lũy kế: {len(written_unemb_ids):,})", flush=True)

    sz_unemb = os.path.getsize(OUT_UNEMBEDDED) / (1024 * 1024)
    print(f"   ✅ Đã tạo {OUT_UNEMBEDDED}: {len(written_unemb_ids):,} samples ({sz_unemb:.2f} MB, {time.time() - t_unemb_start:.1f}s)", flush=True)

    # =========================================================================
    # TỔNG KẾT & ĐỐI SOÁT
    # =========================================================================
    total_sum = len(remaining_ids) + len(written_emb_ids) + len(written_unemb_ids)
    print("\n" + "=" * 70, flush=True)
    print("📊 BÁO CÁO TỔNG KẾT HOÀN TẤT PHÂN TÁCH KHO DỮ LIỆU", flush=True)
    print("=" * 70, flush=True)
    print(f"Tổng số URL chuẩn BTC 1.19M:     {total_urls:>10,}", flush=True)
    print(f"1. Remaining URLs (chưa cào):     {len(remaining_ids):>10,} ({sz_rem:.2f} MB) -> {OUT_REMAINING}", flush=True)
    print(f"2. Embedded Samples (đã nhúng):   {len(written_emb_ids):>10,} ({sz_emb:.2f} MB) -> {OUT_EMBEDDED}", flush=True)
    print(f"3. Unembedded Samples (chờ nhúng):{len(written_unemb_ids):>10,} ({sz_unemb:.2f} MB) -> {OUT_UNEMBEDDED}", flush=True)
    print("  " + "-" * 68, flush=True)
    print(f"Tổng cộng 3 file:                 {total_sum:>10,}", flush=True)
    print(f"Độ chính xác so với BTC:          {'✅ KHỚP 100% TUYỆT ĐỐI' if total_sum == total_urls else f'⚠️ Lệch {abs(total_sum - total_urls)} URLs'}", flush=True)
    print("=" * 70, flush=True)
    print(f"Toàn bộ quy trình hoàn tất trong {time.time() - t0:.2f} giây.", flush=True)

if __name__ == "__main__":
    main()
