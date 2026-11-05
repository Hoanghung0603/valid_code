import os
import sys
import glob
import time
import json
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BTC_LINKS_PATH = "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"
OLD_INDEX_BASE = "data/indices/old_dataset"
NEW_INDEX_DIR = "data/indices/new_dataset/index"
VOCAB_SIZE = 250005
CHUNK_PART_SIZE = 100000
TEXT_PART_SIZE = 50000

def main():
    print("=" * 70, flush=True)
    print("🚀 BẮT ĐẦU GỘP CHỈ MỤC (VECTORS + ARTIFACTS) CHUẨN BTC VÀO THƯ MỤC DUY NHẤT", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    if not os.path.exists(BTC_LINKS_PATH):
        print(f"❌ Không tìm thấy tệp {BTC_LINKS_PATH}!", flush=True)
        return

    # 1. Đọc danh sách ID chuẩn BTC (1.22M URLs)
    print(f"\n1. Đang tải danh sách ID tài liệu hợp lệ từ {BTC_LINKS_PATH}...", flush=True)
    df_btc = pd.read_parquet(BTC_LINKS_PATH, columns=["id"])
    valid_doc_ids = set(df_btc["id"])
    print(f"   -> Đã tải {len(valid_doc_ids):,} mã ID hợp lệ từ BTC.", flush=True)

    # 2. Quét các khối shard duy nhất trong old_dataset
    print(f"\n2. Đang quét các khối chỉ mục tại {OLD_INDEX_BASE}...", flush=True)
    chunk_marker_dirs = sorted(glob.glob(os.path.join(OLD_INDEX_BASE, "**", "chunks"), recursive=True))
    unique_shard_dirs = []
    seen_signatures = {}
    for d in chunk_marker_dirs:
        s_dir = os.path.dirname(d)
        p_files = sorted(glob.glob(os.path.join(d, "part_*.parquet")))
        if not p_files:
            continue
        df_sig = pd.read_parquet(p_files[0], columns=["doc_id"])
        sig = (len(p_files), len(df_sig), tuple(df_sig["doc_id"].iloc[:5].tolist()))
        if sig not in seen_signatures:
            seen_signatures[sig] = s_dir
            unique_shard_dirs.append(s_dir)

    print(f"   -> Tìm thấy {len(unique_shard_dirs)} khối shard DUY NHẤT.", flush=True)

    # 3. Tạo cấu trúc thư mục mới
    chunks_out_dir = os.path.join(NEW_INDEX_DIR, "chunks")
    text_out_dir = os.path.join(NEW_INDEX_DIR, "text")
    emb_out_dir = os.path.join(NEW_INDEX_DIR, "emb_dense")
    os.makedirs(chunks_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)
    os.makedirs(emb_out_dir, exist_ok=True)

    # Buffer lưu trữ trung gian cho Vectors & Chunks
    curr_chunk_id = 0
    buffer_chunk_dfs = []
    buffer_ids = []
    buffer_vecs = []
    buffer_sp_submats = []
    buffered_chunk_count = 0
    emb_part_idx = 0

    # Buffer lưu trữ cho Text
    seen_doc_ids = set()
    buffer_text_records = []
    text_part_idx = 0

    def flush_emb_batch():
        nonlocal emb_part_idx, buffer_chunk_dfs, buffer_ids, buffer_vecs, buffer_sp_submats, buffered_chunk_count
        if buffered_chunk_count == 0:
            return

        part_tag = f"{emb_part_idx:04d}"
        n_items = buffered_chunk_count

        # 1. Ghi chunks parquet
        c_df = pd.concat(buffer_chunk_dfs, ignore_index=True)
        c_path = os.path.join(chunks_out_dir, f"part_{part_tag}.parquet")
        c_df.to_parquet(c_path, index=False, compression="snappy")

        # 2. Ghi ids npy
        ids_arr = np.concatenate(buffer_ids)
        ids_path = os.path.join(emb_out_dir, f"ids_{part_tag}.npy")
        np.save(ids_path, ids_arr)

        # 3. Ghi vec npy (float16 để tiết kiệm dung lượng đĩa và RAM)
        vecs_arr = np.vstack(buffer_vecs)
        vec_path = os.path.join(emb_out_dir, f"vec_{part_tag}.npy")
        np.save(vec_path, vecs_arr)

        # 4. Ghi sparse npz (chuẩn CSR của dự án)
        sp_mat = sp.vstack(buffer_sp_submats, format="csr")
        sp_path = os.path.join(emb_out_dir, f"sp_{part_tag}.npz")
        np.savez_compressed(
            sp_path,
            ids=sp_mat.indices.astype(np.int32),
            vals=sp_mat.data.astype(np.float16),
            ptr=sp_mat.indptr.astype(np.int64)
        )

        sz_mb = (os.path.getsize(c_path) + os.path.getsize(ids_path) + os.path.getsize(vec_path) + os.path.getsize(sp_path)) / (1024 * 1024)
        print(f"      💾 [Đã lưu Khối Vector #{part_tag}] {n_items:,} chunks ({sz_mb:.1f} MB)", flush=True)

        buffer_chunk_dfs = []
        buffer_ids = []
        buffer_vecs = []
        buffer_sp_submats = []
        buffered_chunk_count = 0
        emb_part_idx += 1
        gc.collect()

    def flush_text_batch():
        nonlocal text_part_idx, buffer_text_records
        if not buffer_text_records:
            return

        part_tag = f"{text_part_idx:04d}"
        t_df = pd.DataFrame(buffer_text_records)
        t_path = os.path.join(text_out_dir, f"part_{part_tag}.parquet")
        t_df.to_parquet(t_path, index=False, compression="snappy")
        sz_mb = os.path.getsize(t_path) / (1024 * 1024)
        print(f"      📄 [Đã lưu Khối Văn Bản #{part_tag}] {len(t_df):,} bài viết ({sz_mb:.1f} MB)", flush=True)

        buffer_text_records = []
        text_part_idx += 1
        gc.collect()

    print(f"\n3. Bắt đầu trích xuất và gộp vector từ 20 shards...", flush=True)
    total_valid_chunks_collected = 0

    for s_idx, s_dir in enumerate(unique_shard_dirs, 1):
        rel_s = os.path.relpath(s_dir, OLD_INDEX_BASE).replace("\\", "/")
        t_s = time.time()

        # 3.1. Đọc và lọc chunks mapping của shard này
        chunks_files = sorted(glob.glob(f"{s_dir}/chunks/part_*.parquet"))
        shard_valid_chunk_map = {}  # local_chunk_id -> (doc_id, start, end)
        for cf in chunks_files:
            c_df = pd.read_parquet(cf, columns=["chunk_id", "doc_id", "start", "end"])
            v_df = c_df[c_df["doc_id"].isin(valid_doc_ids)]
            for r in v_df.itertuples(index=False):
                shard_valid_chunk_map[r.chunk_id] = (r.doc_id, r.start, r.end)

        if not shard_valid_chunk_map:
            print(f"   [{s_idx:02d}/20] {rel_s:<38}: 0 chunks hợp lệ (Bỏ qua)", flush=True)
            continue

        # 3.2. Đọc và lọc Vectors & Sparse (Vectorized Block Extraction)
        id_files = sorted(glob.glob(f"{s_dir}/emb_dense/ids_*.npy"))
        shard_chunks_extracted = 0

        for idf in id_files:
            ids_arr = np.load(idf)
            valid_mask = np.isin(ids_arr, list(shard_valid_chunk_map.keys()))
            n_valid = int(np.sum(valid_mask))
            if n_valid == 0:
                continue

            # Dense vectors
            vec_file = idf.replace("ids_", "vec_")
            vec_arr = np.load(vec_file)
            sub_vecs = vec_arr[valid_mask].astype(np.float16)

            # Sparse matrix
            sp_file = idf.replace("ids_", "sp_").replace(".npy", ".npz")
            z = np.load(sp_file)
            M = sp.csr_matrix(
                (z['vals'].astype(np.float32), z['ids'], z['ptr']),
                shape=(len(ids_arr), VOCAB_SIZE)
            )
            sub_sp = M[valid_mask]

            # Vectorized assignment of global chunk IDs
            sub_local_ids = ids_arr[valid_mask]
            new_chunk_ids = np.arange(curr_chunk_id, curr_chunk_id + n_valid, dtype=np.int64)
            curr_chunk_id += n_valid

            # Extract chunk metadata efficiently
            docs = [shard_valid_chunk_map[cid][0] for cid in sub_local_ids]
            starts = [shard_valid_chunk_map[cid][1] for cid in sub_local_ids]
            ends = [shard_valid_chunk_map[cid][2] for cid in sub_local_ids]

            batch_chunk_df = pd.DataFrame({
                "chunk_id": new_chunk_ids,
                "doc_id": docs,
                "start": starts,
                "end": ends
            })

            buffer_chunk_dfs.append(batch_chunk_df)
            buffer_ids.append(new_chunk_ids)
            buffer_vecs.append(sub_vecs)
            buffer_sp_submats.append(sub_sp)
            buffered_chunk_count += n_valid
            shard_chunks_extracted += n_valid

            if buffered_chunk_count >= CHUNK_PART_SIZE:
                flush_emb_batch()

        total_valid_chunks_collected += shard_chunks_extracted

        # 3.3. Đọc và lọc Text
        text_files = sorted(glob.glob(f"{s_dir}/text/part_*.parquet"))
        shard_texts_extracted = 0
        for tf in text_files:
            t_df = pd.read_parquet(tf, columns=["doc_id", "text"])
            v_tdf = t_df[t_df["doc_id"].isin(valid_doc_ids)]
            for r in v_tdf.itertuples(index=False):
                did = int(r.doc_id)
                if did not in seen_doc_ids:
                    seen_doc_ids.add(did)
                    buffer_text_records.append({
                        "doc_id": did,
                        "text": r.text
                    })
                    shard_texts_extracted += 1
                    if len(buffer_text_records) >= TEXT_PART_SIZE:
                        flush_text_batch()

        print(f"   [{s_idx:02d}/20] {rel_s:<38}: +{shard_chunks_extracted:>7,} chunks | +{shard_texts_extracted:>6,} docs ({time.time() - t_s:.1f}s)", flush=True)

    # Flush toàn bộ phần còn lại
    flush_emb_batch()
    flush_text_batch()

    # 4. Ghi tệp stats.json tổng kết cho index mới
    stats_data = {
        "index_name": "ViBioMIR_Consolidated_Index",
        "total_docs": len(seen_doc_ids),
        "total_chunks": curr_chunk_id,
        "dense_dim": 1024,
        "vocab_size": VOCAB_SIZE,
        "model": "BAAI/bge-m3",
        "emb_parts": emb_part_idx,
        "text_parts": text_part_idx,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Consolidated from 20 shards in old_dataset for BTC 1.22M URLs"
    }
    stats_file = os.path.join(NEW_INDEX_DIR, "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    # 5. Báo cáo tổng kết
    total_time = time.time() - t0
    print("\n" + "=" * 70, flush=True)
    print("🎉 HOÀN THÀNH GỘP CHỈ MỤC MỚI VÀO THƯ MỤC DUY NHẤT!", flush=True)
    print("=" * 70, flush=True)
    print(f"Thư mục chỉ mục mới: {NEW_INDEX_DIR}")
    print(f"Tổng tài liệu đã gộp: {len(seen_doc_ids):,} docs (Khớp danh mục embedded: {'✅ 100%' if len(seen_doc_ids) == 462882 else f'{len(seen_doc_ids)}'})")
    print(f"Tổng chunks hợp lệ:   {curr_chunk_id:,} chunks (Loại bỏ sạch 1.85M chunks rác!)")
    print(f"Số khối Vector:       {emb_part_idx} parts (100k chunks/part)")
    print(f"Số khối Văn Bản:      {text_part_idx} parts (50k docs/part)")
    print(f"Thời gian thực hiện:  {total_time:.2f} giây")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
