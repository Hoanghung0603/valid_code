"""Tính cosine bge-m3 cho các cặp (câu, doc) cụ thể từ embedding doc ĐÃ CÓ (blk_*.npz).

Chạy trên z87 GPU. --pairs JSON {qid: [[doc_id, idx], ...]}; nhúng truy vấn bằng bge-m3
(--queries / --query-len, thêm --queries2 / --query-len2 cho truy vấn HyDE) rồi cosine cho
từng cặp. Tái dùng dense_topk.load_embeddings + dense_topk.embed_queries (cùng cấu hình embed).
Điểm round 4 chữ số như embed_run/dense_topk. Doc thiếu embedding bị bỏ qua (đếm trong report).

Dùng (z87):
  python scripts/dense_pairs.py --emb D:\\r2ai\\work\\emb D:\\r2ai\\work\\emb_all \
      --pairs cands_new_all.json --queries queries.jsonl --query-len 256 \
      --queries2 D:\\r2ai\\work\\hyde\\q_hyde.jsonl --query-len2 512 \
      --out cache_new_dense.json --out2 cache_new_hyde.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from dense_topk import embed_queries, load_embeddings


def doc_row_map(ids: np.ndarray) -> dict:
    """{doc_id: hàng trong ma trận emb}."""
    return {int(d): r for r, d in enumerate(ids)}


def pair_cosines(embs: np.ndarray, qe_row: np.ndarray, rows: list) -> np.ndarray:
    """Cosine (emb đã chuẩn hóa) của một truy vấn với các doc ở hàng rows; float32."""
    if not rows:
        return np.zeros(0, dtype=np.float32)
    e = embs[np.asarray(rows, dtype=np.int64)]
    return (e.astype(np.float32) @ qe_row.astype(np.float32))


def build_hits(pairs_q: list, sc: np.ndarray) -> list:
    """[[doc_id, idx, round(cos, 4)], ...] theo thứ tự cặp vào; bỏ cặp không có embedding."""
    out = []
    for k, (d, i) in enumerate(pairs_q):
        out.append([d, i, round(float(sc[k]), 4)])
    return out


def map_pairs(pairs: dict, row_of: dict) -> tuple:
    """({qid: [(doc_id, idx, row), ...]}, số doc thiếu embedding)."""
    out, missing = {}, 0
    for q, rows in pairs.items():
        keep = []
        for d, i in rows:
            r = row_of.get(int(d))
            if r is None:
                missing += 1
                continue
            keep.append((int(d), i, r))
        out[str(q)] = keep
    return out, missing


def score_query_set(embs: np.ndarray, qe: np.ndarray, mapped: dict, qids: list) -> dict:
    """{qid: [[doc_id, idx, cosine], ...]} cho một bộ truy vấn (hàng qe theo thứ tự qids)."""
    out = {}
    for k, q in enumerate(qids):
        rows = mapped.get(str(q), [])
        sc = pair_cosines(embs, qe[k], [r for _, _, r in rows])
        out[str(q)] = build_hits([(d, i) for d, i, _ in rows], sc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, nargs="+", help="thư mục blk_*.npz (chỉ đọc)")
    ap.add_argument("--pairs", required=True, help="JSON {qid: [[doc_id, idx], ...]}")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--queries2", default="", help="JSONL truy vấn thứ 2 (HyDE)")
    ap.add_argument("--query-len2", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out2", default="")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    if bool(a.queries2) != bool(a.out2):
        ap.error("cần --queries2 cùng --out2")

    t0 = time.time()
    ids, embs = load_embeddings(a.emb)
    row_of = doc_row_map(ids)
    pairs = json.loads(Path(a.pairs).read_text(encoding="utf-8"))
    mapped, missing = map_pairs({str(q): v for q, v in pairs.items()}, row_of)
    n_cap = sum(len(v) for v in mapped.values())
    print(f"emb {embs.shape}, {n_cap} cặp, {missing} doc thiếu embedding ({time.time() - t0:.0f}s)",
          flush=True)

    sets = [(a.queries, a.query_len, a.out)]
    if a.queries2:
        sets.append((a.queries2, a.query_len2, a.out2))
    rep = {"cap": n_cap, "doc_thieu_emb": missing, "bo_truy_van": []}
    for qpath, qlen, outp in sets:
        qs = [json.loads(l) for l in Path(qpath).read_text(encoding="utf-8").splitlines() if l.strip()]
        qe = embed_queries(a.model, [q["query"] for q in qs], qlen, a.batch)
        out = score_query_set(embs, qe, mapped, [q["id"] for q in qs])
        Path(outp).write_text(json.dumps(out), encoding="utf-8")
        nd = [len(v) for v in out.values()]
        rep["bo_truy_van"].append({"queries": qpath, "query_len": qlen, "out": outp,
                                   "so_cau": len(out), "TB_cap_cau": sum(nd) / max(len(nd), 1)})
        print(f"ghi {outp}: {len(out)} câu, TB {sum(nd) / max(len(nd), 1):.1f} cặp/câu "
              f"({time.time() - t0:.0f}s)", flush=True)
    rep["tong_s"] = time.time() - t0
    if a.report:
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(f"tổng {rep['tong_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
