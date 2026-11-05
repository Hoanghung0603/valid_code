"""Tính cặp MỚI cần rerank = (dense top-K ∪ BM25 top-M) − doc đã có logit trong pool cũ.

Ứng viên: cache dense {qid: [[doc_id, idx, cosine], ...]} (dense_topk.py/embed_all.py) lấy --dense-k đầu,
thêm cache BM25 {qid: [[doc_id, idx, điểm], ...]} (search_cache.py) lấy --bm-k đầu, bỏ doc đã có logit
trong --old ({qid: [[doc_id, idx, logit], ...]}) và bỏ trùng (nguồn trước thắng: dense trước, BM25 sau).
In số cặp mới; nếu tổng vượt --max-pairs thì tính lại với --dense-k-fallback (mặc định 300) và BÁO LẠI.

Ghi --out ứng viên rerank_run.py: {qid: [[doc_id, idx], ...]}.

Ví dụ:
  python scripts/pool_extend.py --dense data/dense_full_top400.json --bm25 data/cache_vi_k300.json \
      --old data/rerank_pool.json --out data/cands_pool2.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def collect_new(dense_hits: list, bm25_hits: list, have: set, dense_k: int, bm_k: int) -> tuple:
    """Cặp mới [[doc_id, idx], ...] của một câu: dense[:dense_k] rồi BM25[:bm_k], bỏ doc trong have/trùng.

    idx lấy từ nguồn đầu tiên xuất hiện; trả thêm danh sách lệch idx giữa hai nguồn
    [(doc_id, idx_dense, idx_bm25), ...] (doc có trong cả hai lát).
    """
    out, seen = [], set()
    both: dict = {}
    for name, hits, k in (("dense", dense_hits, dense_k), ("bm25", bm25_hits, bm_k)):
        for item in hits[:k]:
            d, i = int(item[0]), int(item[1])
            both.setdefault(d, {})[name] = i
            if d in have or d in seen:
                continue
            seen.add(d)
            out.append([d, i])
    mismatch = [(d, v["dense"], v["bm25"]) for d, v in both.items()
                if "dense" in v and "bm25" in v and v["dense"] != v["bm25"]]
    return out, mismatch


def extend_pairs(dense: dict, bm25: dict, old: dict, dense_k: int = 400, bm_k: int = 300,
                 max_pairs: int = 300000, dense_k_fallback: int = 300) -> tuple:
    """Cặp mới cho mọi câu; nếu tổng > max_pairs thì lùi --dense-k xuống dense_k_fallback.

    Trả (pairs {qid: [[doc_id, idx], ...]}, n_gốc, dùng_fallback, n_sau, n_mismatch_idx).
    """
    def build(dk: int):
        pairs, n, nmis = {}, 0, 0
        for q, hits in old.items():
            have = {int(h[0]) for h in hits}
            new, mm = collect_new(dense.get(q, []), bm25.get(q, []), have, dk, bm_k)
            pairs[q] = new
            n += len(new)
            nmis += len(mm)
        return pairs, n, nmis

    pairs, n0, nmis = build(dense_k)
    if n0 > max_pairs and dense_k_fallback < dense_k:
        pairs, n1, nmis = build(dense_k_fallback)
        return pairs, n0, True, n1, nmis
    return pairs, n0, False, n0, nmis


def per_query_sizes(pairs: dict) -> tuple:
    """(min, TB, max) số cặp mới mỗi câu."""
    sizes = [len(v) for v in pairs.values()]
    return min(sizes), sum(sizes) / max(len(sizes), 1), max(sizes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="JSON {qid: [[doc_id, idx, cosine], ...]} giảm dần")
    ap.add_argument("--bm25", required=True, help="JSON cache search_cache.py (cache_vi_k300.json)")
    ap.add_argument("--old", required=True, help="JSON pool đã có logit (rerank_pool.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense-k", type=int, default=400)
    ap.add_argument("--bm-k", type=int, default=300)
    ap.add_argument("--max-pairs", type=int, default=300000, help="vượt thì lùi --dense-k-fallback")
    ap.add_argument("--dense-k-fallback", type=int, default=300)
    a = ap.parse_args()

    dense = json.loads(Path(a.dense).read_text(encoding="utf-8"))
    bm25 = json.loads(Path(a.bm25).read_text(encoding="utf-8"))
    old = json.loads(Path(a.old).read_text(encoding="utf-8"))
    missing = [q for q in old if q not in dense or q not in bm25]
    if missing:
        raise SystemExit(f"{len(missing)} câu thiếu cache dense/BM25 (vd {missing[:5]})")

    pairs, n0, used_fb, n1, nmis = extend_pairs(
        dense, bm25, old, a.dense_k, a.bm_k, a.max_pairs, a.dense_k_fallback)
    lo, avg, hi = per_query_sizes(pairs)
    print(f"dense top-{a.dense_k} ∪ BM25 top-{a.bm_k} − pool cũ: {n0} cặp mới", flush=True)
    if used_fb:
        print(f"VƯỢT --max-pairs {a.max_pairs} → lùi dense top-{a.dense_k_fallback}: {n1} cặp mới", flush=True)
    n = n1
    print(f"cặp mới/câu: min {lo} TB {avg:.1f} max {hi}; lệch idx giữa hai nguồn: {nmis}", flush=True)
    Path(a.out).write_text(json.dumps(pairs), encoding="utf-8")
    print(f"ghi {a.out}: {len(pairs)} câu, {n} cặp", flush=True)


if __name__ == "__main__":
    main()
