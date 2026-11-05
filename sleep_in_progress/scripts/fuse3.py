"""Hợp nhất 3 thứ hạng (bge pool, Qwen 8B, dense) bằng RRF có trọng số trên ứng viên = doc của pool[q].

fused(d) = sum_s w_s / (k + rank_s(d)), rank 1-based theo thứ tự trong file nguồn s; doc không có trong
nguồn s thì nguồn đó không đóng góp. Đồng điểm giữ thứ tự pool (bge). idx lấy từ pool, thiếu thì từ dense.
Đầu ra {qid: [[doc_id, idx, fused], ...]} sắp giảm dần.

Dùng:
  python scripts/fuse3.py --pool cache_pool_rr.json --q8b cache_qwen8b_sorted.json \
      --dense dense_full_top200.json --w-bge 1 --w-q8b 1 --w-dense 2 --out cache_rrf3d2.json
"""
import argparse
import json
import sys
from pathlib import Path


def ranks(hits: list) -> dict:
    """{str(doc_id): rank 1-based}, giữ lần xuất hiện đầu."""
    out = {}
    for i, h in enumerate(hits or []):
        out.setdefault(str(h[0]), i + 1)
    return out


def fuse_q(pool_hits: list, q8b_hits: list, dense_hits: list, weights=(1.0, 1.0, 1.0), k: int = 60,
           bm25_hits: list | None = None, extra: list | None = None) -> list:
    """weights = (bge, 8B, dense[, bm25]); thiếu phần tử bm25 thì nguồn BM25 không đóng góp.
    extra = [(hits, weight), ...]: nguồn bổ sung tuỳ ý, mỗi nguồn đóng góp w/(k+rank)."""
    srcs = [ranks(pool_hits), ranks(q8b_hits), ranks(dense_hits), ranks(bm25_hits)]
    weights = tuple(weights) + (0.0,) * (4 - len(weights))
    for hits, w in extra or []:
        srcs.append(ranks(hits))
        weights += (w,)
    dense_idx = {str(h[0]): h[1] for h in dense_hits or []}
    rows, seen = [], set()
    for pos, h in enumerate(pool_hits):
        d = str(h[0])
        if d in seen:
            continue
        seen.add(d)
        sc = sum(w / (k + r[d]) for w, r in zip(weights, srcs) if w and d in r)
        idx = h[1] if h[1] is not None else dense_idx.get(d)
        rows.append((sc, pos, [h[0], idx, round(sc, 8)]))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [r for _, _, r in rows]


def fuse3(pool: dict, q8b: dict, dense: dict, weights=(1.0, 1.0, 1.0), k: int = 60,
          bm25: dict | None = None, extras: list | None = None) -> dict:
    q8b = {str(q): v for q, v in q8b.items()}
    dense = {str(q): v for q, v in dense.items()}
    bm25 = {str(q): v for q, v in (bm25 or {}).items()}
    ex = [({str(q): v for q, v in d.items()}, w) for d, w in (extras or [])]
    return {str(q): fuse_q(hits, q8b.get(str(q), []), dense.get(str(q), []), weights, k, bm25.get(str(q), []),
                           [(d.get(str(q), []), w) for d, w in ex])
            for q, hits in pool.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--q8b", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument("--w-bge", type=float, default=1.0)
    ap.add_argument("--w-q8b", type=float, default=1.0)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--bm25", default="", help="cache BM25 (vd cache_vi_k300.json), tuỳ chọn")
    ap.add_argument("--w-bm25", type=float, default=0.0)
    ap.add_argument("--bm25-n", type=int, default=300, help="chỉ dùng top-n của BM25")
    ap.add_argument("--extra", nargs=2, action="append", default=[], metavar=("CACHE", "W"),
                    help="nguồn bổ sung (lặp được): file cache {qid: [[doc_id, ...], ...]} và trọng số")
    ap.add_argument("--extra-n", type=int, default=200, help="chỉ dùng top-n của mỗi nguồn --extra")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--n", type=int, default=30, help="top-n để in số doc dense ngoài pool")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    pool, q8b, dense = rd(a.pool), rd(a.q8b), rd(a.dense)
    bm25 = None
    if a.bm25 and a.w_bm25:
        bm25 = {q: v[: a.bm25_n] for q, v in rd(a.bm25).items()}
    extras = [({q: v[: a.extra_n] for q, v in rd(c).items()}, float(w)) for c, w in a.extra]
    out = fuse3(pool, q8b, dense, (a.w_bge, a.w_q8b, a.w_dense, a.w_bm25), a.k, bm25, extras)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    if bm25:
        cov = sum(len({str(h[0]) for h in hits} & {str(h[0]) for h in bm25.get(str(q), [])})
                  for q, hits in pool.items()) / max(len(pool), 1)
        print(f"BM25 top-{a.bm25_n} w={a.w_bm25}: TB {cov:.1f} doc pool/câu có hạng BM25", flush=True)
    outside = 0
    for q, hits in dense.items():
        pset = {str(h[0]) for h in pool.get(str(q), [])}
        outside += sum(1 for h in hits[: a.n] if str(h[0]) not in pset)
    nq = max(len(out), 1)
    print(f"ghi {a.out}: {len(out)} câu, TB {sum(len(v) for v in out.values()) / nq:.1f} ứng viên/câu, "
          f"w=({a.w_bge},{a.w_q8b},{a.w_dense}) k={a.k}; dense top-{a.n} ngoài pool: TB {outside / nq:.2f} doc/câu",
          flush=True)


if __name__ == "__main__":
    main()
