"""Hợp nhất xếp hạng BM25 và dense (bge-m3) bằng Reciprocal Rank Fusion, ghi ra cache cùng định dạng search_cache.py.

--mode rrf (mặc định): RRF các xếp hạng BM25 + dense (+ rerank nếu truyền --rerank) với trọng số đều 1.0.
--mode rerank: xếp top-N ứng viên BM25 theo logit rerank giảm dần (hoà giữ thứ tự BM25, thiếu logit xếp cuối).
Đầu ra dùng được ngay cho `bipipeline.py --cache-vi` (lấy n đầu theo thứ tự hợp nhất). Chỉ hợp nhất các doc có trong
tập ứng viên (top-N BM25). Doc không có embedding (dense = None) chỉ nhận phần BM25.
"""
import argparse
import json
import sys
from pathlib import Path


def rrf_fuse_multi(rankings: list, k: int = 60) -> list:
    """rankings: [(hits, trọng số)] với hits = [[doc_id, idx, score]] đã theo thứ hạng (đầu danh sách = hạng 1).

    Trả [[doc_id, idx, điểm RRF]] giảm dần; hoà điểm giữ thứ tự của xếp hạng đứng trước trong rankings.
    """
    score: dict = {}
    idx_of: dict = {}
    order: dict = {}
    for j, (hits, w) in enumerate(rankings):
        for r, (d, i, _) in enumerate(hits):
            score[d] = score.get(d, 0.0) + w / (k + r + 1)
            if j == 0:
                idx_of[d] = i
                order[d] = r
            else:
                idx_of.setdefault(d, i)
                order.setdefault(d, 10 ** 9 * j + r)
    docs = sorted(score, key=lambda d: (-score[d], order[d]))
    return [[d, idx_of[d], round(score[d], 6)] for d in docs]


def sort_by_score(hits: list) -> list:
    """Xếp [[doc_id, idx, score]] theo điểm giảm dần; hoà giữ thứ tự gốc; score None xuống cuối."""
    return sorted(hits, key=lambda h: (h[2] is None, 0.0 if h[2] is None else -h[2]))


def merge_cosines(primary: list, secondary: list) -> list:
    """Gộp hai bảng cosine [[doc_id, idx, cosine], ...]; doc có ở cả hai lấy bảng primary (cùng embedding nên ~bằng);
    bỏ cosine None; xếp theo cosine giảm dần, hoà giữ thứ tự primary trước."""
    out, seen = [], set()
    for hits in (primary, secondary):
        for d, i, c in hits:
            if c is not None and d not in seen:
                seen.add(d)
                out.append([d, i, c])
    return sort_by_score(out)


def pool_rrf3(bm25_hits: list, dense_primary: list, dense_secondary: list, pool_hits: list, k: int = 60) -> list:
    """RRF 3 chiều (k=60, trọng số đều) trên pool ứng viên BM25 top-200 ∪ dense-full top-200.

    bm25_hits: [[doc_id, idx, _], ...] theo thứ hạng BM25 (chỉ doc pool BM25 — ai thiếu thì không nhận điểm BM25);
    dense_primary/dense_secondary: bảng cosine [[doc_id, idx, cosine], ...] (primary ưu tiên khi doc trùng);
    pool_hits: [[doc_id, idx, logit], ...] toàn pool theo thứ tự gốc (xếp hạng rerank = logit giảm dần).
    Trả [[doc_id, idx, điểm RRF]] giảm dần; định dạng cache của search_cache.py.
    """
    rr = sort_by_score(pool_hits)
    return rrf_fuse_multi([
        (bm25_hits, 1.0),
        (merge_cosines(dense_primary, dense_secondary), 1.0),
        (rr, 1.0),
    ], k)


def rrf_fuse(bm25_hits: list, dense_hits: list, k: int = 60, w_dense: float = 1.0, w_bm25: float = 1.0,
             rerank_hits: list | None = None, w_rerank: float = 1.0) -> list:
    """bm25_hits: [[doc_id, idx, score]] giảm dần theo BM25; dense_hits: [[doc_id, idx, cosine|None]] (cùng tập doc).

    rerank_hits: [[doc_id, idx, logit|None]] — nếu có thì thành RRF 3 chiều (BM25 + dense + rerank).
    Trả [[doc_id, idx, điểm RRF]] giảm dần; hoà điểm giữ thứ tự BM25.
    """
    rankings = [(bm25_hits, w_bm25),
                (sorted((h for h in dense_hits if h[2] is not None), key=lambda h: -h[2]), w_dense)]
    if rerank_hits is not None:
        rankings.append((sorted((h for h in rerank_hits if h[2] is not None), key=lambda h: -h[2]), w_rerank))
    return rrf_fuse_multi(rankings, k)


def rerank_order(bm25_hits: list, rerank_hits: list, n: int = 200) -> list:
    """Xếp top-n ứng viên BM25 theo logit rerank giảm dần; hoà giữ thứ tự BM25; thiếu logit xếp cuối.

    Trả [[doc_id, idx, logit]] giảm dần theo logit.
    """
    rr = {h[0]: h[2] for h in rerank_hits if h[2] is not None}
    out = [[d, i, rr.get(d)] for d, i, _ in bm25_hits[:n]]
    out.sort(key=lambda h: (h[2] is None, -(h[2] if h[2] is not None else 0.0)))
    return [[d, i, None if s is None else round(s, 6)] for d, i, s in out]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", required=True, help="cache_vi_k300.json")
    ap.add_argument("--dense", default=None, help="JSON từ embed_run.py")
    ap.add_argument("--rerank", default=None, help="JSON từ rerank_run.py ({qid: [[doc_id, idx, logit], ...]})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200, help="chỉ dùng top-N BM25 làm ứng viên")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--mode", choices=["rrf", "rerank"], default="rrf", help="rrf: hợp nhất RRF; rerank: xếp theo logit")
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--w-rerank", type=float, default=1.0)
    a = ap.parse_args()

    bm = json.loads(Path(a.bm25).read_text(encoding="utf-8"))
    dn = json.loads(Path(a.dense).read_text(encoding="utf-8")) if a.dense else {}
    rr = json.loads(Path(a.rerank).read_text(encoding="utf-8")) if a.rerank else {}
    out, overlap30, moved = {}, 0, 0
    for q, hits in bm.items():
        base = hits[: a.n]
        if a.mode == "rerank":
            fused = rerank_order(hits, rr.get(q, []), a.n)
        else:
            fused = rrf_fuse(base, dn.get(q, []), a.k, a.w_dense,
                             rerank_hits=rr.get(q, []) if a.rerank else None, w_rerank=a.w_rerank)
        out[q] = fused
        overlap30 += len({h[0] for h in base[:30]} & {h[0] for h in fused[:30]})
        moved += sum(1 for x, y in zip(base[:30], fused[:30]) if x[0] != y[0])
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    n = max(len(out), 1)
    print(f"ghi {a.out}: {len(out)} câu. Top-30 trùng BM25: TB {overlap30 / n:.1f}/30; vị trí top-30 đổi: TB {moved / n:.1f}/30", flush=True)


if __name__ == "__main__":
    main()
