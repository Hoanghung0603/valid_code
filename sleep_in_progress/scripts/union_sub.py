"""Tạo biến thể bài nộp: relevant_docs = N doc nền + rerank top-K + doc của chunk còn thiếu.

- --docs-from nhận hai định dạng: bài nộp (list {id, relevant_docs, ...}) HOẶC cache xếp hạng
  ({qid: [[doc_id, idx, score], ...]} hay {qid: [doc_id, ...]} — thứ tự trong file = thứ tự xếp hạng).
- Nền: N doc đầu theo thứ tự xếp hạng trong --docs-from, N = --docs-n.
- Thêm (khi --add-k > 0): lấy K doc đầu của --add-rerank ({qid: [[doc_id, idx, logit], ...]}) sắp theo
  logit giảm dần, hoà theo vị trí gốc; bỏ doc đã có, thêm theo thứ tự rerank.
- Cuối cùng: thêm doc_id của relevant_chunks trong --chunks-from chưa có (theo thứ tự xuất hiện lần đầu).
- relevant_chunks lấy nguyên từ --chunks-from (cùng nội dung, cùng thứ tự). Doc không trùng.

K=0 thì giống combine_sub trên N doc đầu (với N=60 là hành vi v14). In thống kê: số câu, doc/câu
(min/TB/max), TB doc thêm ngoài N, chunk ngoài relevant_docs, doc trùng.

Dùng:
  python scripts/union_sub.py --docs-from data/submission_v10.json --chunks-from data/submission_v12.json \
      --docs-n 60 --add-rerank data/rerank_top200.json --add-k 10 --out data/submission_u10.json
"""
import argparse
import json
import sys
from pathlib import Path


def rerank_order(cand: list) -> list:
    """doc_id theo thứ tự rerank: logit giảm dần, hoà theo vị trí ứng viên BM25 gốc; logit None xuống cuối."""

    def key(i: int):
        lg = cand[i][2]
        return (lg is None, 0.0 if lg is None else -lg, i)

    return [cand[i][0] for i in sorted(range(len(cand)), key=key)]


def load_ranked(obj) -> list:
    """[(id, [doc_id, ...])] theo thứ tự xếp hạng từ cache ({qid: [[doc_id, idx, score], ...]}
    hoặc {qid: [doc_id, ...]}) hoặc bài nộp (list {id, relevant_docs, ...})."""
    if isinstance(obj, list):
        return [(it["id"], list(it["relevant_docs"])) for it in obj]
    out = []
    for q, hits in obj.items():
        docs = [h[0] if isinstance(h, (list, tuple)) else h for h in hits]
        out.append((int(q), docs))
    return out


def adaptive_n(hits: list, thr: float, lo: int, hi: int) -> int:
    """Số doc theo ngưỡng điểm: đếm doc có điểm > thr (điểm None không tính), kẹp trong [lo, hi]."""
    k = sum(1 for h in hits if h[2] is not None and h[2] > thr)
    return min(max(k, lo), hi)


def union_sub(docs_sub, chunks_sub: list, rerank_map: dict | None = None,
              docs_n: int = 60, add_k: int = 0, thr: float | None = None,
              min_n: int = 0, max_n: int = 10**9) -> list:
    """Nền: docs_n doc đầu của thứ tự xếp hạng trong docs_sub (cache hoặc bài nộp), rồi rerank top-add_k
    (bỏ doc đã có), rồi doc của chunk còn thiếu; chunks lấy nguyên từ chunks_sub.

    thr khác None (chỉ với cache có điểm): số doc mỗi câu = adaptive_n(hits, thr, min_n, max_n) thay cho docs_n.
    """
    chunks_by_id = {str(it["id"]): it for it in chunks_sub}
    rmap = {str(k): v for k, v in (rerank_map or {}).items()}
    if thr is not None and isinstance(docs_sub, list):
        raise ValueError("thr cần docs_sub dạng cache có điểm")
    n = max(int(docs_n), 0)
    out = []
    for qid, ranked in load_ranked(docs_sub):
        ch_item = chunks_by_id[str(qid)]
        if thr is not None:
            n = adaptive_n(docs_sub[str(qid)], thr, min_n, max_n)
        docs = list(ranked)[:n]
        seen = {str(d) for d in docs}
        if add_k > 0:
            for d in rerank_order(rmap[str(qid)])[:add_k]:
                if str(d) not in seen:
                    seen.add(str(d))
                    docs.append(d)
        for ch in ch_item["relevant_chunks"]:
            d = ch["doc_id"]
            if str(d) not in seen:
                seen.add(str(d))
                docs.append(d)
        out.append({"id": qid, "relevant_docs": docs, "relevant_chunks": ch_item["relevant_chunks"]})
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-from", required=True,
                    help="bài nộp (list {id, relevant_docs,...}) hoặc cache xếp hạng ({qid: [[doc_id, idx, score], ...]})")
    ap.add_argument("--chunks-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs-n", type=int, default=60, help="số doc RRF đầu giữ lại")
    ap.add_argument("--add-rerank", default="", help="JSON {qid: [[doc_id, idx, logit], ...]} theo thứ tự BM25")
    ap.add_argument("--add-k", type=int, default=0, help="lấy K doc đầu theo thứ tự rerank để thêm")
    ap.add_argument("--logit-thr", type=float, default=None,
                    help="số doc theo ngưỡng điểm trong cache (thay --docs-n), kẹp bởi --min-n/--max-n")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10**9)
    a = ap.parse_args()
    if a.add_k > 0 and not a.add_rerank:
        ap.error("--add-k > 0 cần --add-rerank")

    docs_sub = json.loads(Path(a.docs_from).read_text(encoding="utf-8"))
    chunks_sub = json.loads(Path(a.chunks_from).read_text(encoding="utf-8"))
    rerank_map = json.loads(Path(a.add_rerank).read_text(encoding="utf-8")) if a.add_rerank else None
    out = union_sub(docs_sub, chunks_sub, rerank_map, a.docs_n, a.add_k, a.logit_thr, a.min_n, a.max_n)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    nd = [len(s["relevant_docs"]) for s in out]
    n = max(len(out), 1)
    add = [max(len(s["relevant_docs"]) - a.docs_n, 0) for s in out]
    chunks_by_id = {str(it["id"]): it for it in chunks_sub}
    outside = sum(
        1 for s in out
        for ch in chunks_by_id[str(s["id"])]["relevant_chunks"]
        if str(ch.get("doc_id")) not in {str(d) for d in s["relevant_docs"]}
    )
    dup = sum(len(s["relevant_docs"]) - len({str(d) for d in s["relevant_docs"]}) for s in out)
    print(f"ghi {a.out}: {len(out)} câu, doc/câu min {min(nd)} TB {sum(nd) / n:.2f} max {max(nd)}, "
          f"TB thêm ngoài {a.docs_n} doc: {sum(add) / n:.2f}, "
          f"chunk ngoài relevant_docs: {outside}, doc trùng: {dup}", flush=True)


if __name__ == "__main__":
    main()
