"""Bài full8b: số doc thích nghi theo logit bge (pool), thứ tự doc theo ensemble 8B rồi pool.

Mỗi câu q:
  n_q = adaptive_n(pool[q], thr, lo, hi)
  n_q <= len(top8b): base = n_q doc đầu của ens8z[q] (top8b = 100 doc đầu)
  ngược lại: base = 100 doc đầu ens8z[q] + doc của pool[q] theo thứ tự (bỏ trùng), cắt còn n_q
  relevant_chunks = nguyên từ --chunks-from; relevant_docs = base + doc_id của chunk chưa có.
In: TB n_q, TB len(relevant_docs), TB |giao| với relevant_docs của --ref.

Dùng:
  python scripts/full8b_sub.py --pool cache_pool_rr.json --ens cache_ens8z.json \
      --chunks-from chunks_d900t015.json --ref submission_d900t015.json --out submission_full8b.json
"""
import argparse
import json
import sys
from pathlib import Path

from union_sub import adaptive_n


def base_docs(pool_hits: list, ens_hits: list, n: int, top_ens: int = 100) -> list:
    """n doc nền: ens trước (tối đa top_ens), bù bằng pool theo thứ tự, không trùng."""
    ens_ids = [h[0] for h in ens_hits][:top_ens]
    if n <= top_ens:
        return ens_ids[:n]
    docs, seen = list(ens_ids), {str(d) for d in ens_ids}
    for h in pool_hits:
        if len(docs) >= n:
            break
        if str(h[0]) not in seen:
            seen.add(str(h[0]))
            docs.append(h[0])
    return docs[:n]


def full8b_sub(pool: dict, ens: dict, chunks_sub: list, thr: float = -2.6, lo: int = 40, hi: int = 400,
               top_ens: int = 100) -> tuple[list, list]:
    """Trả (bài nộp, danh sách n_q) theo thứ tự câu của chunks_sub."""
    pool = {str(k): v for k, v in pool.items()}
    ens = {str(k): v for k, v in ens.items()}
    out, ns = [], []
    for it in chunks_sub:
        q = str(it["id"])
        n = adaptive_n(pool[q], thr, lo, hi)
        ns.append(n)
        docs = base_docs(pool[q], ens.get(q, []), n, top_ens)
        seen = {str(d) for d in docs}
        for ch in it["relevant_chunks"]:
            d = ch["doc_id"]
            if str(d) not in seen:
                seen.add(str(d))
                docs.append(d)
        out.append({"id": it["id"], "relevant_docs": docs, "relevant_chunks": it["relevant_chunks"]})
    return out, ns


def mean_inter(sub: list, ref: list) -> float:
    rmap = {str(it["id"]): {str(d) for d in it["relevant_docs"]} for it in ref}
    vals = [len({str(d) for d in s["relevant_docs"]} & rmap.get(str(s["id"]), set())) for s in sub]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="cache_pool_rr.json {qid: [[doc_id, idx, logit], ...]}")
    ap.add_argument("--ens", required=True, help="cache_ens8z.json, sắp giảm dần")
    ap.add_argument("--chunks-from", required=True)
    ap.add_argument("--ref", default="", help="bài nộp để đo giao relevant_docs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=-2.6)
    ap.add_argument("--lo", type=int, default=40)
    ap.add_argument("--hi", type=int, default=400)
    ap.add_argument("--top-ens", type=int, default=100)
    a = ap.parse_args()

    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    ens = json.loads(Path(a.ens).read_text(encoding="utf-8"))
    chunks_sub = json.loads(Path(a.chunks_from).read_text(encoding="utf-8"))
    out, ns = full8b_sub(pool, ens, chunks_sub, a.thr, a.lo, a.hi, a.top_ens)
    del chunks_sub
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    nq = max(len(out), 1)
    nd = [len(s["relevant_docs"]) for s in out]
    print(f"ghi {a.out}: {len(out)} câu, TB n_q {sum(ns) / nq:.2f} (min {min(ns)} max {max(ns)}, "
          f">100: {sum(1 for n in ns if n > a.top_ens)} câu), TB len(relevant_docs) {sum(nd) / nq:.2f}", flush=True)
    if a.ref:
        ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
        print(f"TB |giao| relevant_docs với {a.ref}: {mean_inter(out, ref):.2f}", flush=True)


if __name__ == "__main__":
    main()
