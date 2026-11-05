"""Chọn lại chunk theo điểm chunk + điểm doc hợp nhất: s = z(logit chunk) + z(điểm doc rrf3d2).

Đầu vào:
  --rerank: chunk_dw_u900.json.zst {qid: [[doc_id, doc_rank, chunk_idx, logit, chunk_text], ...]}
  --fused:  cache hợp nhất {qid: [[doc_id, idx, fused], ...]} (fuse3.py), sắp giảm dần
  --ref-chunks: chunks_d900t015.json, cho k_q = số chunk cũ của mỗi câu (và để so thống kê)
Mỗi câu q:
  zc = z-score logit trên mọi chunk ứng viên của q (std 0 -> 0)
  zd = z-score điểm fused của doc chứa chunk, mean/std trên mọi doc của q trong --fused;
       doc không có trong --fused -> zd = min zd của q
  sắp s giảm dần (ổn định: đồng điểm giữ thứ tự cũ), bỏ trùng nguyên văn, lấy k_q chunk đầu.
Xuất định dạng --chunks-from: [{id, relevant_docs: [], relevant_chunks: [{doc_id, chunk_text}]}].

Dùng:
  python scripts/chunk_docfuse.py --rerank chunk_dw_u900.json.zst --fused cache_rrf3d2.json \
      --ref-chunks chunks_d900t015.json --out chunks_rrf3d2_cf.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from chunks_from_rerank import load_rerank


def zscores(vals: list) -> list:
    if not vals:
        return []
    m = sum(vals) / len(vals)
    sd = statistics.pstdev(vals)
    if sd == 0:
        return [0.0] * len(vals)
    return [(v - m) / sd for v in vals]


def doc_z(fused_hits: list) -> dict:
    """{str(doc_id): z điểm fused} trên mọi doc của câu."""
    ids = [str(h[0]) for h in fused_hits]
    return dict(zip(ids, zscores([float(h[2]) for h in fused_hits])))


def rescore_q(rows: list, fused_hits: list, wd: float = 1.0) -> list:
    """[(s, row)] với s = zc + wd * zd, sắp s giảm dần, ổn định."""
    zc = zscores([float(r[3]) for r in rows])
    zdm = doc_z(fused_hits)
    zmin = min(zdm.values()) if zdm else 0.0
    scored = [(c + wd * zdm.get(str(r[0]), zmin), r) for c, r in zip(zc, rows)]
    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    return [scored[i] for i in order]


def pick_q(rows: list, fused_hits: list, k: int, wd: float = 1.0) -> list[dict]:
    out, seen = [], set()
    for _, r in rescore_q(rows, fused_hits, wd):
        if len(out) >= k:
            break
        if r[4] in seen:
            continue
        seen.add(r[4])
        out.append({"doc_id": r[0], "chunk_text": r[4]})
    return out


def chunk_docfuse(rerank: dict, fused: dict, ref_chunks: list, wd: float = 1.0) -> list:
    rr = {str(q): v for q, v in rerank.items()}
    fu = {str(q): v for q, v in fused.items()}
    out = []
    for it in ref_chunks:
        q = str(it["id"])
        k = len(it["relevant_chunks"])
        out.append({"id": it["id"], "relevant_docs": [],
                    "relevant_chunks": pick_q(rr.get(q, []), fu.get(q, []), k, wd)})
    return out


def pct(vals: list, p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def compare(new: list, old: list, fused: dict) -> dict:
    """Thống kê so sánh chunk mới với cũ; hạng rrf 1-based (doc không có trong fused: len+1)."""
    om = {str(it["id"]): it["relevant_chunks"] for it in old}
    fu = {str(q): v for q, v in fused.items()}
    nq = max(len(new), 1)
    ov = nd_old = nd_new = nc_old = nc_new = 0
    rk_old, rk_new = [], []
    for it in new:
        q = str(it["id"])
        a, b = it["relevant_chunks"], om.get(q, [])
        nc_new += len(a)
        nc_old += len(b)
        ov += len({(str(c["doc_id"]), c["chunk_text"]) for c in a} & {(str(c["doc_id"]), c["chunk_text"]) for c in b})
        nd_new += len({str(c["doc_id"]) for c in a})
        nd_old += len({str(c["doc_id"]) for c in b})
        hits = fu.get(q, [])
        rank = {}
        for i, h in enumerate(hits):
            rank.setdefault(str(h[0]), i + 1)
        rk_new += [rank.get(str(c["doc_id"]), len(hits) + 1) for c in a]
        rk_old += [rank.get(str(c["doc_id"]), len(hits) + 1) for c in b]

    def dist(v):
        return {"TB": sum(v) / max(len(v), 1), "p50": pct(v, 0.5), "p90": pct(v, 0.9)}

    return {"chunk/câu mới": nc_new / nq, "chunk/câu cũ": nc_old / nq, "overlap chunk TB": ov / nq,
            "doc khác nhau/câu cũ": nd_old / nq, "doc khác nhau/câu mới": nd_new / nq,
            "hạng rrf doc nguồn cũ": dist(rk_old), "hạng rrf doc nguồn mới": dist(rk_new)}


def identical_diff(a: list, b: list) -> int:
    """Số câu có relevant_chunks khác nhau (kể cả thứ tự) giữa hai file chunk, ghép theo id."""
    bm = {str(it["id"]): it["relevant_chunks"] for it in b}
    return sum(1 for it in a if it["relevant_chunks"] != bm.get(str(it["id"])))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--ref-chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wd", type=float, default=1.0, help="trọng số zd: s = zc + wd * zd (mặc định 1.0)")
    ap.add_argument("--compare-with", default="", help="chunks JSON để so sánh nguyên văn (kiểm tái lập)")
    a = ap.parse_args()
    rerank = load_rerank(a.rerank)
    fused = json.loads(Path(a.fused).read_text(encoding="utf-8"))
    ref = json.loads(Path(a.ref_chunks).read_text(encoding="utf-8"))
    out = chunk_docfuse(rerank, fused, ref, a.wd)
    del rerank
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    short = sum(1 for n, o in zip(out, ref) if len(n["relevant_chunks"]) < len(o["relevant_chunks"]))
    print(f"ghi {a.out} (wd={a.wd}): {len(out)} câu, câu thiếu chunk so với cũ: {short}", flush=True)
    print(json.dumps(compare(out, ref, fused), ensure_ascii=False), flush=True)
    if a.compare_with:
        prev = json.loads(Path(a.compare_with).read_text(encoding="utf-8"))
        print(f"so với {a.compare_with}: câu khác nhau {identical_diff(out, prev)} / {len(out)}", flush=True)
        print("so với file trên (cũ = file đó): " + json.dumps(compare(out, prev, fused), ensure_ascii=False),
              flush=True)


if __name__ == "__main__":
    main()
