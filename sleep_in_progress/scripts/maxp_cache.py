"""Cache điểm doc = max logit cửa sổ trong docwin, sắp giảm dần — cho bài chẩn đoán maxp_top30.

docwin (docwin_run.py --out): {qid: {doc_id: [[start_char, end_char, logit], ...]}}. Mỗi doc:
điểm = max logit các cửa sổ; doc không có cửa sổ nào bị bỏ qua. Ghi cache
{qid: [[doc_id, idx, score], ...]} — thứ tự file = thứ tự xếp hạng (union_sub/chunks_wholedoc
--orders rr dùng được ngay). Đồng điểm: vị trí trong --cands (nếu có), không thì doc_id tăng dần.

--bge: cache tham chiếu (vd cache_pool_rr) để in overlap top-K với maxp.

Dùng:
  python scripts/maxp_cache.py --docwin docwin_top100.json.zst --cands cands_chunkdocs100.json \
    --out cache_maxp.json --bge cache_pool_rr.json --top 30
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _did(d):
    return int(d) if str(d).isdigit() else d


def doc_max_scores(wins_q: dict) -> dict:
    """{doc_id: max logit cửa sổ}; doc không có cửa sổ bị bỏ qua."""
    out = {}
    for d, wins in wins_q.items():
        if wins:
            out[_did(d)] = max(w[2] for w in wins)
    return out


def build_maxp(docwin: dict, cands: dict | None = None) -> dict:
    """{qid: [[doc_id, idx, score], ...]} sắp max-logit giảm dần (None nếu thiếu điểm — không có)."""
    out = {}
    for q, wins_q in docwin.items():
        scores = doc_max_scores(wins_q)
        pos, idx = {}, {}
        for i, item in enumerate((cands or {}).get(q, [])):
            pos[_did(item[0])] = i
            idx[_did(item[0])] = item[1] if len(item) > 1 else 0
        rows = sorted(scores, key=lambda d: (-scores[d], pos.get(d, 1 << 30), d))
        out[q] = [[d, idx.get(d, 0), scores[d]] for d in rows]
    return out


def overlap_topk(rows_a: list, rows_b: list, k: int) -> int:
    """Số doc_id trùng giữa top-k của hai danh sách xếp hạng [[doc_id, ...], ...]."""
    ka = {r[0] for r in rows_a[:k]}
    kb = {r[0] for r in rows_b[:k]}
    return len(ka & kb)


def overlap_stats(maxp: dict, ref: dict, k: int) -> dict:
    """Trùng top-K maxp vs ref trên các câu chung: {"câu", "trung bình", "min", "max"}."""
    vals = [overlap_topk(maxp[q], ref[q], k) for q in maxp if q in ref]
    n = len(vals) or 1
    return {"câu": len(vals), "trung bình": round(sum(vals) / n, 4), "min": min(vals, default=0),
            "max": max(vals, default=0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docwin", required=True, help=".json/.json.zst của docwin_run.py")
    ap.add_argument("--cands", default="", help="JSON {qid: [[doc_id, idx], ...]} để lấy idx + thứ tự hoà")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bge", default="", help="cache tham chiếu để in overlap top-K")
    ap.add_argument("--top", type=int, default=30, help="K tính overlap")
    a = ap.parse_args()

    from chunks_from_docwin import load_json

    docwin = load_json(a.docwin)
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8")) if a.cands else None
    maxp = build_maxp(docwin, cands)
    Path(a.out).write_text(json.dumps(maxp, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for v in maxp.values())
    nc = [len(v) for v in maxp.values()]
    print(f"ghi {a.out}: {len(maxp)} câu, {n} dòng, doc/câu min {min(nc)} TB {sum(nc) / len(nc):.2f} max {max(nc)}",
          flush=True)
    if a.bge:
        ref = json.loads(Path(a.bge).read_text(encoding="utf-8"))
        st = overlap_stats(maxp, ref, a.top)
        print(f"overlap top-{a.top} maxp vs {Path(a.bge).name}: {json.dumps(st, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
