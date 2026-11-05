"""Cắt relevant_docs của bài nộp còn --n-docs doc đầu (giữ nguyên thứ tự xếp hạng), giữ nguyên relevant_chunks.

Chunk có doc_id ngoài số doc giữ lại KHÔNG bị xoá âm thầm mà được đếm và báo ra (kỳ vọng 0 vì chunk
lấy từ 30 doc đầu). In thống kê: số câu, doc/câu (min/TB/max), chunk/câu (min/TB/max), chunk ngoài, doc trùng.
"""
import argparse
import json
import sys
from pathlib import Path


def truncate_sub(sub: list, n_docs: int) -> tuple[list, list[tuple]]:
    """Trả (bài mới với relevant_docs[:n_docs] giữ thứ tự, [(id, doc_id) của chunk nằm ngoài doc giữ lại])."""
    out, outside = [], []
    for item in sub:
        kept = item["relevant_docs"][:n_docs]
        kept_set = {str(d) for d in kept}
        for ch in item["relevant_chunks"]:
            if str(ch.get("doc_id")) not in kept_set:
                outside.append((item["id"], ch.get("doc_id")))
        out.append({"id": item["id"], "relevant_docs": list(kept), "relevant_chunks": item["relevant_chunks"]})
    return out, outside


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, required=True)
    a = ap.parse_args()

    sub = json.loads(Path(a.inp).read_text(encoding="utf-8"))
    out, outside = truncate_sub(sub, a.n_docs)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    nd = [len(s["relevant_docs"]) for s in out]
    nc = [len(s["relevant_chunks"]) for s in out]
    dup = sum(len(s["relevant_docs"]) - len({str(d) for d in s["relevant_docs"]}) for s in out)
    n = max(len(out), 1)
    print(f"ghi {a.out}: {len(out)} câu, doc/câu min {min(nd)} TB {sum(nd) / n:.2f} max {max(nd)}, "
          f"chunk/câu min {min(nc)} TB {sum(nc) / n:.2f} max {max(nc)}, "
          f"chunk ngoài relevant_docs: {len(outside)}, doc trùng: {dup}", flush=True)
    if outside:
        print(f"chunk nằm ngoài doc giữ lại (id, doc_id): {outside[:20]}", flush=True)


if __name__ == "__main__":
    main()
