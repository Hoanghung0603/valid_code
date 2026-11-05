"""Tạo file chunk (định dạng bài nộp, relevant_docs rỗng) từ kết quả chunk_rerank.py.

Đầu vào: {qid: [[doc_id, doc_rank, chunk_idx, logit, chunk_text], ...]} đã sắp logit giảm dần (.json hoặc .json.zst).
Số chunk mỗi câu: k cố định (--k), hoặc theo ngưỡng logit (--thr) kẹp trong [--min-k, --max-k].
Bỏ chunk trùng nguyên văn trong cùng câu. Ghép với doc bằng union_sub.py --chunks-from.

Dùng:
  python scripts/chunks_from_rerank.py --rerank data/chunk_rerank_w_top50.json.zst --k 40 --out data/chunks_crw40.json
"""
import argparse
import json
import sys
from pathlib import Path


def load_rerank(path: str) -> dict:
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with p.open("rb") as f:
            raw = zstandard.ZstdDecompressor().stream_reader(f).read()
        return json.loads(raw)
    return json.loads(p.read_text(encoding="utf-8"))


def diversify(rows: list) -> list:
    """Xếp lại theo lượt: chunk tốt nhất của mỗi doc trước (giữ thứ tự logit), rồi chunk thứ 2 của mỗi doc, ..."""
    cnt: dict = {}
    keyed = []
    for i, r in enumerate(rows):
        j = cnt.get(r[0], 0)
        cnt[r[0]] = j + 1
        keyed.append((j, i, r))
    return [r for _, _, r in sorted(keyed, key=lambda x: (x[0], x[1]))]


def pick(rows: list, k: int | None = None, thr: float | None = None,
         min_k: int = 0, max_k: int = 10**9, spread: bool = False) -> list[dict]:
    """Chọn chunk theo thứ tự logit: k đầu (k khác None) hoặc logit > thr kẹp [min_k, max_k]; bỏ trùng nguyên văn.

    spread=True: rải theo doc (diversify) trước khi cắt k; với thr thì n đếm như cũ, chỉ đổi chunk nào được giữ.
    """
    seen, uniq = set(), []
    for r in rows:
        if r[4] in seen:
            continue
        seen.add(r[4])
        uniq.append(r)
    if spread:
        n_thr = sum(1 for r in uniq if thr is not None and r[3] > thr)
        uniq = diversify(uniq)
        if k is None:
            n = min(max(n_thr, min_k), max_k)
            return [{"doc_id": r[0], "chunk_text": r[4]} for r in uniq[:n]]
    if k is not None:
        n = k
    else:
        n = sum(1 for r in uniq if r[3] > thr)
        n = min(max(n, min_k), max_k)
    return [{"doc_id": r[0], "chunk_text": r[4]} for r in uniq[:n]]


def build(rr: dict, **kw) -> list[dict]:
    return [{"id": int(q), "relevant_docs": [], "relevant_chunks": pick(rows, **kw)} for q, rows in rr.items()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--min-k", type=int, default=0)
    ap.add_argument("--max-k", type=int, default=10**9)
    ap.add_argument("--spread", action="store_true", help="rải chunk theo doc: chunk tốt nhất mỗi doc trước")
    a = ap.parse_args()
    if (a.k is None) == (a.thr is None):
        ap.error("cần đúng một trong --k hoặc --thr")
    out = build(load_rerank(a.rerank), k=a.k, thr=a.thr, min_k=a.min_k, max_k=a.max_k, spread=a.spread)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    nc = [len(s["relevant_chunks"]) for s in out]
    print(f"ghi {a.out}: {len(out)} câu, chunk/câu min {min(nc)} TB {sum(nc) / len(nc):.2f} max {max(nc)}", flush=True)


if __name__ == "__main__":
    main()
