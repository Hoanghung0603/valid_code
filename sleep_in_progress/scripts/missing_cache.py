"""Lọc top-N mỗi câu của cache xếp hạng, chỉ giữ doc CHƯA có text trong các file docs (JSONL.zst).

Đầu ra {qid: [[doc_id, idx, score], ...]} (câu không thiếu doc nào vẫn có khoá, list rỗng) để đem
export_docs.py --cache xuất phần text còn thiếu. In số doc thiếu duy nhất.

Dùng:
  python scripts/missing_cache.py --cache dense_full_top200.json --n 30 \
      --docs chunkdocs100.part000.jsonl.zst chunkdocs100.part001.jsonl.zst --out cache_dense30_missing.json
"""
import argparse
import json
import sys
from pathlib import Path

from rerank_run import read_docs


def doc_ids(paths) -> set:
    return {str(r["id"]) for p in paths for r in read_docs(p)}


def missing_cache(cache: dict, have: set, n: int = 30) -> dict:
    return {q: [list(h) for h in hits[:n] if str(h[0]) not in have] for q, hits in cache.items()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--docs", required=True, nargs="+")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    have = doc_ids(a.docs)
    out = missing_cache(cache, have, a.n)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    uniq = {str(h[0]) for hits in out.values() for h in hits}
    top = {str(h[0]) for hits in cache.values() for h in hits[: a.n]}
    print(f"{len(have)} doc có text; top-{a.n}: {len(top)} doc duy nhất, thiếu {len(uniq)} doc duy nhất "
          f"({sum(len(v) for v in out.values())} lượt, {sum(1 for v in out.values() if v)} câu); ghi {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
