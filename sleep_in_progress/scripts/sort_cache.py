"""Sắp xếp cache điểm {qid: [[doc_id, idx, score], ...]} theo score giảm dần, ghi JSON cùng định dạng.

Mỗi câu giữ nguyên bộ ba [doc_id, idx, score]; score None xuống cuối, đồng điểm giữ thứ tự gốc
(sort ổn định). Dùng cho union_sub/chunks_wholedoc (thứ tự file = thứ tự xếp hạng).

Dùng:
  python scripts/sort_cache.py --cache cache_qwen06.json --out cache_qwen06_sorted.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def sort_rows(rows: list) -> list:
    """Sắp một câu theo score giảm dần; None xuống cuối, đồng điểm giữ thứ tự gốc."""

    def key(i: int):
        sc = rows[i][2]
        return (sc is None, 0.0 if sc is None else -sc)

    return [rows[i] for i in sorted(range(len(rows)), key=key)]


def sort_cache(cache: dict) -> dict:
    """{qid: rows} -> {qid: rows đã sắp}; giữ nguyên thứ tự câu."""
    return {q: sort_rows(rows) for q, rows in cache.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sắp cache điểm theo score giảm dần (None cuối, ổn định)")
    ap.add_argument("--cache", required=True, help="JSON {qid: [[doc_id, idx, score], ...]}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    out = sort_cache(cache)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for v in out.values())
    print(f"ghi {a.out}: {len(out)} câu, {n} dòng", flush=True)


if __name__ == "__main__":
    main()
