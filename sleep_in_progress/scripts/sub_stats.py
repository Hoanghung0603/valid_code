"""Thống kê nhanh bài nộp: TB doc/câu, chunk/câu, số doc KHÁC NHAU có chunk/câu, ký tự/chunk.

Dùng: python scripts/sub_stats.py a.json b.json ...
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def stats(sub: list) -> dict:
    n = max(len(sub), 1)
    nd = sum(len(s["relevant_docs"]) for s in sub) / n
    nc = sum(len(s["relevant_chunks"]) for s in sub) / n
    ndc = sum(len({c["doc_id"] for c in s["relevant_chunks"]}) for s in sub) / n
    chars = [len(c["chunk_text"]) for s in sub for c in s["relevant_chunks"]]
    return {"doc/câu": round(nd, 1), "chunk/câu": round(nc, 1), "doc có chunk/câu": round(ndc, 1),
            "ký tự/chunk TB": round(sum(chars) / max(len(chars), 1))}


def main() -> None:
    for p in sys.argv[1:]:
        print(Path(p).name, json.dumps(stats(json.loads(Path(p).read_text(encoding="utf-8"))), ensure_ascii=False),
              flush=True)


if __name__ == "__main__":
    main()
