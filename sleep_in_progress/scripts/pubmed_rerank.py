"""Wrapper rerank PubMed trên z87: nới --docs của rerank_run.py để nhận cả thư mục (glob *.jsonl.zst).

Còn lại dùng nguyên rerank_run.py (bge-reranker-v2-m3, resume theo phần 50 câu):
  python scripts/pubmed_rerank.py --docs D:\\r2ai\\work\\pm\\docs --queries ... --cands ... --out ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerank_run  # noqa: E402


def expand(args: list) -> list:
    out = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            out.extend(str(f) for f in sorted(p.glob("*.jsonl.zst")))
        else:
            out.append(a)
    return out


def main() -> None:
    argv = sys.argv[1:]
    if "--docs" in argv:
        i = argv.index("--docs")
        j = i + 1
        rest = []
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        docs = expand(argv[i + 1:j])
        if not docs:
            raise SystemExit("không thấy file docs nào")
        print(f"{len(docs)} file docs", flush=True)
        argv = argv[: i + 1] + docs + argv[j:]
    sys.argv = [sys.argv[0]] + argv
    rerank_run.main()


if __name__ == "__main__":
    main()
