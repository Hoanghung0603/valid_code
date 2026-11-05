"""Route Exp89 candidate passages to ICD-10 chapters with a resumable cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.icd10_router import ICD10ChapterRouter, ICDEntry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/exp89_qwen_kaggle/exp89_qwen_input.parquet")
    parser.add_argument("--entries", type=Path, default=ROOT / "data/reference/icd10/entries_en.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reference/icd10/exp89_passage_routes.parquet")
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    entries = pd.read_parquet(args.entries)
    router = ICD10ChapterRouter([
        ICDEntry(str(row.code), str(row.chapter), str(row.text_en))
        for row in entries.itertuples(index=False)
    ])
    passages = pd.read_parquet(args.input, columns=["query_id", "doc_id", "stage_rank", "passage_text"])
    if len(passages) != 300_000:
        raise RuntimeError("Expected exactly 300,000 candidate passages")
    records: list[dict] = []
    t0 = time.monotonic()
    for start in range(0, len(passages), args.batch_size):
        block = passages.iloc[start : start + args.batch_size]
        for row in block.itertuples(index=False):
            result = router.route(str(row.passage_text), str(row.passage_text), top_k=3)
            item = {
                "query_id": int(row.query_id),
                "doc_id": int(row.doc_id),
                "stage_rank": int(row.stage_rank),
                "confidence": result.confidence,
                "unknown": result.unknown,
                "margin": result.margin,
                "matched_terms": result.matched_terms,
            }
            for rank in range(3):
                route = result.routes[rank] if rank < len(result.routes) else None
                item[f"chapter_{rank + 1}"] = route.chapter if route else None
                item[f"score_{rank + 1}"] = route.score if route else 0.0
            records.append(item)
        elapsed = time.monotonic() - t0
        print(json.dumps({
            "complete": len(records),
            "total": len(passages),
            "rows_per_second": round(len(records) / max(elapsed, 0.001), 1),
            "eta_seconds": round((len(passages) - len(records)) * elapsed / max(len(records), 1), 1),
        }), flush=True)
    result = pd.DataFrame.from_records(records)
    if len(result) != len(passages):
        raise RuntimeError("Route output is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False, compression="zstd")
    summary = {
        "rows": len(result),
        "queries": int(result.query_id.nunique()),
        "confidence": {str(key): int(value) for key, value in result.confidence.value_counts().items()},
        "seconds": round(time.monotonic() - t0, 2),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
