"""Finalize a full QuerySpec table after a resumable API batch stops early.

Successful current records take priority, then audited pilot records. Every
remaining official query receives an explicit original-query fallback row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from run_openrouter_queryspec import export_parquet_outputs, write_json_atomic, write_jsonl_atomic
except ModuleNotFoundError:
    from scripts.run_openrouter_queryspec import export_parquet_outputs, write_json_atomic, write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quality(record: dict) -> int:
    return {"ok": 3, "fallback_exp122": 2, "needs_review": 1}.get(str(record.get("status")), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/query_specs/queryspec_production_v1")
    parser.add_argument("--pilot-dir", type=Path,
                        default=ROOT / "data/audits/queryspec_pilot_v6_final")
    parser.add_argument("--queries", type=Path,
                        default=ROOT / "data/raw/btc_provided/query.parquet")
    args = parser.parse_args()
    current = read_jsonl(args.output_dir / "results.jsonl")
    pilot = read_jsonl(args.pilot_dir / "results.jsonl")
    official = pd.read_parquet(args.queries, columns=["id", "query"]).sort_values("id")
    if len(official) != 1200 or official.id.nunique() != 1200:
        raise RuntimeError("Expected 1,200 official queries")
    chosen: dict[int, dict] = {}
    for record in current + pilot:
        qid = int(record["query_id"])
        if qid not in chosen or quality(record) > quality(chosen[qid]):
            chosen[qid] = record
    records: list[dict] = []
    for row in official.itertuples(index=False):
        qid = int(row.id)
        record = chosen.get(qid)
        if record is None or record.get("status") == "needs_review":
            error = "missing_after_interrupted_batch" if record is None else str(
                (record.get("provenance") or {}).get("error") or "api_or_runner_error"
            )
            record = {
                "query_id": qid,
                "query": str(row.query),
                "status": "fallback_original",
                "fallback_reason": error[:500],
                "parsed": None,
                "raw_content": "",
                "validation_errors": record.get("validation_errors", []) if record else [],
                "repaired": False,
                "request_hash": "",
                "provenance": {"cached": False, "error": error[:500]},
            }
        records.append(record)
    final_path = args.output_dir / "results_final.jsonl"
    write_jsonl_atomic(final_path, records)
    specs_path, atoms_path = export_parquet_outputs(args.output_dir, records)
    statuses = pd.Series([record["status"] for record in records]).value_counts().to_dict()
    summary = {
        "schema_version": "queryspec_production_partial_v1",
        "queries": 1200,
        "statuses": statuses,
        "usable_queryspec_rate": statuses.get("ok", 0) / 1200,
        "fallback_rate": 1 - statuses.get("ok", 0) / 1200,
        "source_records": {"interrupted_batch": len(current), "audited_pilot": len(pilot)},
        "stop_reason": "OpenRouter HTTP 402 insufficient credits; no failed response is used as expansion",
        "fallback_policy": "original query plus separate audited production translations downstream",
        "query_specs_path": str(specs_path),
        "evidence_atoms_path": str(atoms_path),
    }
    write_json_atomic(args.output_dir / "summary_final.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
