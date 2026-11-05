"""Run a resumable Qwen3-14B QuerySpec pilot or full production batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.openrouter_client import OpenRouterClient, OpenRouterConfig  # noqa: E402
from src.llm.query_spec import (  # noqa: E402
    build_query_spec_messages,
    build_repair_messages,
    query_spec_json_schema,
    select_stratified_pilot,
    validate_query_spec_for_query,
)


DEFAULT_CONFIG = ROOT / "configs/query_spec_v1.yaml"
DEFAULT_QUERIES = ROOT / "data/raw/btc_provided/query.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data/audits/queryspec_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, help="Override pilot size; use 1 for a smoke test")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all official queries in ID order instead of selecting a stratified pilot",
    )
    parser.add_argument("--concurrency", type=int, help="Override worker count")
    parser.add_argument(
        "--query-ids",
        help="Comma-separated query IDs for a targeted semantic regression run",
    )
    parser.add_argument("--force", action="store_true", help="Ignore matching cached responses")
    parser.add_argument("--dry-run", action="store_true", help="Select and validate inputs without API calls")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or "openrouter" not in value or "system_prompt" not in value:
        raise ValueError(f"Invalid QuerySpec config: {path}")
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def summarize(records: list[dict[str, Any]], pilot: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    statuses = pd.Series([record["status"] for record in records]).value_counts().to_dict()
    latencies = [record["provenance"].get("latency_ms") or 0 for record in records]
    tokens = [record["provenance"].get("total_tokens") or 0 for record in records]
    return {
        "version": config["version"],
        "selected_queries": len(pilot),
        "completed_records": len(records),
        "statuses": statuses,
        "usable_queryspec_rate": statuses.get("ok", 0) / max(len(records), 1),
        "safe_fallback_rate": statuses.get("fallback_exp122", 0) / max(len(records), 1),
        "cached": sum(bool(record["provenance"].get("cached")) for record in records),
        "latency_ms_total": sum(latencies),
        "tokens_total": sum(tokens),
        "strata": pilot["stratum"].value_counts().sort_index().to_dict(),
        "model": config["openrouter"]["model"],
        "providers": config["openrouter"]["providers"],
        "prompt_version": config["openrouter"]["prompt_version"],
        "schema_version": config["openrouter"]["schema_version"],
    }


def export_parquet_outputs(output_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    spec_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    for record in records:
        spec = record.get("parsed") or {}
        provenance = record.get("provenance") or {}
        cardinality = spec.get("expected_evidence_cardinality") or {}
        atoms = spec.get("evidence_atoms") or []
        spec_rows.append(
            {
                "query_id": int(record["query_id"]),
                "query": record.get("query", ""),
                "status": record.get("status", ""),
                "fallback_reason": record.get("fallback_reason", ""),
                "query_type": spec.get("query_type", ""),
                "canonical_vi": spec.get("canonical_vi", ""),
                "query_en": spec.get("query_en", ""),
                "query_zh": spec.get("query_zh", ""),
                "atom_count": len(atoms),
                "needs_multihop": spec.get("needs_multihop"),
                "expected_min_docs": cardinality.get("min_docs"),
                "expected_max_docs": cardinality.get("max_docs"),
                "confidence": spec.get("confidence"),
                "entities_json": json.dumps(spec.get("entities") or {}, ensure_ascii=False),
                "subject_facts_json": json.dumps(spec.get("subject_facts") or [], ensure_ascii=False),
                "temporal_constraints_json": json.dumps(
                    spec.get("temporal_constraints") or [], ensure_ascii=False
                ),
                "negations_json": json.dumps(spec.get("negations") or [], ensure_ascii=False),
                "global_exclusions_json": json.dumps(spec.get("exclusions") or [], ensure_ascii=False),
                "synonym_pairs_json": json.dumps(spec.get("synonym_pairs") or [], ensure_ascii=False),
                "request_hash": record.get("request_hash", ""),
                "model": provenance.get("model_returned", ""),
                "provider": provenance.get("provider_actual", ""),
                "prompt_version": provenance.get("prompt_version", ""),
                "schema_version": provenance.get("schema_version", ""),
            }
        )
        if record.get("status") != "ok":
            continue
        for atom_index, atom in enumerate(atoms, 1):
            atom_rows.append(
                {
                    "query_id": int(record["query_id"]),
                    "atom_index": atom_index,
                    "atom_id": atom.get("atom_id", ""),
                    "intent": atom.get("intent", ""),
                    "retrieval_query_vi": atom.get("retrieval_query_vi", ""),
                    "retrieval_query_en": atom.get("retrieval_query_en", ""),
                    "retrieval_query_zh": atom.get("retrieval_query_zh", ""),
                    "must_match_json": json.dumps(atom.get("must_match") or [], ensure_ascii=False),
                    "optional_terms_json": json.dumps(
                        atom.get("optional_terms") or [], ensure_ascii=False
                    ),
                    "exclusions_json": json.dumps(atom.get("exclusions") or [], ensure_ascii=False),
                }
            )
    specs_path = output_dir / "query_specs.parquet"
    atoms_path = output_dir / "evidence_atoms.parquet"
    pd.DataFrame(spec_rows).sort_values("query_id").to_parquet(
        specs_path, index=False, compression="snappy"
    )
    pd.DataFrame(atom_rows).sort_values(["query_id", "atom_index"]).to_parquet(
        atoms_path, index=False, compression="snappy"
    )
    return specs_path, atoms_path


def run_one(
    client: OpenRouterClient,
    query_id: int,
    query: str,
    system_prompt: str,
    force: bool,
) -> dict[str, Any]:
    validator = lambda value: validate_query_spec_for_query(
        value, expected_query_id=query_id, original_query=query
    )
    result = client.complete_json(
        query_id=query_id,
        messages=build_query_spec_messages(query_id, query, system_prompt),
        json_schema=query_spec_json_schema(),
        validator=validator,
        force=force,
    )
    repaired = False
    unsafe_fallback = any("unsafe multi-subject negation" in error for error in result.validation_errors)
    if result.status == "invalid" and not unsafe_fallback:
        result = client.complete_json(
            query_id=f"{query_id}:repair",
            messages=build_repair_messages(
                query_id, query, system_prompt, result.raw_content, result.validation_errors
            ),
            json_schema=query_spec_json_schema(),
            validator=validator,
            force=force,
        )
        repaired = result.status == "ok"
    record = result.to_record()
    record["query_id"] = query_id
    record["query"] = query
    record["repaired"] = repaired
    record["provenance"]["cached"] = result.cached
    if result.status == "invalid":
        record["status"] = "fallback_exp122"
        record["fallback_reason"] = "semantic_validation"
    elif result.status != "ok":
        record["status"] = "needs_review"
        record["fallback_reason"] = "api_or_runner_error"
    else:
        record["fallback_reason"] = ""
    return record


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env", override=False)
    config = load_config(args.config)
    pilot_config = config.get("pilot", {})
    if args.all and (args.limit is not None or args.query_ids):
        raise ValueError("--all cannot be combined with --limit or --query-ids")
    limit = len(pd.read_parquet(args.queries, columns=["id"])) if args.all else (
        args.limit or int(pilot_config.get("size", 120))
    )
    concurrency = args.concurrency or int(pilot_config.get("concurrency", 4))
    checkpoint_every = int(pilot_config.get("checkpoint_every", 10))
    if limit < 1 or concurrency < 1:
        raise ValueError("limit and concurrency must be positive")

    queries = pd.read_parquet(args.queries, columns=["id", "query"])
    if args.all:
        pilot = select_stratified_pilot(queries, n=len(queries)).sort_values("id").reset_index(drop=True)
    elif args.query_ids:
        requested_ids = {int(value.strip()) for value in args.query_ids.split(",") if value.strip()}
        missing_ids = requested_ids - set(queries["id"].astype(int))
        if missing_ids:
            raise ValueError(f"Unknown query IDs: {sorted(missing_ids)}")
        pilot = queries[queries["id"].astype(int).isin(requested_ids)].copy()
        feature_frame = select_stratified_pilot(pilot, n=len(pilot))
        pilot = feature_frame.sort_values("id").reset_index(drop=True)
    else:
        pilot = select_stratified_pilot(queries, n=limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "pilot_manifest.parquet"
    pilot.to_parquet(manifest_path, index=False)
    print(f"Selected {len(pilot)} queries across {pilot['stratum'].nunique()} strata -> {manifest_path}")
    if args.dry_run:
        print("Dry run complete; no API request was sent.")
        return
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing from environment/.env")

    client_config = OpenRouterConfig.from_mapping(config["openrouter"], root=ROOT)
    client = OpenRouterClient(client_config)
    output_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.json"
    records_by_id: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                run_one,
                client,
                int(row.id),
                str(row.query),
                str(config["system_prompt"]),
                args.force,
            ): int(row.id)
            for row in pilot.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            query_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # keep the batch resumable even on an unexpected item failure
                record = {
                    "query_id": query_id,
                    "query": str(pilot.loc[pilot.id.eq(query_id), "query"].iloc[0]),
                    "status": "needs_review",
                    "parsed": None,
                    "raw_content": "",
                    "validation_errors": [f"runner_error: {type(exc).__name__}: {exc}"],
                    "repaired": False,
                    "provenance": {"error": f"{type(exc).__name__}: {exc}", "cached": False},
                }
            records_by_id[query_id] = record
            if completed % checkpoint_every == 0 or completed == len(futures):
                records = [records_by_id[key] for key in sorted(records_by_id)]
                write_jsonl_atomic(output_path, records)
                write_json_atomic(summary_path, summarize(records, pilot, config))
            print(
                f"{completed}/{len(futures)} query_id={query_id} status={record['status']} "
                f"cached={record['provenance'].get('cached', False)}",
                flush=True,
            )

    summary = summarize([records_by_id[key] for key in sorted(records_by_id)], pilot, config)
    records = [records_by_id[key] for key in sorted(records_by_id)]
    specs_path, atoms_path = export_parquet_outputs(args.output_dir, records)
    summary["query_specs_path"] = str(specs_path)
    summary["evidence_atoms_path"] = str(atoms_path)
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
