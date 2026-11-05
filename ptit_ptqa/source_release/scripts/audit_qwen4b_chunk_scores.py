"""Audit Qwen4B exact-chunk score lineage and coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(
    input_path: Path,
    score_path: Path,
    output_path: Path,
    part_index: int = 0,
    part_count: int = 1,
) -> dict:
    source = pd.read_parquet(input_path)
    if part_count < 1 or not 0 <= part_index < part_count:
        raise ValueError("Invalid partition")
    if part_count > 1:
        query_ids = sorted(source.query_id.astype(int).unique())
        owned = {query_id for index, query_id in enumerate(query_ids) if index % part_count == part_index}
        source = source[source.query_id.isin(owned)].copy().reset_index(drop=True)
    scores = pd.read_parquet(score_path)
    required = {
        "pair_id", "query_id", "chunk_ordinal", "doc_id", "raw_score",
        "normalized_score", "input_tokens", "status", "model_revision",
        "query_sha256", "passage_sha256",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise RuntimeError(f"Missing score columns: {missing}")
    if source.pair_id.duplicated().any() or scores.pair_id.duplicated().any():
        raise RuntimeError("Duplicate pair_id")
    if set(source.pair_id) != set(scores.pair_id):
        raise RuntimeError("Score coverage does not equal source coverage")

    keys = ["query_id", "chunk_ordinal", "doc_id", "query_sha256", "passage_sha256"]
    joined = source[["pair_id", *keys]].merge(
        scores[["pair_id", *keys]], on="pair_id", suffixes=("_source", "_score"),
        validate="one_to_one",
    )
    mismatches = {
        key: int((joined[f"{key}_source"] != joined[f"{key}_score"]).sum())
        for key in keys
    }
    if any(mismatches.values()):
        raise RuntimeError(f"Lineage mismatch: {mismatches}")

    statuses = scores.status.value_counts(dropna=False).to_dict()
    if set(statuses) - {"ok", "too_long"}:
        raise RuntimeError(f"Unexpected statuses: {statuses}")
    ok = scores.status.eq("ok")
    too_long = scores.status.eq("too_long")
    if not np.isfinite(scores.loc[ok, ["raw_score", "normalized_score"]].to_numpy()).all():
        raise RuntimeError("Non-finite successful score")
    if scores.loc[ok, "input_tokens"].gt(4096).any():
        raise RuntimeError("Successful row exceeds token limit")
    if scores.loc[too_long, "input_tokens"].le(4096).any():
        raise RuntimeError("too_long row does not exceed token limit")
    if scores.loc[too_long, ["raw_score", "normalized_score"]].notna().any().any():
        raise RuntimeError("too_long row unexpectedly has a score")
    if set(scores.model_revision.astype(str)) != {MODEL_REVISION}:
        raise RuntimeError("Model revision mismatch")

    normalized = scores.loc[ok, "normalized_score"]
    report = {
        "schema_version": "qwen4b_original_chunk_score_audit_v1",
        "passed": True,
        "part_index": part_index,
        "part_count": part_count,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "scores": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "rows": int(len(scores)),
        "queries": int(scores.query_id.nunique()),
        "statuses": {str(k): int(v) for k, v in statuses.items()},
        "too_long_fraction": float(too_long.mean()),
        "lineage_mismatches": mismatches,
        "model_revision": MODEL_REVISION,
        "input_tokens": {
            "median_ok": float(scores.loc[ok, "input_tokens"].median()),
            "p90_ok": float(scores.loc[ok, "input_tokens"].quantile(0.9)),
            "max_ok": int(scores.loc[ok, "input_tokens"].max()),
            "min_too_long": int(scores.loc[too_long, "input_tokens"].min()),
            "max_too_long": int(scores.loc[too_long, "input_tokens"].max()),
        },
        "normalized_score": {
            "min": float(normalized.min()),
            "p10": float(normalized.quantile(0.1)),
            "median": float(normalized.median()),
            "p90": float(normalized.quantile(0.9)),
            "max": float(normalized.max()),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--part-count", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.input, args.scores, args.output,
        part_index=args.part_index, part_count=args.part_count,
    ), indent=2))


if __name__ == "__main__":
    main()
