"""Merge independently audited Qwen4B chunk-score partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from audit_qwen4b_chunk_scores import audit, sha256_file
except ModuleNotFoundError:
    from scripts.audit_qwen4b_chunk_scores import audit, sha256_file


def merge(input_path: Path, score_paths: list[Path], output_dir: Path) -> dict:
    if len(score_paths) < 2:
        raise ValueError("Expected at least two score partitions")
    frames = [pd.read_parquet(path) for path in score_paths]
    query_sets = [set(frame.query_id.astype(int)) for frame in frames]
    for left in range(len(query_sets)):
        for right in range(left + 1, len(query_sets)):
            overlap = query_sets[left] & query_sets[right]
            if overlap:
                raise RuntimeError(f"Query overlap between parts {left} and {right}: {len(overlap)}")
    merged = pd.concat(frames, ignore_index=True)
    if merged.pair_id.duplicated().any():
        raise RuntimeError("Duplicate pair_id after merge")
    merged = merged.sort_values(["query_id", "chunk_ordinal"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    score_output = output_dir / "qwen4b_chunk_scores_full.parquet"
    audit_output = output_dir / "audit.json"
    manifest_output = output_dir / "manifest.json"
    merged.to_parquet(score_output, index=False, compression="zstd")
    report = audit(input_path, score_output, audit_output)
    manifest = {
        "schema_version": "qwen4b_original_chunk_scores_merged_v1",
        "input_sha256": sha256_file(input_path),
        "parts": [
            {"path": str(path), "sha256": sha256_file(path), "rows": len(frame),
             "queries": int(frame.query_id.nunique())}
            for path, frame in zip(score_paths, frames, strict=True)
        ],
        "output": {"path": str(score_output), "sha256": sha256_file(score_output)},
        "rows": int(len(merged)),
        "queries": int(merged.query_id.nunique()),
        "statuses": {str(k): int(v) for k, v in merged.status.value_counts().items()},
        "audit_passed": bool(report["passed"]),
    }
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input, args.scores, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
