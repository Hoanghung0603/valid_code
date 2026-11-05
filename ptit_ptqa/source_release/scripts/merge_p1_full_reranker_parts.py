"""Merge and verify query-disjoint P1 full-reranker Kaggle output parts."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from generate_translation_submission import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FULL_PAIRS_SHA256 = "732e313ece7c2f46f21ae543a2b831c044c9925726c6b9a111518352a02cea28"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=Path, nargs="+")
    parser.add_argument("--model", choices=("bge", "qwen4b"), required=True)
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_part(path: Path) -> tuple[pd.DataFrame, dict, dict]:
    with zipfile.ZipFile(path) as archive:
        required = {"p1_doc_scores.parquet", "validation.json", "run_manifest.json"}
        if archive.testzip() is not None or not required.issubset(archive.namelist()):
            raise RuntimeError(f"Invalid P1 full part: {path}")
        docs = pd.read_parquet(io.BytesIO(archive.read("p1_doc_scores.parquet")))
        validation = json.loads(archive.read("validation.json"))
        manifest = json.loads(archive.read("run_manifest.json"))
    return docs, validation, manifest


def validate_merged_docs(
    docs: pd.DataFrame,
    model: str,
    expected_queries: int = 1200,
    docs_per_query: int = 216,
) -> None:
    if docs.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Full part outputs overlap on query-doc rows")
    counts = docs.groupby("query_id").size()
    expected_rows = expected_queries * docs_per_query
    if (
        docs.query_id.nunique() != expected_queries
        or len(docs) != expected_rows
        or counts.min() != docs_per_query
        or counts.max() != docs_per_query
    ):
        raise RuntimeError({"queries": docs.query_id.nunique(), "query_docs": len(docs), "min": counts.min(), "max": counts.max()})
    score_column = f"{model}_score"
    rank_column = f"{model}_rank"
    if not {score_column, rank_column}.issubset(docs.columns):
        raise RuntimeError("Merged full scores miss model score/rank columns")
    if not np.isfinite(docs[score_column].to_numpy(dtype=float)).all():
        raise RuntimeError("Merged full scores contain non-finite values")
    expected_ranks = set(range(1, docs_per_query + 1))
    bad_rank_queries = sum(
        set(group[rank_column].astype(int)) != expected_ranks
        for _, group in docs.groupby("query_id", sort=False)
    )
    if bad_rank_queries:
        raise RuntimeError(f"{bad_rank_queries} queries do not contain each rank 1..{docs_per_query} exactly once")


def main() -> None:
    args = parse_args()
    if len(args.parts) != args.part_count:
        raise ValueError(f"Expected {args.part_count} part ZIPs, received {len(args.parts)}")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {args.output}")

    frames = []
    validations = []
    manifests = []
    part_hashes = {}
    for path in args.parts:
        docs, validation, manifest = read_part(path)
        if validation["model_key"] != args.model or int(validation["part_count"]) != args.part_count:
            raise RuntimeError(f"Model/part-count mismatch in {path}")
        if validation["full_input_pairs_sha256"] != FULL_PAIRS_SHA256:
            raise RuntimeError(f"Full input hash mismatch in {path}")
        frames.append(docs)
        validations.append(validation)
        manifests.append(manifest)
        part_hashes[str(path.resolve())] = sha256_file(path)

    indices = [int(item["part_index"]) for item in validations]
    if sorted(indices) != list(range(args.part_count)):
        raise RuntimeError(f"Parts are missing or duplicated: {indices}")
    revisions = {str(item["model_revision"]) for item in validations}
    if len(revisions) != 1:
        raise RuntimeError(f"Model revisions differ across parts: {revisions}")

    docs = pd.concat(frames, ignore_index=True)
    validate_merged_docs(docs, args.model)
    expected_pairs = sum(int(item["pairs"]) for item in validations)
    if expected_pairs != 406_014:
        raise RuntimeError(f"Partition pair count mismatch: {expected_pairs}")

    docs = docs.sort_values(["query_id", "doc_rank"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output.with_suffix(".parquet")
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    docs.to_parquet(parquet_path, index=False, compression="zstd")
    manifest = {
        "schema_version": "p1_full_reranker_merged_v1",
        "model_key": args.model,
        "model_revision": next(iter(revisions)),
        "part_count": args.part_count,
        "queries": 1200,
        "query_docs": 259_200,
        "pairs_scored": expected_pairs,
        "full_input_pairs_sha256": FULL_PAIRS_SHA256,
        "source_part_zip_sha256": part_hashes,
        "p1_doc_scores_sha256": sha256_file(parquet_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(parquet_path, arcname="p1_doc_scores.parquet")
        archive.write(manifest_path, arcname="run_manifest.json")
    with zipfile.ZipFile(args.output) as archive:
        if archive.namelist() != ["p1_doc_scores.parquet", "run_manifest.json"] or archive.testzip() is not None:
            raise RuntimeError("Merged P1 ZIP failed member/CRC validation")
    print(json.dumps({**manifest, "output_zip": str(args.output), "output_zip_sha256": sha256_file(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
