"""Verify BGE-M3 ColBERT Kaggle output against its exact input shortlist.

This is an integrity and ranking-diversity audit, not a relevance evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(input_path: Path, output_path: Path) -> dict:
    with zipfile.ZipFile(output_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Output ZIP has a corrupt member")
        if set(archive.namelist()) != {"scores.parquet", "run_manifest.json"}:
            raise ValueError("Unexpected output ZIP members")
        manifest = json.loads(archive.read("run_manifest.json"))
        raw_scores = archive.read("scores.parquet")
    if manifest.get("status") != "complete":
        raise ValueError("Kaggle run is not marked complete")
    if sha256_file(input_path) != manifest.get("input_sha256"):
        raise ValueError("Input SHA256 mismatch")
    if hashlib.sha256(raw_scores).hexdigest() != manifest.get("scores_sha256"):
        raise ValueError("Score SHA256 mismatch")

    inputs = pd.read_parquet(input_path, columns=[
        "pair_id", "query_id", "doc_id", "doc_pick_rank", "passage_rank"
    ])
    scores = pd.read_parquet(io.BytesIO(raw_scores))
    required = {"pair_id", "query_id", "doc_id", "doc_pick_rank", "passage_rank", "colbert_score"}
    if set(scores.columns) != required:
        raise ValueError(f"Unexpected score schema: {list(scores.columns)}")
    if inputs.pair_id.duplicated().any() or scores.pair_id.duplicated().any():
        raise ValueError("Duplicate pair_id")
    if len(scores) != len(inputs) or len(scores) != int(manifest["pairs"]):
        raise ValueError("Pair count mismatch")
    if scores.query_id.nunique() != int(manifest["queries"]):
        raise ValueError("Query count mismatch")
    keys = ["pair_id", "query_id", "doc_id", "doc_pick_rank", "passage_rank"]
    checked = scores.merge(inputs, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not checked._merge.eq("both").all():
        raise ValueError("Score rows differ from input pair identity/metadata")
    if not np.isfinite(scores.colbert_score.to_numpy()).all():
        raise ValueError("Non-finite score")

    docs = scores.groupby(["query_id", "doc_id", "doc_pick_rank"], as_index=False).colbert_score.max()
    agreements = []
    for _, group in docs.groupby("query_id"):
        before = set(group.nsmallest(10, "doc_pick_rank").doc_id)
        after = set(group.nlargest(10, "colbert_score").doc_id)
        agreements.append(len(before & after))
    return {
        "status": "pass",
        "pairs": len(scores),
        "queries": int(scores.query_id.nunique()),
        "query_docs": len(docs),
        "model": manifest["model"],
        "revision": manifest["revision"],
        "seconds_this_session": manifest["seconds_this_session"],
        "pairs_per_second_this_session": manifest["pairs_per_second_this_session"],
        "score_min": float(scores.colbert_score.min()),
        "score_median": float(scores.colbert_score.median()),
        "score_max": float(scores.colbert_score.max()),
        "top10_overlap_mean": float(np.mean(agreements)),
        "top10_overlap_min": int(min(agreements)),
        "btc_score": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
