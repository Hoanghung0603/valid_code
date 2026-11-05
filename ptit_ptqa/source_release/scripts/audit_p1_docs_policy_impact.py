"""Measure the structural impact of a P1 docs-only cutoff, without relevance claims.

This is a preflight audit, not a calibration or submission builder. It reports
how many Exp122 documents would be removed after protecting chunk-owning docs.
It does not use BTC qrels, weak labels, or the synthetic local dev set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_p1_verified_docs_submission import CONTROL, load_control, load_scores, retained_docs
from generate_translation_submission import sha256_file


RANK_BUCKETS = ((1, 50), (51, 100), (101, 150), (151, 200), (201, 216))


def audit_impact(control: list[dict], scores: pd.DataFrame, model: str, keep_tops: list[int]) -> dict:
    score_col = "bge_score" if model == "bge" else "qwen_score"
    required = {"query_id", "doc_id", score_col}
    if not required.issubset(scores.columns):
        raise ValueError(f"Missing score columns: {sorted(required - set(scores.columns))}")
    if scores[["query_id", "doc_id", score_col]].isna().any().any():
        raise ValueError("Null score or identifier")
    if not np.isfinite(scores[score_col].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite score")
    if scores.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Duplicate query-doc scores")
    if len(set(keep_tops)) != len(keep_tops) or any(not 1 <= k <= 215 for k in keep_tops):
        raise ValueError("Keep-top values must be unique integers in [1, 215]")

    control_ids = {int(row["id"]) for row in control}
    by_query = {int(qid): group for qid, group in scores.groupby("query_id", sort=False)}
    if len(control_ids) != len(control) or set(by_query) != control_ids:
        raise ValueError("Query IDs differ between scores and control")

    summaries = {k: {"changed_queries": 0, "removed_query_docs": 0,
                     "owner_docs_beyond_cutoff_retained": 0,
                     "removed_control_rank_buckets": {f"{lo}-{hi}": 0 for lo, hi in RANK_BUCKETS},
                     "per_query_removed": []} for k in keep_tops}
    total_docs = 0
    for row in control:
        qid = int(row["id"])
        docs = [int(value) for value in row["relevant_docs"]]
        owners = {int(chunk["doc_id"]) for chunk in row["relevant_chunks"]} & set(docs)
        if len(docs) != 216 or len(set(docs)) != 216:
            raise ValueError(f"Invalid control docs for query {qid}")
        group = by_query[qid]
        if len(group) != 216 or set(group.doc_id.astype(int)) != set(docs):
            raise ValueError(f"Score doc coverage differs for query {qid}")
        ranked = group.sort_values([score_col, "doc_id"], ascending=[False, True]).doc_id.astype(int).tolist()
        control_rank = {doc_id: rank for rank, doc_id in enumerate(docs, 1)}
        total_docs += len(docs)
        for k in keep_tops:
            kept = set(retained_docs(docs, owners, ranked, k))
            removed = set(docs) - kept
            if owners & removed:
                raise AssertionError("Chunk-owning document was removed")
            summary = summaries[k]
            summary["changed_queries"] += bool(removed)
            summary["removed_query_docs"] += len(removed)
            summary["owner_docs_beyond_cutoff_retained"] += len(owners - set(ranked[:k]))
            summary["per_query_removed"].append(len(removed))
            for doc_id in removed:
                rank = control_rank[doc_id]
                bucket = next(f"{lo}-{hi}" for lo, hi in RANK_BUCKETS if lo <= rank <= hi)
                summary["removed_control_rank_buckets"][bucket] += 1

    result = {"schema_version": "p1_docs_policy_impact_v1", "interpretation":
              "Structural cutoff impact only; no relevance labels or BTC score prediction.",
              "model": model, "queries": len(control), "control_query_docs": total_docs,
              "cutoffs": {}}
    for k, summary in summaries.items():
        removals = summary.pop("per_query_removed")
        result["cutoffs"][str(k)] = {**summary,
            "remaining_query_docs": total_docs - summary["removed_query_docs"],
            "removed_fraction": summary["removed_query_docs"] / total_docs,
            "removed_per_query_median": float(np.median(removals)),
            "removed_per_query_max": max(removals)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--model", choices=("bge", "qwen4b"), required=True)
    parser.add_argument("--keep-top", type=int, nargs="+", required=True)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    control = load_control(args.control)
    scores, score_payload_hash = load_scores(args.scores)
    result = audit_impact(control, scores, args.model, args.keep_top)
    result["control_sha256"] = sha256_file(args.control)
    result["score_payload_sha256"] = score_payload_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
