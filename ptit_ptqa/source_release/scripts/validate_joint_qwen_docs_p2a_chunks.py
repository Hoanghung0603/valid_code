"""Independent streaming validator for the joint Qwen-doc/P2a-chunk artifact."""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

try:
    from build_p1_verified_docs_submission import load_scores
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.build_p1_verified_docs_submission import load_scores
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def validate(base: Path, filtered: Path, scores_path: Path, candidate: Path, keep_top: int) -> dict:
    if candidate.stat().st_size >= 100_000_000:
        raise ValueError("Candidate exceeds decimal 100 MB")
    scores, score_hash = load_scores(scores_path)
    if scores.duplicated(["query_id", "doc_id"]).any() or scores.qwen_score.isna().any():
        raise ValueError("Invalid Qwen scores")
    if not np.isfinite(scores.qwen_score.to_numpy(float)).all():
        raise ValueError("Non-finite Qwen scores")
    ranked = {
        int(qid): group.sort_values(
            ["qwen_score", "doc_id"], ascending=[False, True], kind="mergesort"
        ).doc_id.astype(int).tolist()
        for qid, group in scores.groupby("query_id", sort=False)
    }

    stats = {"queries": 0, "docs_before": 0, "docs_after": 0,
             "chunks_before": 0, "chunks_after": 0, "external_owner_query_docs": 0}
    with zipfile.ZipFile(base) as bz, zipfile.ZipFile(filtered) as fz, zipfile.ZipFile(candidate) as cz:
        for archive in (bz, fz, cz):
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise ValueError("ZIP layout/CRC failure")
        with (
            bz.open("submission.json") as bb, fz.open("submission.json") as fb,
            cz.open("submission.json") as cb,
            io.TextIOWrapper(bb, encoding="utf-8") as bt,
            io.TextIOWrapper(fb, encoding="utf-8") as ft,
            io.TextIOWrapper(cb, encoding="utf-8") as ct,
        ):
            for base_row, filtered_row, candidate_row in zip(
                iter_json_array(bt), iter_json_array(ft), iter_json_array(ct), strict=True
            ):
                qid = int(base_row["id"])
                if qid != int(filtered_row["id"]) or qid != int(candidate_row["id"]):
                    raise ValueError("Query IDs differ or are reordered")
                if set(candidate_row) != {"id", "relevant_docs", "relevant_chunks"}:
                    raise ValueError("Unexpected candidate row keys")
                if candidate_row["relevant_chunks"] != filtered_row["relevant_chunks"]:
                    raise ValueError(f"Candidate chunks differ from P2a for query {qid}")
                base_docs = [int(value) for value in base_row["relevant_docs"]]
                if len(base_docs) != 216 or len(set(base_docs)) != 216:
                    raise ValueError(f"Invalid base docs for query {qid}")
                if qid not in ranked or len(ranked[qid]) != 216 or set(ranked[qid]) != set(base_docs):
                    raise ValueError(f"Qwen score coverage differs for query {qid}")
                owners_all = {int(chunk["doc_id"]) for chunk in filtered_row["relevant_chunks"]}
                owners = owners_all & set(base_docs)
                keep = set(ranked[qid][:keep_top]) | owners
                expected_docs = [doc_id for doc_id in base_docs if doc_id in keep]
                docs = [int(value) for value in candidate_row["relevant_docs"]]
                if docs != expected_docs or len(docs) != len(set(docs)):
                    raise ValueError(f"Candidate docs differ from exact joint policy for query {qid}")
                stats["queries"] += 1
                stats["docs_before"] += len(base_docs)
                stats["docs_after"] += len(docs)
                stats["chunks_before"] += len(base_row["relevant_chunks"])
                stats["chunks_after"] += len(candidate_row["relevant_chunks"])
                stats["external_owner_query_docs"] += len(owners_all - set(base_docs))
    if stats["queries"] != 1200 or len(ranked) != 1200:
        raise ValueError("Expected exact 1,200-query coverage")
    return {
        "schema_version": "joint_qwen_docs_p2a_chunks_validation_v1",
        **stats,
        "docs_removed": stats["docs_before"] - stats["docs_after"],
        "chunks_removed": stats["chunks_before"] - stats["chunks_after"],
        "candidate_zip_bytes": candidate.stat().st_size,
        "candidate_zip_sha256": sha256_file(candidate),
        "base_zip_sha256": sha256_file(base),
        "filtered_zip_sha256": sha256_file(filtered),
        "score_payload_sha256": score_hash,
        "exact_joint_policy": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--keep-top", type=int, default=130)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.base, args.filtered, args.scores, args.candidate, args.keep_top)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
