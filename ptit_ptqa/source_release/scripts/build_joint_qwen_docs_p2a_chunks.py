"""Build one joint trial: Qwen top-K docs plus P2a-kept chunk owners.

The official-score evidence behind the two components is kept explicit:
Qwen top130 improved document F2, while P2a keep90 increased chunk precision but
slightly reduced chunk F2.  This builder does not rescore or retune either
component.  It makes their membership consistent by releasing a document from
owner protection only when P2a removed every chunk that referred to it.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from build_p1_verified_docs_submission import load_scores, package_lzma64
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.build_p1_verified_docs_submission import load_scores, package_lzma64
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submission_exp122_exp121_add19_miss.zip"
FILTERED = ROOT / "submission_p2a_bge_keep90_full.zip"
SCORES = ROOT / "data/audits/p1_full_qwen4b_merged/p1_full_qwen4b_merged.parquet"
OUTPUT = ROOT / "submission_joint_qwen_top130_p2a_keep90.zip"
BASE_SHA256 = "8c7f221cfafa401dfb5435df7da06464999c4b60594e69bc7726bb4ffb379886"
FILTERED_SHA256 = "5c81bb37ed361d198f7f5fca7f72cb3bda7b42fbc1c788429605ab8caab4d673"
SCORES_SHA256 = "bc36af1b525069c43927cf150cd04f3f6c636e692409c8ffcfdf86895ac650a4"


def is_subsequence(items: list[dict], source: list[dict]) -> bool:
    """Return whether items occur byte-semantically in source order."""
    position = 0
    for item in items:
        while position < len(source) and source[position] != item:
            position += 1
        if position == len(source):
            return False
        position += 1
    return True


def stable_joint_docs(base_docs: list[int], ranked_docs: list[int], owners: set[int], keep_top: int) -> list[int]:
    if len(base_docs) != len(set(base_docs)) or set(ranked_docs) != set(base_docs):
        raise ValueError("Ranked docs must exactly cover unique base docs")
    if not 1 <= keep_top < len(base_docs):
        raise ValueError("keep_top must be in [1, len(base_docs)-1]")
    if not owners.issubset(set(base_docs)):
        raise ValueError("Chunk owner lies outside base docs")
    keep = set(ranked_docs[:keep_top]) | owners
    return [doc_id for doc_id in base_docs if doc_id in keep]


def build(base_zip: Path, filtered_zip: Path, scores_path: Path, output: Path, keep_top: int) -> dict:
    manifest_path = output.with_name(output.stem + "_manifest.json")
    json_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, manifest_path, json_path)):
        raise FileExistsError("Output, manifest, or staging JSON already exists")
    if sha256_file(base_zip) != BASE_SHA256:
        raise RuntimeError("Unexpected Exp122 base hash")
    if sha256_file(filtered_zip) != FILTERED_SHA256:
        raise RuntimeError("Unexpected BTC-scored P2a artifact hash")

    scores, score_payload_hash = load_scores(scores_path)
    if score_payload_hash != SCORES_SHA256:
        raise RuntimeError("Unexpected audited Qwen4B score payload hash")
    required = {"query_id", "doc_id", "qwen_score"}
    if not required.issubset(scores.columns):
        raise RuntimeError(f"Missing score columns: {sorted(required - set(scores.columns))}")
    if scores[list(required)].isna().any().any() or not np.isfinite(scores.qwen_score.to_numpy(float)).all():
        raise RuntimeError("Null/non-finite Qwen scores")
    if scores.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Duplicate query-doc score")
    by_query = {int(qid): group for qid, group in scores.groupby("query_id", sort=False)}

    stats = Counter()
    after_counts: list[int] = []
    try:
        with zipfile.ZipFile(base_zip) as base_archive, zipfile.ZipFile(filtered_zip) as filtered_archive:
            for archive in (base_archive, filtered_archive):
                if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                    raise RuntimeError("Source ZIP layout/CRC failure")
            with (
                base_archive.open("submission.json") as base_binary,
                filtered_archive.open("submission.json") as filtered_binary,
                io.TextIOWrapper(base_binary, encoding="utf-8") as base_text,
                io.TextIOWrapper(filtered_binary, encoding="utf-8") as filtered_text,
                json_path.open("x", encoding="utf-8", newline="\n") as dest,
            ):
                dest.write("[")
                for base_row, filtered_row in zip(
                    iter_json_array(base_text), iter_json_array(filtered_text), strict=True
                ):
                    qid = int(base_row["id"])
                    if qid != int(filtered_row["id"]):
                        raise RuntimeError("Query IDs differ or are reordered")
                    base_docs = [int(value) for value in base_row["relevant_docs"]]
                    if len(base_docs) != 216 or len(set(base_docs)) != 216:
                        raise RuntimeError(f"Invalid base docs for query {qid}")
                    base_chunks = base_row["relevant_chunks"]
                    chunks = filtered_row["relevant_chunks"]
                    if not chunks or not is_subsequence(chunks, base_chunks):
                        raise RuntimeError(f"P2a chunks are not an exact ordered subset for query {qid}")
                    if qid not in by_query or len(by_query[qid]) != 216:
                        raise RuntimeError(f"Incomplete Qwen scores for query {qid}")
                    ranked = (
                        by_query[qid]
                        .sort_values(["qwen_score", "doc_id"], ascending=[False, True], kind="mergesort")
                        .doc_id.astype(int).tolist()
                    )
                    owners_all = {int(chunk["doc_id"]) for chunk in chunks}
                    owners = owners_all & set(base_docs)
                    external_owners = owners_all - set(base_docs)
                    docs = stable_joint_docs(base_docs, ranked, owners, keep_top)
                    if not owners.issubset(set(docs)):
                        raise AssertionError("Filtered internal chunk owner was not retained")
                    row = {"id": qid, "relevant_docs": docs, "relevant_chunks": chunks}
                    if stats["queries"]:
                        dest.write(",\n")
                    dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    stats["queries"] += 1
                    stats["docs_before"] += len(base_docs)
                    stats["docs_after"] += len(docs)
                    stats["chunks_before"] += len(base_chunks)
                    stats["chunks_after"] += len(chunks)
                    stats["owner_query_docs"] += len(owners)
                    stats["external_owner_query_docs"] += len(external_owners)
                    after_counts.append(len(docs))
                dest.write("\n]\n")
        if stats["queries"] != 1200 or set(by_query) != set(
            int(qid) for qid in scores.query_id.unique()
        ) or len(by_query) != 1200:
            raise RuntimeError("Expected exact 1,200-query coverage")

        staged = package_lzma64(json_path, output)
        if staged.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged.stat().st_size}")
        with zipfile.ZipFile(staged) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Built ZIP layout/CRC failure")
        os.replace(staged, output)
    except Exception:
        output.unlink(missing_ok=True)
        output.with_name(output.stem + ".building.zip").unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": "joint_qwen_docs_p2a_chunks_v1",
        "status": "structural_checks_passed_btc_score_pending",
        "policy": {"qwen_keep_top": keep_top, "chunk_policy": "exact P2a BGE keep90 artifact"},
        "base_sha256": sha256_file(base_zip),
        "filtered_chunks_source_sha256": sha256_file(filtered_zip),
        "scores_payload_sha256": score_payload_hash,
        "queries": stats["queries"],
        "docs_before": stats["docs_before"],
        "docs_after": stats["docs_after"],
        "docs_removed": stats["docs_before"] - stats["docs_after"],
        "docs_per_query_min": min(after_counts),
        "docs_per_query_max": max(after_counts),
        "owner_query_docs": stats["owner_query_docs"],
        "external_owner_query_docs_inherited": stats["external_owner_query_docs"],
        "chunks_before": stats["chunks_before"],
        "chunks_after": stats["chunks_after"],
        "chunks_removed": stats["chunks_before"] - stats["chunks_after"],
        "submission_zip_sha256": sha256_file(output),
        "submission_zip_bytes": output.stat().st_size,
        "submission_json_sha256": sha256_file(json_path),
        "quality_claim": "None; component BTC scores do not determine joint document F2.",
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--filtered", type=Path, default=FILTERED)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--keep-top", type=int, default=130)
    args = parser.parse_args()
    build(args.base, args.filtered, args.scores, args.output, args.keep_top)


if __name__ == "__main__":
    main()
