"""Build a Qwen4B chunk-filter candidate from exact audited chunk scores."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path

import pandas as pd

try:
    from build_p1_verified_docs_submission import package_lzma64
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.build_p1_verified_docs_submission import package_lzma64
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_best_axes_docs_joint_chunks_top130.zip"
CONTROL_SHA256 = "b6bc6b304a8ccf5d55ebffdfd55e49633ab6822c1736556be65d8438a1e5215d"
SCORES = ROOT / "data/audits/qwen4b_chunks_full_merged/qwen4b_chunk_scores_full.parquet"
SCORE_AUDIT = ROOT / "data/audits/qwen4b_chunks_full_merged/audit.json"
MERGE_MANIFEST = ROOT / "data/audits/qwen4b_chunks_full_merged/manifest.json"
OUTPUT = ROOT / "submission_qwen4b_chunks_keep90_full.zip"
MANIFEST = ROOT / "submission_qwen4b_chunks_keep90_full_manifest.json"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
EXPECTED_CHUNKS = 130_108
EXPECTED_QUERIES = 1_200


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def kept_ordinals(group: pd.DataFrame, keep_fraction: float) -> set[int]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if group.chunk_ordinal.duplicated().any():
        raise RuntimeError("Duplicate chunk ordinal")
    if not group.status.isin(["ok", "too_long"]).all():
        raise RuntimeError("Unexpected score status")
    scored = group[group.status.eq("ok")]
    if scored.normalized_score.isna().any():
        raise RuntimeError("A scored chunk has a missing score")
    keep_n = max(1, math.ceil(len(scored) * keep_fraction)) if len(scored) else 0
    ordered = scored.sort_values(
        ["normalized_score", "chunk_ordinal"], ascending=[False, True], kind="mergesort"
    )
    # Chunks outside the model's 4,096-token limit have unknown relevance and
    # are retained instead of being treated as low-scoring negatives.
    selected = set(group.loc[group.status.eq("too_long"), "chunk_ordinal"].astype(int))
    selected.update(ordered.head(keep_n).chunk_ordinal.astype(int))
    return selected


def build(control: Path, scores_path: Path, audit_path: Path, merge_manifest_path: Path,
          output: Path, manifest_path: Path, keep_fraction: float = 0.90) -> dict:
    if any(path.exists() for path in (output, manifest_path, output.with_suffix(".json"),
                                      output.with_name(output.stem + ".building.zip"))):
        raise FileExistsError("Candidate, manifest, staging JSON, or staged ZIP already exists")
    control_hash = sha256_file(control)
    if control_hash != CONTROL_SHA256:
        raise RuntimeError(f"Unexpected best-axis control hash: {control_hash}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    merge_manifest = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
    if (audit.get("schema_version") != "qwen4b_original_chunk_score_audit_v1"
            or audit.get("passed") is not True or audit.get("rows") != EXPECTED_CHUNKS
            or audit.get("queries") != EXPECTED_QUERIES
            or audit.get("model_revision") != MODEL_REVISION):
        raise RuntimeError("A passing full Qwen4B chunk-score audit is required")
    if (merge_manifest.get("audit_passed") is not True
            or merge_manifest.get("input_sha256") != audit["input"]["sha256"]
            or merge_manifest.get("output", {}).get("sha256") != sha256_file(scores_path)):
        raise RuntimeError("Merged score manifest does not match audited scores")
    scores = pd.read_parquet(scores_path)
    if scores.duplicated(["query_id", "chunk_ordinal"]).any():
        raise RuntimeError("Duplicate scored query/chunk keys")
    if len(scores) != EXPECTED_CHUNKS or scores.query_id.nunique() != EXPECTED_QUERIES:
        raise RuntimeError("Unexpected score coverage")
    selected_by_query = {
        int(qid): kept_ordinals(group, keep_fraction)
        for qid, group in scores.groupby("query_id", sort=False)
    }
    score_counts = scores.groupby("query_id").size().to_dict()
    expected_scored_ordinals = {
        int(qid): set(group.chunk_ordinal.astype(int))
        for qid, group in scores.groupby("query_id", sort=False)
    }

    stage_json = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    queries = kept_total = removed_total = changed_queries = 0
    retained_counts: list[int] = []
    removed_counts: list[int] = []
    try:
        with zipfile.ZipFile(control) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Control ZIP layout/CRC invalid")
            with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as source:
                with stage_json.open("x", encoding="utf-8", newline="\n") as destination:
                    destination.write("[")
                    for row in iter_json_array(source):
                        qid = int(row["id"])
                        if qid not in selected_by_query:
                            raise RuntimeError(f"No scores for query {qid}")
                        original = row["relevant_chunks"]
                        if len(original) != int(score_counts[qid]):
                            raise RuntimeError(f"Chunk count mismatch for query {qid}")
                        if set(range(len(original))) != expected_scored_ordinals[qid]:
                            raise RuntimeError(f"Chunk ordinal coverage mismatch for query {qid}")
                        selected = selected_by_query[qid]
                        kept = [chunk for ordinal, chunk in enumerate(original) if ordinal in selected]
                        if not kept:
                            raise RuntimeError(f"Filtering removed all chunks for query {qid}")
                        row["relevant_chunks"] = kept
                        if queries:
                            destination.write(",\n")
                        destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                        queries += 1
                        kept_total += len(kept)
                        removed_total += len(original) - len(kept)
                        changed_queries += len(kept) != len(original)
                        retained_counts.append(len(kept))
                        removed_counts.append(len(original) - len(kept))
                    destination.write("\n]\n")
        if queries != EXPECTED_QUERIES or len(selected_by_query) != queries:
            raise RuntimeError(f"Expected {EXPECTED_QUERIES} unique queries; found {queries}")
        staged = package_lzma64(stage_json, output)
        if staged.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged.stat().st_size}")
        with zipfile.ZipFile(staged) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Candidate ZIP layout/CRC invalid")
            with archive.open("submission.json") as packed, stage_json.open("rb") as raw:
                packed_hash = sha256_stream(packed)
                raw_hash = sha256_file(stage_json)
                if packed_hash != raw_hash:
                    raise RuntimeError("Packaged JSON differs from source JSON")
        os.replace(staged, output)
    except Exception:
        output.unlink(missing_ok=True)
        output.with_name(output.stem + ".building.zip").unlink(missing_ok=True)
        raise
    finally:
        stage_json.unlink(missing_ok=True)

    result = {
        "schema_version": "qwen4b_chunk_filter_candidate_v1",
        "status": "exploratory_btc_trial_pending",
        "policy": {
            "kind": "per_query_keep_fraction_of_scored_chunks",
            "keep_fraction": keep_fraction,
            "ranking": "descending Qwen4B normalized yes/no logit score",
            "tie_break": "ascending original chunk ordinal",
            "too_long_policy": "preserve because no model score is available",
            "minimum_scored_chunks_kept_per_query": 1,
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "control_zip_sha256": control_hash,
        "scores_parquet_sha256": sha256_file(scores_path),
        "score_audit_sha256": sha256_file(audit_path),
        "merge_manifest_sha256": sha256_file(merge_manifest_path),
        "input_pairs_sha256": audit["input"]["sha256"],
        "submission_zip_sha256": sha256_file(output),
        "submission_json_sha256": packed_hash,
        "zip_bytes": output.stat().st_size,
        "queries": queries,
        "chunks": {
            "original": EXPECTED_CHUNKS, "kept": kept_total, "removed": removed_total,
            "changed_queries": changed_queries,
            "kept_per_query_min": min(retained_counts), "kept_per_query_max": max(retained_counts),
            "removed_per_query_min": min(removed_counts), "removed_per_query_max": max(removed_counts),
        },
        "docs_unchanged": True,
        "retained_chunk_text_exact": True,
        "quality_claim": "No BTC score is implied; submit as one controlled trial.",
    }
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--audit", type=Path, default=SCORE_AUDIT)
    parser.add_argument("--merge-manifest", type=Path, default=MERGE_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--keep-fraction", type=float, default=0.90)
    args = parser.parse_args()
    report = build(args.control, args.scores, args.audit, args.merge_manifest,
                   args.output, args.manifest, args.keep_fraction)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
