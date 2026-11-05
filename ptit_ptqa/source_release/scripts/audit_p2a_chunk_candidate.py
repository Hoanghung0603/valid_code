"""Independently verify a P2a chunk-filter submission against its control/scores."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.generate_translation_submission import iter_json_array, sha256_file

from build_p2a_chunk_filter_submission import choose_kept_ordinals


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_p1_qwen4b_top195_full.zip"
CANDIDATE = ROOT / "submission_p2a_bge_keep90_full.zip"
MANIFEST = ROOT / "submission_p2a_bge_keep90_full_manifest.json"
SCORES = ROOT / "data/kaggle_outputs/p2a_chunk_full/p2a_bge_chunk_full_scores.zip"
EXPECTED_CONTROL_SHA256 = "785637cb462f014f1bf35c1c994372a2561bac924ef416c0ccaea0a6b10b0c28"


def rows_from_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"Invalid submission ZIP: {path}")
        with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
            yield from iter_json_array(stream)


def audit(control: Path, candidate: Path, manifest_path: Path, scores_zip: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "exploratory_btc_trial_pending":
        raise RuntimeError("Candidate manifest is not marked exploratory/pending")
    control_hash = sha256_file(control)
    if control_hash != EXPECTED_CONTROL_SHA256 or manifest.get("control_zip_sha256") != control_hash:
        raise RuntimeError("P1 top195 control hash mismatch")
    if sha256_file(candidate) != manifest.get("submission_zip_sha256"):
        raise RuntimeError("Candidate ZIP hash mismatch")
    if candidate.stat().st_size != manifest.get("zip_bytes") or candidate.stat().st_size >= 100_000_000:
        raise RuntimeError("Candidate size is inconsistent or exceeds BTC limit")

    fraction = float(manifest["policy"]["keep_fraction"])
    with zipfile.ZipFile(scores_zip) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Score ZIP CRC failed")
        validation = json.loads(archive.read("validation.json"))
        scores = pd.read_parquet(io.BytesIO(archive.read("p2a_chunk_scores.parquet")))
    if (validation.get("scope") != "full" or
        validation.get("control_zip_sha256") != control_hash or
        validation.get("model_revision") != manifest["model"]["revision"] or
        sha256_file(scores_zip) != manifest.get("score_zip_sha256")):
        raise RuntimeError("Candidate is linked to a different score run")
    expected_keep = {
        int(qid): choose_kept_ordinals(group, fraction)
        for qid, group in scores.groupby("query_id", sort=False)
    }

    query_count = 0
    original_chunks = retained_chunks = 0
    changed_queries = 0
    kept_min = None
    kept_max = 0
    removed_min = None
    removed_max = 0
    for base, treatment in zip(rows_from_zip(control), rows_from_zip(candidate), strict=True):
        qid = int(base["id"])
        if treatment["id"] != qid or set(base) != {"id", "relevant_docs", "relevant_chunks"}:
            raise RuntimeError(f"Query/schema mismatch at query {query_count + 1}")
        if treatment["relevant_docs"] != base["relevant_docs"]:
            raise RuntimeError(f"Docs changed for query {qid}")
        source_chunks = base["relevant_chunks"]
        expected_ordinals = expected_keep.get(qid)
        if expected_ordinals is None:
            raise RuntimeError(f"Missing scores for query {qid}")
        expected_chunks = [chunk for i, chunk in enumerate(source_chunks) if i in expected_ordinals]
        if treatment["relevant_chunks"] != expected_chunks:
            raise RuntimeError(f"Chunk content/order/policy mismatch for query {qid}")
        n_kept = len(expected_chunks)
        n_removed = len(source_chunks) - n_kept
        original_chunks += len(source_chunks)
        retained_chunks += n_kept
        changed_queries += n_removed > 0
        kept_min = n_kept if kept_min is None else min(kept_min, n_kept)
        kept_max = max(kept_max, n_kept)
        removed_min = n_removed if removed_min is None else min(removed_min, n_removed)
        removed_max = max(removed_max, n_removed)
        query_count += 1

    if query_count != validation.get("queries") or query_count != 1200:
        raise RuntimeError(f"Expected 1,200 queries, found {query_count}")
    reported = manifest["chunks"]
    observed = {"original": original_chunks, "kept": retained_chunks,
                "removed": original_chunks - retained_chunks, "changed_queries": changed_queries,
                "kept_per_query_min": kept_min, "kept_per_query_max": kept_max,
                "removed_per_query_min": removed_min, "removed_per_query_max": removed_max}
    if observed != reported:
        raise RuntimeError(f"Manifest counts mismatch: {observed} != {reported}")
    with zipfile.ZipFile(candidate) as archive:
        payload = archive.read("submission.json")
    if hashlib.sha256(payload).hexdigest() != manifest.get("submission_json_sha256"):
        raise RuntimeError("Candidate JSON payload hash mismatch")
    return {"schema_version": "p2a_chunk_candidate_audit_v1", "queries": query_count,
            "docs_unchanged": True, "retained_chunk_text_exact": True,
            "top_fraction_policy_exact": True, "chunks": observed,
            "zip_bytes": candidate.stat().st_size, "under_100mb": True,
            "control_zip_sha256": control_hash,
            "candidate_zip_sha256": sha256_file(candidate),
            "score_zip_sha256": sha256_file(scores_zip)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--output", type=Path, default=ROOT / "data/audits/p2a_chunk_candidate_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.control, args.candidate, args.manifest, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
