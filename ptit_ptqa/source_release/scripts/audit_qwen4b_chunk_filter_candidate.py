"""Independently validate a Qwen4B chunk-filter candidate against its sources."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    from build_qwen4b_chunk_filter_submission import CONTROL_SHA256, kept_ordinals
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.build_qwen4b_chunk_filter_submission import CONTROL_SHA256, kept_ordinals
    from scripts.generate_translation_submission import iter_json_array, sha256_file


def stream_sha256(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"Invalid submission ZIP: {path}")
        with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
            yield from iter_json_array(text)


def audit(control: Path, candidate: Path, scores_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "exploratory_btc_trial_pending":
        raise RuntimeError("Candidate manifest is not marked exploratory/pending")
    control_hash = sha256_file(control)
    if control_hash != CONTROL_SHA256 or manifest.get("control_zip_sha256") != control_hash:
        raise RuntimeError("Best-axis control hash mismatch")
    candidate_hash = sha256_file(candidate)
    if (candidate_hash != manifest.get("submission_zip_sha256")
            or candidate.stat().st_size != manifest.get("zip_bytes")
            or candidate.stat().st_size >= 100_000_000):
        raise RuntimeError("Candidate hash/size mismatch or over BTC size limit")
    scores = pd.read_parquet(scores_path)
    if sha256_file(scores_path) != manifest.get("scores_parquet_sha256"):
        raise RuntimeError("Score parquet hash mismatch")
    fraction = float(manifest["policy"]["keep_fraction"])
    selected = {
        int(qid): kept_ordinals(group, fraction)
        for qid, group in scores.groupby("query_id", sort=False)
    }
    statuses = {
        int(qid): set(group.loc[group.status.eq("too_long"), "chunk_ordinal"].astype(int))
        for qid, group in scores.groupby("query_id", sort=False)
    }
    ordinals = {
        int(qid): set(group.chunk_ordinal.astype(int))
        for qid, group in scores.groupby("query_id", sort=False)
    }
    counts = scores.groupby("query_id").size().to_dict()

    n_queries = before = after = changed = 0
    retained_counts: list[int] = []
    removed_counts: list[int] = []
    for base, treatment in zip(rows(control), rows(candidate), strict=True):
        qid = int(base["id"])
        if int(treatment["id"]) != qid or set(treatment) != {"id", "relevant_docs", "relevant_chunks"}:
            raise RuntimeError(f"Query/schema/order mismatch at row {n_queries + 1}")
        if treatment["relevant_docs"] != base["relevant_docs"]:
            raise RuntimeError(f"Documents changed for query {qid}")
        source_chunks = base["relevant_chunks"]
        if len(source_chunks) != int(counts[qid]) or set(range(len(source_chunks))) != ordinals[qid]:
            raise RuntimeError(f"Control chunks do not align to score ordinals for query {qid}")
        expected = [chunk for ordinal, chunk in enumerate(source_chunks) if ordinal in selected[qid]]
        if treatment["relevant_chunks"] != expected:
            raise RuntimeError(f"Candidate chunks differ from score policy for query {qid}")
        if not statuses[qid].issubset(selected[qid]):
            raise RuntimeError(f"Unscored too_long chunks were removed for query {qid}")
        before += len(source_chunks)
        after += len(expected)
        changed += len(source_chunks) != len(expected)
        retained_counts.append(len(expected))
        removed_counts.append(len(source_chunks) - len(expected))
        n_queries += 1

    if n_queries != 1200 or before != len(scores):
        raise RuntimeError(f"Coverage mismatch: {n_queries} queries, {before} source chunks, {len(scores)} scores")
    reported = manifest["chunks"]
    observed = {
        "original": before, "kept": after, "removed": before - after,
        "changed_queries": changed,
        "kept_per_query_min": min(retained_counts), "kept_per_query_max": max(retained_counts),
        "removed_per_query_min": min(removed_counts), "removed_per_query_max": max(removed_counts),
    }
    if reported != observed:
        raise RuntimeError(f"Manifest counts differ: {reported} != {observed}")
    with zipfile.ZipFile(candidate) as archive, archive.open("submission.json") as payload:
        payload_hash = stream_sha256(payload)
    if payload_hash != manifest.get("submission_json_sha256"):
        raise RuntimeError("Submission JSON hash mismatch")
    return {
        "schema_version": "qwen4b_chunk_filter_candidate_audit_v1",
        "passed": True,
        "queries": n_queries,
        "chunks": observed,
        "docs_unchanged": True,
        "retained_chunk_text_order_exact": True,
        "too_long_chunks_preserved": True,
        "candidate_zip_bytes": candidate.stat().st_size,
        "candidate_zip_sha256": candidate_hash,
        "control_zip_sha256": control_hash,
        "scores_parquet_sha256": sha256_file(scores_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.control, args.candidate, args.scores, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
