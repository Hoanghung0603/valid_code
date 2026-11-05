"""Independently audit P2a Kaggle scores against exact submitted chunk text."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from generate_translation_submission import iter_json_array, sha256_file
from prepare_p2a_chunk_pairs import CONTROL, digest


EXPECTED_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MEMBERS = {"p2a_pair_scores.parquet", "p2a_chunk_scores.parquet", "validation.json", "run_manifest.json"}
PAIR_LINEAGE = ("query_id", "doc_id", "chunk_ordinal", "passage_rank", "window_start",
                "window_end", "chunk_sha256", "window_sha256", "language")
CHUNK_LINEAGE = ("query_id", "chunk_ordinal", "doc_id", "chunk_sha256",
                 "chunk_chars", "language", "window_count")


def validate(input_dir: Path, result_zip: Path, control: Path = CONTROL) -> dict:
    manifest = json.loads((input_dir / "p2a_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "p2a_chunk_windows_v1":
        raise RuntimeError("Wrong input schema")
    if sha256_file(control) != manifest["control_zip_sha256"]:
        raise RuntimeError("BTC-scored P1 control hash mismatch")
    for name in ("p2a_pairs.parquet", "p2a_chunks.parquet"):
        if sha256_file(input_dir / name) != manifest["artifacts"][name]["sha256"]:
            raise RuntimeError(f"Input artifact hash mismatch: {name}")
    pairs = pd.read_parquet(input_dir / "p2a_pairs.parquet")
    chunks = pd.read_parquet(input_dir / "p2a_chunks.parquet")
    if len(pairs) != manifest["validation"]["pairs"] or len(chunks) != manifest["validation"]["chunks"]:
        raise RuntimeError("Input row counts changed")
    if pairs.pair_id.duplicated().any() or chunks.duplicated(["query_id", "chunk_ordinal"]).any():
        raise RuntimeError("Input has duplicate identifiers")

    with zipfile.ZipFile(result_zip) as archive:
        if set(archive.namelist()) != MEMBERS or archive.testzip() is not None:
            raise RuntimeError("Result ZIP layout/CRC failed")
        pair_scores = pd.read_parquet(io.BytesIO(archive.read("p2a_pair_scores.parquet")))
        chunk_scores = pd.read_parquet(io.BytesIO(archive.read("p2a_chunk_scores.parquet")))
        validation = json.loads(archive.read("validation.json"))
        run_manifest = json.loads(archive.read("run_manifest.json"))
        for name in ("p2a_pair_scores.parquet", "p2a_chunk_scores.parquet"):
            import hashlib
            if hashlib.sha256(archive.read(name)).hexdigest() != run_manifest["outputs"][name]:
                raise RuntimeError(f"Output hash mismatch: {name}")
    if validation["schema_version"] != "p2a_chunk_scores_v1" or validation["model_key"] != "bge":
        raise RuntimeError("Wrong result schema/model")
    if (validation["scope"] != manifest["scope"] or
        validation["control_zip_sha256"] != manifest["control_zip_sha256"] or
        validation["input_pairs_sha256"] != manifest["artifacts"]["p2a_pairs.parquet"]["sha256"] or
        validation["input_chunks_sha256"] != manifest["artifacts"]["p2a_chunks.parquet"]["sha256"]):
        raise RuntimeError("Result is not linked to the requested P2a input")
    if validation["model_revision"] != EXPECTED_REVISION:
        raise RuntimeError(f"Unexpected BGE checkpoint revision: {validation['model_revision']}")
    if (validation["pairs"] != len(pairs) or validation["chunks"] != len(chunks) or
        validation["queries"] != chunks.query_id.nunique()):
        raise RuntimeError("Result row counts differ from input")
    if pair_scores.pair_id.duplicated().any() or set(pair_scores.pair_id) != set(pairs.pair_id):
        raise RuntimeError("Missing/duplicate/extra scored pair IDs")
    if chunk_scores.duplicated(["query_id", "chunk_ordinal"]).any() or len(chunk_scores) != len(chunks):
        raise RuntimeError("Missing/duplicate scored chunks")
    paired = pairs.set_index("pair_id").sort_index()
    scored = pair_scores.set_index("pair_id").sort_index()
    for column in PAIR_LINEAGE:
        if not paired[column].equals(scored[column]):
            raise RuntimeError(f"Pair lineage mismatch: {column}")
    original_chunks = chunks.set_index(["query_id", "chunk_ordinal"]).sort_index()
    scored_chunks = chunk_scores.set_index(["query_id", "chunk_ordinal"]).sort_index()
    if not original_chunks.index.equals(scored_chunks.index):
        raise RuntimeError("Chunk identifiers differ")
    for column in CHUNK_LINEAGE[2:]:
        if not original_chunks[column].equals(scored_chunks[column]):
            raise RuntimeError(f"Chunk lineage mismatch: {column}")
    if not np.isfinite(pair_scores[["raw_score", "normalized_score"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite pair score")
    if not np.isfinite(chunk_scores[["bge_chunk_score", "bge_chunk_raw"]].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite chunk score")
    grouped = pair_scores.groupby(["query_id", "chunk_ordinal"]).agg(
        max_score=("normalized_score", "max"), max_raw=("raw_score", "max"), windows=("pair_id", "size"))
    if not grouped.index.equals(scored_chunks.index):
        raise RuntimeError("Window→chunk group membership differs")
    if (not np.allclose(grouped.max_score, scored_chunks.bge_chunk_score, rtol=0, atol=1e-7) or
        not np.allclose(grouped.max_raw, scored_chunks.bge_chunk_raw, rtol=0, atol=1e-7) or
        not np.array_equal(grouped.windows, scored_chunks.windows_scored) or
        not np.array_equal(grouped.windows, scored_chunks.window_count)):
        raise RuntimeError("Window→chunk max aggregation differs")
    for _, group in chunk_scores.groupby("query_id"):
        if set(group.bge_chunk_rank.astype(int)) != set(range(1, len(group) + 1)):
            raise RuntimeError("Chunk ranks are not a permutation")

    pair_groups = {int(qid): group.groupby("chunk_ordinal") for qid, group in pairs.groupby("query_id")}
    chunk_groups = {int(qid): group.set_index("chunk_ordinal") for qid, group in chunks.groupby("query_id")}
    audited_chunks = audited_windows = 0
    with zipfile.ZipFile(control) as archive:
        with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
            for row in iter_json_array(stream):
                qid = int(row["id"])
                if qid not in pair_groups:
                    continue
                expected_chunks = chunk_groups[qid]
                if len(row["relevant_chunks"]) != len(expected_chunks):
                    raise RuntimeError(f"Control chunk count changed for query {qid}")
                for ordinal, chunk in enumerate(row["relevant_chunks"]):
                    original_text = chunk["chunk_text"]
                    expected = expected_chunks.loc[ordinal]
                    if (int(chunk["doc_id"]) != int(expected.doc_id) or
                        digest(original_text) != expected.chunk_sha256 or
                        len(original_text) != int(expected.chunk_chars)):
                        raise RuntimeError(f"Original chunk lineage failed: {qid}:{ordinal}")
                    for window in pair_groups[qid].get_group(ordinal).itertuples(index=False):
                        source_window = original_text[int(window.window_start):int(window.window_end)]
                        if source_window != window.passage_text or digest(source_window) != window.window_sha256:
                            raise RuntimeError(f"Window is not exact source text: {window.pair_id}")
                        audited_windows += 1
                    audited_chunks += 1
    if audited_chunks != len(chunks) or audited_windows != len(pairs):
        raise RuntimeError("Original control source audit incomplete")
    report = {
        "schema_version": "p2a_chunk_score_audit_v1", "scope": manifest["scope"],
        "model_revision": validation["model_revision"], "queries": chunks.query_id.nunique(),
        "chunks": len(chunks), "windows": len(pairs), "exact_source_chunks": audited_chunks,
        "exact_source_windows": audited_windows, "score_lineage_exact": True,
        "window_to_chunk_max_exact": True, "nonfinite_scores": 0,
        "windows_at_token_limit": int((pair_scores.input_tokens >= 1024).sum()),
        "fully_covered_chunks": int(chunks.window_coverage.eq(1.0).sum()),
        "control_zip_sha256": manifest["control_zip_sha256"],
        "input_pairs_sha256": manifest["artifacts"]["p2a_pairs.parquet"]["sha256"],
        "output_zip_sha256": sha256_file(result_zip),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = validate(args.input_dir, args.result, args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
