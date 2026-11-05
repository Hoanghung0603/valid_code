"""Build a chunk-membership-only P2a candidate from audited BGE chunk scores."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

try:  # Support both `python scripts/...` and package-style test imports.
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_p1_qwen4b_top195_full.zip"
SCORES = ROOT / "data/kaggle_outputs/p2a_chunk_full/p2a_bge_chunk_full_scores.zip"
AUDIT = ROOT / "data/audits/p2a_chunk_full_audit.json"
INPUT_DIR = ROOT / "data/kaggle_inputs/p2a_chunk_full_v1"
OUTPUT = ROOT / "submission_p2a_bge_keep90_full.zip"
OUTPUT_MANIFEST = ROOT / "submission_p2a_bge_keep90_full_manifest.json"
CONTROL_SHA256 = "785637cb462f014f1bf35c1c994372a2561bac924ef416c0ccaea0a6b10b0c28"
EXPECTED_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EXPECTED_QUERIES = 1200


def choose_kept_ordinals(query_scores: pd.DataFrame, keep_fraction: float) -> set[int]:
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if query_scores.empty:
        raise ValueError("Every query must have at least one scored chunk")
    if query_scores["chunk_ordinal"].duplicated().any():
        raise ValueError("Duplicate chunk ordinal in query scores")
    if query_scores["bge_chunk_score"].isna().any():
        raise ValueError("Missing chunk score")
    ordered = query_scores.sort_values(
        ["bge_chunk_score", "chunk_ordinal"], ascending=[False, True], kind="mergesort"
    )
    keep_n = max(1, math.ceil(len(ordered) * keep_fraction))
    return set(ordered.iloc[:keep_n]["chunk_ordinal"].astype(int))


def build_candidate(
    control_zip: Path,
    scores_zip: Path,
    audit_json: Path,
    input_dir: Path,
    output_zip: Path,
    output_manifest: Path,
    keep_fraction: float = 0.90,
) -> dict:
    if output_zip.exists() or output_manifest.exists():
        raise FileExistsError("Refusing to overwrite a prior candidate or manifest")
    control_hash = sha256_file(control_zip)
    if control_hash != CONTROL_SHA256:
        raise RuntimeError(f"Unexpected P1 top195 control hash: {control_hash}")

    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "p2a_chunk_score_audit_v1" or audit.get("scope") != "full":
        raise RuntimeError("A passing full-scope P2a audit is required")
    if audit.get("control_zip_sha256") != control_hash:
        raise RuntimeError("Audit control hash mismatch")
    if audit.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Unexpected reranker revision")
    if audit.get("score_lineage_exact") is not True or audit.get("window_to_chunk_max_exact") is not True:
        raise RuntimeError("Score lineage / aggregation audit did not pass")

    input_manifest_bytes = (input_dir / "p2a_manifest.json").read_bytes()
    input_manifest = json.loads(input_manifest_bytes)
    if input_manifest.get("scope") != "full" or input_manifest.get("control_zip_sha256") != control_hash:
        raise RuntimeError("Full input manifest is not linked to the control")
    for name in ("p2a_pairs.parquet", "p2a_chunks.parquet"):
        if sha256_file(input_dir / name) != input_manifest["artifacts"][name]["sha256"]:
            raise RuntimeError(f"P2a full input changed: {name}")

    with zipfile.ZipFile(scores_zip) as archive:
        expected_members = {
            "p2a_pair_scores.parquet", "p2a_chunk_scores.parquet", "validation.json", "run_manifest.json"
        }
        if set(archive.namelist()) != expected_members or archive.testzip() is not None:
            raise RuntimeError("Score ZIP layout or CRC invalid")
        score_validation = json.loads(archive.read("validation.json"))
        chunk_scores = pd.read_parquet(io.BytesIO(archive.read("p2a_chunk_scores.parquet")))
    if (score_validation.get("scope") != "full" or
        score_validation.get("control_zip_sha256") != control_hash or
        score_validation.get("model_revision") != EXPECTED_MODEL_REVISION or
        len(chunk_scores) != audit.get("chunks")):
        raise RuntimeError("Score ZIP does not match audited full run")
    if chunk_scores.duplicated(["query_id", "chunk_ordinal"]).any():
        raise RuntimeError("Duplicate scored chunk keys")
    if chunk_scores.bge_chunk_score.isna().any():
        raise RuntimeError("Missing BGE chunk scores")

    keep_by_query = {
        int(qid): choose_kept_ordinals(group, keep_fraction)
        for qid, group in chunk_scores.groupby("query_id", sort=False)
    }
    score_counts = chunk_scores.groupby("query_id").size().to_dict()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    total_original = total_kept = changed_queries = 0
    kept_counts: list[int] = []
    removed_counts: list[int] = []
    temporary_json = output_zip.with_suffix(".json")
    if temporary_json.exists():
        raise FileExistsError(temporary_json)

    try:
        with zipfile.ZipFile(control_zip) as control:
            if control.namelist() != ["submission.json"] or control.testzip() is not None:
                raise RuntimeError("Control ZIP must contain only a valid submission.json")
            with control.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
                with temporary_json.open("w", encoding="utf-8", newline="\n") as out:
                    out.write("[")
                    for row in iter_json_array(stream):
                        qid = int(row["id"])
                        if qid not in keep_by_query:
                            raise RuntimeError(f"No scores for query {qid}")
                        chunks = row["relevant_chunks"]
                        selected = keep_by_query[qid]
                        if len(chunks) != int(score_counts[qid]):
                            raise RuntimeError(f"Control/score chunk count mismatch for query {qid}")
                        kept = [chunk for ordinal, chunk in enumerate(chunks) if ordinal in selected]
                        if not kept:
                            raise RuntimeError(f"Filtering removed every chunk for query {qid}")
                        row["relevant_chunks"] = kept
                        if rows:
                            out.write(",\n")
                        out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                        rows += 1
                        total_original += len(chunks)
                        total_kept += len(kept)
                        kept_counts.append(len(kept))
                        removed_counts.append(len(chunks) - len(kept))
                        changed_queries += len(kept) != len(chunks)
                    if rows != EXPECTED_QUERIES or rows != len(keep_by_query):
                        raise RuntimeError(f"Expected {EXPECTED_QUERIES} unique scored queries, found {rows}")
                    out.write("\n]\n")

        # Match the source submission's ZIP-LZMA 64 MiB dictionary. Python's
        # default ZIP-LZMA dictionary can expand this large verbatim payload.
        seven_zip = shutil.which("7z")
        if not seven_zip:
            raise RuntimeError("7z is required to package the submission under BTC's 100 MB limit")
        staged_zip = output_zip.with_name(output_zip.stem + ".building.zip")
        if staged_zip.exists():
            raise FileExistsError(staged_zip)
        with temporary_json.open("rb") as source:
            subprocess.run(
                [seven_zip, "a", "-tzip", "-mm=LZMA", "-md=64m", "-mx=9",
                 str(staged_zip.resolve()), "-sisubmission.json"],
                stdin=source, capture_output=True, text=True, check=True,
            )
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Candidate ZIP validation failed")
            with archive.open("submission.json") as source, temporary_json.open("rb") as expected:
                if hashlib.sha256(source.read()).digest() != hashlib.sha256(expected.read()).digest():
                    raise RuntimeError("Candidate ZIP payload differs from candidate JSON")
        if staged_zip.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds BTC's decimal 100 MB limit: {staged_zip.stat().st_size} bytes")
        os.replace(staged_zip, output_zip)
    except Exception:
        if output_zip.exists():
            output_zip.unlink()
        staged_zip = output_zip.with_name(output_zip.stem + ".building.zip")
        staged_zip.unlink(missing_ok=True)
        raise
    finally:
        temporary_json.unlink(missing_ok=True)

    manifest = {
        "experiment": "P2a BGE chunk-membership filter on P1 top195, keep top 90% per query",
        "status": "exploratory_btc_trial_pending",
        "policy": {"kind": "per_query_rank_fraction", "keep_fraction": keep_fraction,
                   "ranking": "descending max normalized BGE score per exact source chunk",
                   "tie_break": "ascending original chunk ordinal", "minimum_kept_per_query": 1},
        "model": {"id": "BAAI/bge-reranker-v2-m3", "revision": EXPECTED_MODEL_REVISION},
        "control_zip_sha256": control_hash,
        "score_zip_sha256": sha256_file(scores_zip),
        "score_audit_sha256": sha256_file(audit_json),
        "input_manifest_sha256": hashlib.sha256(input_manifest_bytes).hexdigest(),
        "submission_zip_sha256": sha256_file(output_zip),
        "submission_json_sha256": sha256_file(temporary_json) if temporary_json.exists() else None,
        "zip_bytes": output_zip.stat().st_size,
        "queries": rows,
        "chunks": {"original": total_original, "kept": total_kept, "removed": total_original-total_kept,
                   "changed_queries": changed_queries,
                   "kept_per_query_min": min(kept_counts), "kept_per_query_max": max(kept_counts),
                   "removed_per_query_min": min(removed_counts), "removed_per_query_max": max(removed_counts)},
        "docs_unchanged": True,
        "retained_chunk_text_exact": True,
        "quality_claim": "None; no reliable qrels/local scorer. Evaluate with BTC only.",
    }
    # The JSON was deleted after packaging; retain its payload hash from the ZIP.
    with zipfile.ZipFile(output_zip) as archive:
        manifest["submission_json_sha256"] = hashlib.sha256(archive.read("submission.json")).hexdigest()
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--keep-fraction", type=float, default=0.90)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.control, args.scores, args.audit, args.input_dir,
                                     args.output, args.manifest, args.keep_fraction),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
