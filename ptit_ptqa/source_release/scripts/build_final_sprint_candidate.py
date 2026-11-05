"""Build the bounded final-sprint candidate from audited local score caches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
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
BEST_AXIS = ROOT / "submission_best_axes_docs_joint_chunks_top130.zip"
QWEN_KEEP90 = ROOT / "submission_qwen4b_chunks_keep90_full.zip"
QWEN_DOCS = ROOT / "data/audits/p1_full_qwen4b_merged/p1_full_qwen4b_merged.parquet"
EXP141_DOCS = 130
QUERY_COUNT = 1200


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"Invalid source ZIP: {path}")
        with archive.open("submission.json") as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            yield from iter_json_array(text)


def build_exp141(output: Path, manifest_path: Path) -> dict:
    if any(p.exists() for p in (output, manifest_path, output.with_suffix(".json"),
                                 output.with_name(output.stem + ".building.zip"))):
        raise FileExistsError("Refusing to overwrite candidate, manifest, or staging files")
    qwen = pd.read_parquet(QWEN_DOCS)
    if (len(qwen) != 259_200 or qwen.groupby("query_id").size().ne(216).any()
            or qwen.duplicated(["query_id", "doc_id"]).any()):
        raise RuntimeError("Qwen doc score cache has invalid coverage")
    ranked = {
        int(qid): set(group.sort_values(
            ["qwen4b_score", "doc_id"], ascending=[False, True], kind="mergesort"
        ).head(EXP141_DOCS).doc_id.astype(int))
        for qid, group in qwen.groupby("query_id", sort=False)
    }
    if len(ranked) != QUERY_COUNT:
        raise RuntimeError("Expected exactly 1,200 ranked queries")

    staging = output.with_suffix(".json")
    rows = docs = chunks = removed_docs = 0
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as dest:
            dest.write("[")
            for row in read_rows(QWEN_KEEP90):
                qid = int(row["id"])
                if qid not in ranked:
                    raise RuntimeError(f"No Qwen ranking for query {qid}")
                base_docs = set(map(int, row["relevant_docs"]))
                selected = ranked[qid]
                if len(selected) != EXP141_DOCS or not selected.issubset(base_docs):
                    raise RuntimeError(f"Top130 lineage differs from best-axis docs for {qid}")
                row["relevant_docs"] = [int(doc) for doc in row["relevant_docs"] if int(doc) in selected]
                if len(row["relevant_docs"]) != EXP141_DOCS:
                    raise RuntimeError(f"Docs were lost while preserving control order for {qid}")
                if rows:
                    dest.write(",\n")
                dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                rows += 1
                docs += len(selected)
                chunks += len(row["relevant_chunks"])
                removed_docs += len(base_docs - selected)
            dest.write("\n]\n")
        if rows != QUERY_COUNT:
            raise RuntimeError(f"Expected 1,200 output rows, found {rows}")
        staged_zip = package_lzma64(staging, output)
        if staged_zip.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged_zip.stat().st_size}")
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Candidate ZIP layout/CRC failure")
            with archive.open("submission.json") as packed, staging.open("rb") as raw:
                if sha256_stream(packed) != sha256_stream(raw):
                    raise RuntimeError("Packaged payload differs from staging JSON")
        os.replace(staged_zip, output)
    except Exception:
        output.unlink(missing_ok=True)
        output.with_name(output.stem + ".building.zip").unlink(missing_ok=True)
        raise
    finally:
        staging.unlink(missing_ok=True)

    manifest = {
        "schema_version": "final_sprint_exp141_v1",
        "status": "exploratory_btc_trial_pending",
        "policy": "Qwen4B pure top130 docs; exact Exp139 keep90 chunks; stable control order",
        "queries": rows,
        "docs": docs,
        "chunks": chunks,
        "removed_docs_vs_exp139": removed_docs,
        "source_zip_sha256": sha256_file(QWEN_KEEP90),
        "score_cache_sha256": sha256_file(QWEN_DOCS),
        "submission_zip_sha256": sha256_file(output),
        "submission_zip_bytes": output.stat().st_size,
        "source_chunk_text_preserved": True,
        "warning": "Structural candidate only; no local qrels or BTC score prediction.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["141"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "submission_exp141_qwen_top130_keep90.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "submission_exp141_qwen_top130_keep90_manifest.json")
    args = parser.parse_args()
    print(json.dumps(build_exp141(args.output, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
