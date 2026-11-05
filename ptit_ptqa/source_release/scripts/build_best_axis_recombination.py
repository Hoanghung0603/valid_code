"""Recombine the best BTC-scored document and chunk axes without rescoring.

BTC specifies independent document/chunk macro F2 metrics followed by their
arithmetic mean.  This builder takes relevant_docs from the scored joint
owner-release artifact and relevant_chunks from the scored Qwen-top130
docs-only artifact.  It never edits IDs or chunk text.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

try:
    from build_p1_verified_docs_submission import package_lzma64
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.build_p1_verified_docs_submission import package_lzma64
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = ROOT / "submission_joint_qwen_top130_p2a_keep90_v1.zip"
CHUNKS_SOURCE = ROOT / "submission_p1_qwen4b_top130_full.zip"
OUTPUT = ROOT / "submission_best_axes_docs_joint_chunks_top130.zip"
DOCS_SOURCE_SHA256 = "d1e618b33b4106ff832cabb2fd281a25b6abe44decead29199388c80294c7da0"
CHUNKS_SOURCE_SHA256 = "ec32de34c32aca79b8a9986962291162cdc3042a267c2ef7fa26db369ebf6d27"


def compose_row(docs_row: dict, chunks_row: dict) -> dict:
    if int(docs_row["id"]) != int(chunks_row["id"]):
        raise ValueError("Query IDs differ")
    docs = [int(value) for value in docs_row["relevant_docs"]]
    chunks = chunks_row["relevant_chunks"]
    if not docs or len(docs) != len(set(docs)):
        raise ValueError("Documents must be nonempty unique integer IDs")
    if not chunks or any(
        not isinstance(chunk.get("doc_id"), int) or not isinstance(chunk.get("chunk_text"), str)
        or not chunk["chunk_text"] for chunk in chunks
    ):
        raise ValueError("Chunks must preserve valid integer owners and nonempty text")
    return {"id": int(docs_row["id"]), "relevant_docs": docs, "relevant_chunks": chunks}


def build() -> dict:
    manifest_path = OUTPUT.with_name(OUTPUT.stem + "_manifest.json")
    json_path = OUTPUT.with_suffix(".json")
    if any(path.exists() for path in (OUTPUT, manifest_path, json_path)):
        raise FileExistsError("Output, manifest, or staging JSON already exists")
    if sha256_file(DOCS_SOURCE) != DOCS_SOURCE_SHA256:
        raise RuntimeError("Unexpected best-doc source hash")
    if sha256_file(CHUNKS_SOURCE) != CHUNKS_SOURCE_SHA256:
        raise RuntimeError("Unexpected best-chunk source hash")

    stats = Counter()
    try:
        with zipfile.ZipFile(DOCS_SOURCE) as dz, zipfile.ZipFile(CHUNKS_SOURCE) as cz:
            for archive in (dz, cz):
                if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                    raise RuntimeError("Source ZIP layout/CRC failure")
            with (
                dz.open("submission.json") as db, cz.open("submission.json") as cb,
                io.TextIOWrapper(db, encoding="utf-8") as dt,
                io.TextIOWrapper(cb, encoding="utf-8") as ct,
                json_path.open("x", encoding="utf-8", newline="\n") as dest,
            ):
                dest.write("[")
                for docs_row, chunks_row in zip(iter_json_array(dt), iter_json_array(ct), strict=True):
                    row = compose_row(docs_row, chunks_row)
                    if stats["queries"]:
                        dest.write(",\n")
                    dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                    docs = set(row["relevant_docs"])
                    owners = {int(chunk["doc_id"]) for chunk in row["relevant_chunks"]}
                    stats["queries"] += 1
                    stats["docs"] += len(docs)
                    stats["chunks"] += len(row["relevant_chunks"])
                    stats["chunk_owner_query_docs_outside_relevant_docs"] += len(owners - docs)
                dest.write("\n]\n")
        if stats["queries"] != 1200:
            raise RuntimeError(f"Expected 1,200 queries, got {stats['queries']}")

        staged = package_lzma64(json_path, OUTPUT)
        if staged.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged.stat().st_size}")

        # Independent streaming equality check against both axis sources.
        with (
            zipfile.ZipFile(DOCS_SOURCE) as dz, zipfile.ZipFile(CHUNKS_SOURCE) as cz,
            zipfile.ZipFile(staged) as oz,
        ):
            if oz.namelist() != ["submission.json"] or oz.testzip() is not None:
                raise RuntimeError("Output ZIP layout/CRC failure")
            with (
                dz.open("submission.json") as db, cz.open("submission.json") as cb,
                oz.open("submission.json") as ob,
                io.TextIOWrapper(db, encoding="utf-8") as dt,
                io.TextIOWrapper(cb, encoding="utf-8") as ct,
                io.TextIOWrapper(ob, encoding="utf-8") as ot,
            ):
                validated = 0
                for docs_row, chunks_row, output_row in zip(
                    iter_json_array(dt), iter_json_array(ct), iter_json_array(ot), strict=True
                ):
                    if output_row != compose_row(docs_row, chunks_row):
                        raise RuntimeError("Packaged row differs from exact axis recombination")
                    validated += 1
                if validated != 1200:
                    raise RuntimeError("Packaged query count differs")
        os.replace(staged, OUTPUT)
    except Exception:
        OUTPUT.unlink(missing_ok=True)
        OUTPUT.with_name(OUTPUT.stem + ".building.zip").unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": "best_axis_recombination_v1",
        "status": "structural_checks_passed_btc_score_pending",
        "queries": stats["queries"],
        "docs": stats["docs"],
        "chunks": stats["chunks"],
        "chunk_owner_query_docs_outside_relevant_docs": stats[
            "chunk_owner_query_docs_outside_relevant_docs"
        ],
        "docs_source_sha256": DOCS_SOURCE_SHA256,
        "chunks_source_sha256": CHUNKS_SOURCE_SHA256,
        "submission_json_sha256": sha256_file(json_path),
        "submission_zip_sha256": sha256_file(OUTPUT),
        "submission_zip_bytes": OUTPUT.stat().st_size,
        "btc_scored_axis_metrics": {
            "docs_f2macro": 0.3731,
            "chunks_f2macro": 0.2565,
            "formula_mean": 0.3148,
            "note": "Expected only if BTC evaluator/version remains identical; official score pending.",
        },
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
