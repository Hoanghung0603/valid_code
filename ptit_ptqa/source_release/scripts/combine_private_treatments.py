"""Combine audited docs-only and chunks-only Exp143 descendants by query ID."""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from scripts.build_final_sprint_exp142_missing_parents import read_rows
from scripts.build_p1_verified_docs_submission import package_lzma64
from scripts.generate_translation_submission import sha256_file


def combine(docs_zip: Path, chunks_zip: Path, control_zip: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    stage = output.with_suffix(".json")
    doc_iter = read_rows(docs_zip)
    chunk_iter = read_rows(chunks_zip)
    control_iter = read_rows(control_zip)
    rows = docs = chunks = 0
    try:
        with stage.open("x", encoding="utf-8", newline="\n") as dest:
            dest.write("[")
            for doc_row, chunk_row, control in zip(doc_iter, chunk_iter, control_iter, strict=True):
                qid = int(control["id"])
                if int(doc_row["id"]) != qid or int(chunk_row["id"]) != qid:
                    raise RuntimeError("Query order/ID mismatch")
                old_docs = set(map(int, control["relevant_docs"]))
                new_docs = list(map(int, doc_row["relevant_docs"]))
                if not old_docs.issubset(new_docs) or len(new_docs) != len(set(new_docs)):
                    raise RuntimeError("Docs treatment removed/duplicated docs")
                if doc_row["relevant_chunks"] != control["relevant_chunks"]:
                    raise RuntimeError("Docs-only input changed chunks")
                if chunk_row["relevant_docs"] != control["relevant_docs"]:
                    raise RuntimeError("Chunks-only input changed docs")
                old_chunks = {(int(c["doc_id"]), str(c["chunk_text"])) for c in control["relevant_chunks"]}
                new_chunks = {(int(c["doc_id"]), str(c["chunk_text"])) for c in chunk_row["relevant_chunks"]}
                if not old_chunks.issubset(new_chunks) or len(new_chunks) != len(chunk_row["relevant_chunks"]):
                    raise RuntimeError("Chunks treatment removed/duplicated chunks")
                # BTC requires each chunk's doc_id to identify its source, but
                # does not require it to appear in relevant_docs. The P2A source
                # offset/hash checks are performed by the chunks-only builder.
                doc_row["relevant_chunks"] = chunk_row["relevant_chunks"]
                if rows:
                    dest.write(",\n")
                dest.write(json.dumps(doc_row, ensure_ascii=False, separators=(",", ":")))
                rows += 1
                docs += len(new_docs)
                chunks += len(new_chunks)
            dest.write("\n]\n")
        if rows != 1200:
            raise RuntimeError("Incomplete query coverage")
        building = package_lzma64(stage, output)
        if building.stat().st_size >= 100_000_000:
            raise RuntimeError("ZIP exceeds 100 decimal MB")
        with zipfile.ZipFile(building) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("ZIP layout/CRC invalid")
        os.replace(building, output)
    finally:
        stage.unlink(missing_ok=True)
    manifest = {"treatment": "docs BM25 consensus add2 + chunks P2A add5 vs Exp143",
                "queries": rows, "docs": docs, "chunks": chunks,
                "docs_input_sha256": sha256_file(docs_zip),
                "chunks_input_sha256": sha256_file(chunks_zip),
                "control_sha256": sha256_file(control_zip),
                "output_sha256": sha256_file(output), "output_bytes": output.stat().st_size,
                "btc_score": None}
    output.with_name(output.stem + "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=Path, required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(combine(a.docs, a.chunks, a.control, a.output), ensure_ascii=False, indent=2))
