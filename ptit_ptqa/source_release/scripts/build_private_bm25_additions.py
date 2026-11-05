"""Add a small, gated BM25 multi-view document set to the Exp143 control.

No labels are used; output is an exploratory private treatment, not a winner.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

from scripts.build_final_sprint_exp142_missing_parents import read_rows
from scripts.build_p1_verified_docs_submission import package_lzma64
from scripts.generate_translation_submission import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_private01_exp143_replay.zip"
CANDIDATES = ROOT / "data/indices/bm25_document/queryspec_candidates_v1.parquet"
SPECS = ROOT / "data/query_specs/queryspec_production_v1/query_specs.parquet"
CORPUS = ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"


def build(*, cap: int, min_hits: int, output: Path) -> dict:
    if not 1 <= cap <= 8 or not 1 <= min_hits <= 10:
        raise ValueError("Unsafe expansion parameters")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    valid = set(pd.read_parquet(SPECS, columns=["query_id", "status"]).query("status == 'ok'").query_id.astype(int))
    official = set(pd.read_parquet(CORPUS, columns=["id"])["id"].astype(int))
    frame = pd.read_parquet(CANDIDATES, columns=["query_id", "rank", "doc_id", "view_hits", "best_view_family"])
    frame = frame[(frame.query_id.isin(valid)) & (frame["rank"] <= 150) &
                  (frame.view_hits >= min_hits) &
                  (frame.best_view_family.isin(["original_vi", "canonical_vi", "atom_vi"]))]
    if frame.duplicated(["query_id", "doc_id"]).any() or not set(frame.doc_id.astype(int)).issubset(official):
        raise RuntimeError("Candidate duplicate or out-of-corpus ID")
    choices = {int(q): list(map(int, group.sort_values("rank").doc_id))
               for q, group in frame.groupby("query_id")}
    stage = output.with_suffix(".json")
    manifest_path = output.with_name(output.stem + "_manifest.json")
    changed = added = docs = chunks = rows = 0
    try:
        with stage.open("x", encoding="utf-8", newline="\n") as dest:
            dest.write("[")
            for row in read_rows(CONTROL):
                qid = int(row["id"])
                old = list(map(int, row["relevant_docs"]))
                seen = set(old)
                addition = []
                for doc_id in choices.get(qid, []):
                    if doc_id not in seen:
                        addition.append(doc_id)
                        seen.add(doc_id)
                        if len(addition) == cap:
                            break
                if addition:
                    row["relevant_docs"] = old + addition
                    changed += 1
                    added += len(addition)
                if rows:
                    dest.write(",\n")
                dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                docs += len(row["relevant_docs"])
                chunks += len(row["relevant_chunks"])
                rows += 1
            dest.write("\n]\n")
        if rows != 1200 or chunks != 127271 or not added:
            raise RuntimeError("Submission coverage/chunk preservation failed")
        building = package_lzma64(stage, output)
        if building.stat().st_size >= 100_000_000:
            raise RuntimeError("ZIP exceeds 100 decimal MB")
        with zipfile.ZipFile(building) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("ZIP layout or CRC invalid")
        os.replace(building, output)
    finally:
        stage.unlink(missing_ok=True)
    manifest = {
        "treatment": "Exp143 docs append BM25 multi-view VI consensus; Exp143 chunks unchanged",
        "cap": cap, "min_view_hits": min_hits, "max_candidate_rank": 150,
        "query_gate": "QuerySpec status=ok only", "queries": rows,
        "queries_changed": changed, "docs_added": added, "docs_total": docs,
        "chunks_total": chunks, "input_candidates_sha256": sha256_file(CANDIDATES),
        "control_sha256": sha256_file(CONTROL), "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size, "btc_score": None,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cap", type=int, required=True)
    p.add_argument("--min-hits", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(cap=a.cap, min_hits=a.min_hits, output=a.output), ensure_ascii=False, indent=2))
