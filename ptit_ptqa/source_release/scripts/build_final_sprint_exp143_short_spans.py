"""Append a few exact P2a windows for long chunks while preserving Exp142."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

try:
    from build_p1_verified_docs_submission import package_lzma64
    from build_final_sprint_exp142_missing_parents import read_rows
    from generate_translation_submission import sha256_file
except ModuleNotFoundError:
    from scripts.build_p1_verified_docs_submission import package_lzma64
    from scripts.build_final_sprint_exp142_missing_parents import read_rows
    from scripts.generate_translation_submission import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_exp142_missing_owners_add5.zip"
PAIRS = ROOT / "data/kaggle_outputs/p2a_chunk_full/p2a_chunk_full/p2a_pairs.parquet"
SCORES = ROOT / "data/kaggle_outputs/p2a_chunk_full/p2a_chunk_full/p2a_pair_scores.parquet"
MAX_PER_QUERY = 3
LONG_QUANTILE = 0.90
QUERY_COUNT = 1200


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(output: Path, manifest_path: Path, max_per_query: int = MAX_PER_QUERY) -> dict:
    if not 1 <= max_per_query <= 8:
        raise ValueError("max_per_query must be 1..8")
    staging = output.with_suffix(".json")
    building = output.with_name(output.stem + ".building.zip")
    if any(p.exists() for p in (output, manifest_path, staging, building)):
        raise FileExistsError("Refusing to overwrite Exp143 output or staging files")

    # Only use P2a spans whose source chunk hash exists verbatim in Exp142.
    scores = pd.read_parquet(SCORES, columns=["pair_id", "normalized_score"])
    pairs = pd.read_parquet(PAIRS, columns=["pair_id", "query_id", "doc_id", "chunk_sha256",
                                           "window_start", "window_end", "passage_text"])
    frame = pairs.merge(scores, on="pair_id", how="inner", validate="one_to_one")
    if len(frame) != 342_548:
        raise RuntimeError(f"Unexpected P2a window rows: {len(frame)}")
    by_query: dict[int, list[dict]] = {}
    existing_by_query: dict[int, set[tuple[int, str]]] = {}
    source_chunks: dict[tuple[int, int, str], str] = {}
    rows = chunks_control = 0
    for row in read_rows(CONTROL):
        qid = int(row["id"])
        if qid in existing_by_query:
            raise RuntimeError(f"Duplicate query {qid} in Exp142")
        seen = set()
        for chunk in row["relevant_chunks"]:
            doc_id = int(chunk["doc_id"])
            text = str(chunk["chunk_text"])
            key = (doc_id, text)
            if key in seen:
                raise RuntimeError(f"Duplicate control chunk at query {qid}")
            seen.add(key)
            source_chunks[(qid, doc_id, digest(text))] = text
        existing_by_query[qid] = seen
        rows += 1
        chunks_control += len(seen)
    if rows != QUERY_COUNT or chunks_control != 123_835:
        raise RuntimeError(f"Unexpected Exp142 control size: {rows=} {chunks_control=}")

    frame["chunk_chars"] = [len(source_chunks[(int(r.query_id), int(r.doc_id), r.chunk_sha256)])
                             if (int(r.query_id), int(r.doc_id), r.chunk_sha256) in source_chunks else -1
                            for r in frame.itertuples(index=False)]
    # Per chunk, keep its best P2a scoring window; then restrict to the global
    # longest decile of exact-matched control chunks.
    matched = frame[frame.chunk_chars > 0].copy()
    long_cutoff = float(matched.drop_duplicates(["query_id", "doc_id", "chunk_sha256"])
                         .chunk_chars.quantile(LONG_QUANTILE))
    matched = matched[matched.chunk_chars >= long_cutoff]
    best = (matched.sort_values(["query_id", "normalized_score", "doc_id", "window_start"],
                                ascending=[True, False, True, True])
                   .drop_duplicates(["query_id", "doc_id", "chunk_sha256"]))
    for qid, group in best.groupby("query_id", sort=False):
        group = group.sort_values(["normalized_score", "doc_id"], ascending=[False, True]).head(max_per_query)
        by_query[int(qid)] = [{"doc_id": int(r.doc_id), "chunk_text": str(r.passage_text),
                               "chunk_sha256": str(r.chunk_sha256),
                               "window_start": int(r.window_start), "window_end": int(r.window_end),
                               "score": float(r.normalized_score)} for r in group.itertuples(index=False)]

    # Verify exact source slicing using the P2a pair text and control hash,
    # and exclude duplicate spans before writing.
    additions = 0
    qid_coverage = 0
    for qid, candidates in by_query.items():
        qid_coverage += bool(candidates)
        for item in candidates:
            doc_id, text = item["doc_id"], item["chunk_text"]
            original = source_chunks.get((qid, doc_id, item["chunk_sha256"]))
            if original is None or digest(original) != item["chunk_sha256"]:
                raise RuntimeError("P2a source chunk hash does not match current control")
            if item["window_end"] - item["window_start"] != len(text):
                raise RuntimeError("P2a window offsets/hash do not match a current source chunk")
            if original[item["window_start"]:item["window_end"]] != text:
                raise RuntimeError("P2a window text is not the exact source slice")
            if (doc_id, text) in existing_by_query[qid]:
                continue
            additions += 1
    if not additions:
        raise RuntimeError("No unique P2a window additions selected")

    written_queries = added_rows = chunk_total = docs_total = 0
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as dest:
            dest.write("[")
            for row in read_rows(CONTROL):
                qid = int(row["id"])
                existing = {(int(c["doc_id"]), str(c["chunk_text"])) for c in row["relevant_chunks"]}
                selected = by_query.get(qid, [])
                selected = [x for x in selected if (x["doc_id"], x["chunk_text"]) not in existing]
                selected = selected[:max_per_query]
                for item in selected:
                    row["relevant_chunks"].append({"doc_id": item["doc_id"], "chunk_text": item["chunk_text"]})
                    existing.add((item["doc_id"], item["chunk_text"]))
                    added_rows += 1
                if written_queries:
                    dest.write(",\n")
                dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                written_queries += 1
                docs_total += len(row["relevant_docs"])
                chunk_total += len(row["relevant_chunks"])
            dest.write("\n]\n")
        if written_queries != QUERY_COUNT or added_rows != additions:
            raise RuntimeError("Exp143 output row/addition count mismatch")
        staged_zip = package_lzma64(staging, output)
        if staged_zip.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged_zip.stat().st_size}")
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Exp143 ZIP/CRC validation failed")
            with archive.open("submission.json") as packed, staging.open("rb") as raw:
                if hashlib.sha256(packed.read()).digest() != hashlib.sha256(raw.read()).digest():
                    raise RuntimeError("Packaged JSON differs from staged JSON")
        os.replace(staged_zip, output)
    except Exception:
        output.unlink(missing_ok=True)
        building.unlink(missing_ok=True)
        raise
    finally:
        staging.unlink(missing_ok=True)

    manifest = {
        "experiment": "Exp143",
        "status": "exploratory_btc_trial_pending",
        "control": "Exp142; docs and all existing chunks preserved",
        "selection": f"top 10% longest exact-hash-matched chunks; best P2a window per chunk; top {max_per_query}/query by normalized score",
        "queries": written_queries, "docs": docs_total,
        "control_chunks": 123_835, "added_chunks": added_rows, "total_chunks": chunk_total,
        "queries_with_additions": sum(bool(v) for v in by_query.values()),
        "long_chunk_char_cutoff": long_cutoff,
        "p2a_control_zip_sha256": sha256_file(CONTROL),
        "output_zip_bytes": output.stat().st_size,
        "output_zip_sha256": sha256_file(output),
        "source_hashes_match": True,
        "warning": "P2a normalized relevance scores are ranks, not calibrated probabilities; only BTC score can assess this add-only treatment.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    out = ROOT / "submission_exp143_p2a_long_chunks_add3.zip"
    manifest = ROOT / "submission_exp143_p2a_long_chunks_add3_manifest.json"
    print(json.dumps(build(out, manifest), ensure_ascii=False, indent=2))
