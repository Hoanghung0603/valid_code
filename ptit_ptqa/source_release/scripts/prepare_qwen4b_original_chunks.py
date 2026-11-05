"""Prepare exact control chunks for Qwen4B verification on Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_best_axes_docs_joint_chunks_top130.zip"
CONTROL_SHA256 = "b6bc6b304a8ccf5d55ebffdfd55e49633ab6822c1736556be65d8438a1e5215d"
QUERIES = ROOT / "data/raw/btc_provided/query.parquet"
PILOT = ROOT / "data/audits/queryspec_pilot_v6_final/pilot_manifest.parquet"

SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("query_id", pa.int64()), ("chunk_ordinal", pa.int32()),
    ("doc_id", pa.int64()), ("query_text", pa.string()), ("passage_text", pa.string()),
    ("query_sha256", pa.string()), ("passage_sha256", pa.string()),
    ("passage_chars", pa.int32()),
])


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare(scope: str, output: Path) -> dict:
    output = output.resolve()
    pairs_path = output / "qwen_chunk_pairs.parquet"
    manifest_path = output / "qwen_chunk_manifest.json"
    if any(path.exists() for path in (pairs_path, manifest_path)):
        raise FileExistsError(output)
    if sha256_file(CONTROL) != CONTROL_SHA256:
        raise RuntimeError("Best-axis control hash changed")
    official = pd.read_parquet(QUERIES, columns=["id", "query"])
    if len(official) != 1200 or official.id.duplicated().any():
        raise RuntimeError("Expected 1,200 official queries")
    query_map = dict(zip(official.id.astype(int), official["query"].astype(str), strict=True))
    if scope == "pilot":
        pilot = pd.read_parquet(PILOT, columns=["id"])
        if len(pilot) != 120 or pilot.id.duplicated().any():
            raise RuntimeError("Expected fixed 120-query pilot")
        wanted = set(pilot.id.astype(int))
    elif scope == "full":
        wanted = set(query_map)
    else:
        raise ValueError(scope)

    output.mkdir(parents=True, exist_ok=False)
    writer = pq.ParquetWriter(pairs_path, SCHEMA, compression="zstd")
    seen: set[int] = set()
    chunks = 0
    lengths: list[int] = []
    owners = Counter()
    try:
        with zipfile.ZipFile(CONTROL) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Invalid control ZIP")
            with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
                for row in iter_json_array(text):
                    qid = int(row["id"])
                    if qid not in wanted:
                        continue
                    if qid in seen:
                        raise RuntimeError(f"Duplicate query {qid}")
                    seen.add(qid)
                    query = query_map[qid]
                    records = []
                    for ordinal, chunk in enumerate(row["relevant_chunks"]):
                        passage = chunk["chunk_text"]
                        if not isinstance(passage, str) or not passage.strip():
                            raise RuntimeError(f"Empty chunk q={qid}, ordinal={ordinal}")
                        doc_id = int(chunk["doc_id"])
                        records.append({
                            "pair_id": f"{qid}:{ordinal}", "query_id": qid,
                            "chunk_ordinal": ordinal, "doc_id": doc_id,
                            "query_text": query, "passage_text": passage,
                            "query_sha256": digest(query), "passage_sha256": digest(passage),
                            "passage_chars": len(passage),
                        })
                        chunks += 1; lengths.append(len(passage)); owners[doc_id] += 1
                    if not records:
                        raise RuntimeError(f"No chunks for query {qid}")
                    writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
                    if len(seen) % 20 == 0:
                        print(f"prepared {len(seen)}/{len(wanted)} queries, {chunks} chunks", flush=True)
    finally:
        writer.close()
    expected_chunks = 130_108 if scope == "full" else None
    if seen != wanted or (expected_chunks is not None and chunks != expected_chunks):
        raise RuntimeError(f"Coverage mismatch: {len(seen)} queries, {chunks} chunks")

    series = pd.Series(lengths)
    manifest = {
        "schema_version": "qwen4b_original_chunks_v1", "scope": scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Score exact BTC-submitted chunks; no segmentation/local qrels/quality evaluation.",
        "control_zip_sha256": CONTROL_SHA256,
        "queries_sha256": sha256_file(QUERIES),
        "pilot_sha256": sha256_file(PILOT) if scope == "pilot" else None,
        "queries": len(seen), "chunks": chunks, "unique_doc_ids": len(owners),
        "passage_chars": {"min": int(series.min()), "median": float(series.median()),
                          "p90": float(series.quantile(.9)), "p99": float(series.quantile(.99)),
                          "max": int(series.max())},
        "artifacts": {pairs_path.name: {"bytes": pairs_path.stat().st_size,
                                         "sha256": sha256_file(pairs_path)}},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.scope, args.output)


if __name__ == "__main__":
    main()
