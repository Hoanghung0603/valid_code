"""Package exact submitted chunks as query-window pairs for P2a reranking.

Windows are scoring views only. The eventual submission may retain or remove
original chunks, but must never replace their text with these windows.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_p1_qwen4b_top195_full.zip"
CONTROL_SHA256 = "785637cb462f014f1bf35c1c994372a2561bac924ef416c0ccaea0a6b10b0c28"
QUERIES = ROOT / "data/raw/btc_provided/query.parquet"
PILOT = ROOT / "data/audits/queryspec_pilot_v6_final/pilot_manifest.parquet"
OUTPUT_ROOT = ROOT / "data/kaggle_inputs"

PAIR_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("query_id", pa.int64()), ("doc_id", pa.int64()),
    ("passage_rank", pa.int16()), ("chunk_ordinal", pa.int32()),
    ("window_start", pa.int32()), ("window_end", pa.int32()),
    ("chunk_sha256", pa.string()), ("window_sha256", pa.string()),
    ("query_text", pa.string()), ("passage_text", pa.string()),
    ("language", pa.string()),
])
CHUNK_SCHEMA = pa.schema([
    ("query_id", pa.int64()), ("chunk_ordinal", pa.int32()),
    ("doc_id", pa.int64()), ("chunk_sha256", pa.string()),
    ("chunk_chars", pa.int32()), ("language", pa.string()),
    ("window_count", pa.int16()), ("window_coverage", pa.float32()),
])


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def script_language(value: str) -> str:
    sample = value[:6000]
    visible = sum(not char.isspace() for char in sample)
    chinese = sum("\u4e00" <= char <= "\u9fff" for char in sample)
    if visible and chinese / visible >= 0.05:
        return "zh"
    if any(char in "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ" for char in sample):
        return "vi"
    return "en_or_plain_latin"


def window_ranges(length: int, width: int, max_windows: int) -> list[tuple[int, int]]:
    if length <= 0 or width <= 0 or max_windows <= 0:
        raise ValueError("Positive text length, width, and max_windows required")
    if length <= width:
        return [(0, length)]
    count = min(max_windows, math.ceil(length / width))
    if count == 1:
        return [(0, width)]
    starts = sorted({round(i * (length - width) / (count - 1)) for i in range(count)})
    return [(start, start + width) for start in starts]


def coverage_fraction(ranges: list[tuple[int, int]], length: int) -> float:
    covered = 0
    last_end = 0
    for start, end in sorted(ranges):
        covered += max(0, end - max(start, last_end))
        last_end = max(last_end, end)
    return covered / length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--latin-window-chars", type=int, default=1800)
    parser.add_argument("--zh-window-chars", type=int, default=900)
    parser.add_argument("--max-windows", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.latin_window_chars, args.zh_window_chars, args.max_windows) < 1:
        raise ValueError("Window settings must be positive")
    output = args.output_dir or OUTPUT_ROOT / f"p2a_chunk_{args.scope}_v1"
    output = output.resolve()
    pair_path = output / "p2a_pairs.parquet"
    chunk_path = output / "p2a_chunks.parquet"
    manifest_path = output / "p2a_manifest.json"
    zip_path = output.with_suffix(".zip")
    if any(path.exists() for path in (pair_path, chunk_path, manifest_path, zip_path)):
        raise FileExistsError(f"P2a pack already exists; choose another output directory: {output}")
    if sha256_file(args.control) != CONTROL_SHA256:
        raise RuntimeError("P2a control ZIP differs from the BTC-scored P1 top195 submission")

    official = pd.read_parquet(args.queries, columns=["id", "query"])
    if len(official) != 1200 or official.id.duplicated().any():
        raise RuntimeError("Expected 1,200 unique official queries")
    query_map = dict(zip(official.id.astype(int), official["query"].astype(str), strict=True))
    if args.scope == "pilot":
        pilot = pd.read_parquet(args.pilot, columns=["id"])
        if len(pilot) != 120 or pilot.id.duplicated().any():
            raise RuntimeError("Expected the fixed 120-query P1 pilot split")
        wanted = set(pilot.id.astype(int))
    else:
        wanted = set(query_map)
    if not wanted.issubset(query_map):
        raise RuntimeError("Pilot IDs differ from official query IDs")

    output.mkdir(parents=True, exist_ok=True)
    pair_writer = pq.ParquetWriter(pair_path, PAIR_SCHEMA, compression="zstd")
    chunk_writer = pq.ParquetWriter(chunk_path, CHUNK_SCHEMA, compression="zstd")
    seen: set[int] = set()
    chunk_count = pair_count = full_coverage = 0
    language_counts: Counter[str] = Counter()
    coverage_values: list[float] = []
    try:
        with zipfile.ZipFile(args.control) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Invalid P2a control ZIP layout/CRC")
            with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
                for row in iter_json_array(stream):
                    qid = int(row["id"])
                    if qid not in wanted:
                        continue
                    if qid in seen:
                        raise RuntimeError(f"Duplicate query ID {qid}")
                    seen.add(qid)
                    query = query_map[qid]
                    pair_records: list[dict] = []
                    chunk_records: list[dict] = []
                    for ordinal, chunk in enumerate(row["relevant_chunks"]):
                        doc_id = int(chunk["doc_id"])
                        text = chunk["chunk_text"]
                        if not isinstance(text, str) or not text.strip():
                            raise RuntimeError(f"Empty/non-text chunk at query={qid}, ordinal={ordinal}")
                        language = script_language(text)
                        width = args.zh_window_chars if language == "zh" else args.latin_window_chars
                        spans = window_ranges(len(text), width, args.max_windows)
                        coverage = coverage_fraction(spans, len(text))
                        chunk_hash = digest(text)
                        for rank, (start, end) in enumerate(spans, 1):
                            window = text[start:end]
                            pair_records.append({
                                "pair_id": f"{qid}:{ordinal}:{rank}", "query_id": qid,
                                "doc_id": doc_id, "passage_rank": rank,
                                "chunk_ordinal": ordinal, "window_start": start,
                                "window_end": end, "chunk_sha256": chunk_hash,
                                "window_sha256": digest(window), "query_text": query,
                                "passage_text": window, "language": language,
                            })
                        chunk_records.append({
                            "query_id": qid, "chunk_ordinal": ordinal, "doc_id": doc_id,
                            "chunk_sha256": chunk_hash, "chunk_chars": len(text),
                            "language": language, "window_count": len(spans),
                            "window_coverage": coverage,
                        })
                        chunk_count += 1
                        pair_count += len(spans)
                        full_coverage += coverage == 1.0
                        language_counts[language] += 1
                        coverage_values.append(coverage)
                    if not pair_records:
                        raise RuntimeError(f"Query {qid} has no chunks")
                    pair_writer.write_table(pa.Table.from_pylist(pair_records, schema=PAIR_SCHEMA))
                    chunk_writer.write_table(pa.Table.from_pylist(chunk_records, schema=CHUNK_SCHEMA))
                    if len(seen) % 20 == 0:
                        print(f"Prepared {len(seen)}/{len(wanted)} queries, {chunk_count} chunks, {pair_count} windows", flush=True)
    finally:
        pair_writer.close()
        chunk_writer.close()
    if seen != wanted or (args.scope == "full" and chunk_count != 130_108):
        raise RuntimeError(f"Wrong query/chunk coverage: queries={len(seen)}/{len(wanted)}, chunks={chunk_count}")

    manifest = {
        "schema_version": "p2a_chunk_windows_v1", "scope": args.scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Score existing P1-submitted chunks only; windows are never submission text; no local relevance/qrels evaluation.",
        "control_zip_sha256": CONTROL_SHA256,
        "official_queries_sha256": sha256_file(args.queries),
        "pilot_manifest_sha256": sha256_file(args.pilot) if args.scope == "pilot" else None,
        "window_policy": {"latin_chars": args.latin_window_chars, "zh_chars": args.zh_window_chars,
                          "max_windows": args.max_windows, "selection": "uniform, deterministic"},
        "validation": {"queries": len(seen), "chunks": chunk_count, "pairs": pair_count,
                       "full_coverage_chunks": full_coverage,
                       "full_coverage_fraction": full_coverage / chunk_count,
                       "window_coverage_p10": float(np.quantile(coverage_values, 0.1)),
                       "language_chunks": dict(language_counts)},
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (pair_path, chunk_path)
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (pair_path, chunk_path, manifest_path):
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path) as archive:
        if set(archive.namelist()) != {pair_path.name, chunk_path.name, manifest_path.name} or archive.testzip() is not None:
            raise RuntimeError("P2a input ZIP layout/CRC failed")
    print(json.dumps({**manifest["validation"], "zip": str(zip_path),
                      "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha256_file(zip_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
