"""Build private Kaggle input shards for a secondary Qwen3 embedding index.

The pack contains only official corpus text already present in the active index.
Offsets are applied locally so Kaggle never needs the large BGE vector/sparse
artifacts. ``row_id`` is the canonical active-index row and must not be changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/indices/new_dataset/index"
DEFAULT_OUTPUT = ROOT / "data/qwen3_embedding_kaggle/input"
QUERY_SOURCE = ROOT / "data/raw/btc_provided/query_translated.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-per-shard", type=int, default=100_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Pilot only: stop after this many chunks. Omit for the production pack.",
    )
    return parser.parse_args()


def write_shard(frame: pd.DataFrame, output_dir: Path, number: int) -> dict:
    frame = frame.sort_values("row_id", kind="stable").reset_index(drop=True)
    if frame.row_id.duplicated().any():
        raise RuntimeError("Duplicate row_id inside output shard")
    path = output_dir / f"chunks_{number:05d}.parquet"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", compression_level=6)
    return {
        "file": path.name,
        "rows": len(frame),
        "row_min": int(frame.row_id.min()),
        "row_max": int(frame.row_id.max()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    index_dir = args.index_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.rows_per_shard <= 0 or (args.max_rows is not None and args.max_rows <= 0):
        raise ValueError("Row limits must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = json.loads((index_dir / "stats.json").read_text(encoding="utf-8"))
    metadata_path = index_dir / "chunks_metadata.parquet"
    metadata = pd.read_parquet(metadata_path, columns=["chunk_id", "doc_id", "start", "end"])
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    if not np.array_equal(metadata.chunk_id.to_numpy(), metadata.row_id.to_numpy()):
        raise RuntimeError("Active chunk_id is not identical to canonical row_id")
    if metadata.duplicated(["doc_id", "start", "end"]).any():
        raise RuntimeError("Duplicate chunk span in active metadata")
    expected_rows = int(stats["total_chunks"])
    if len(metadata) != expected_rows:
        raise RuntimeError(f"Stats/metadata mismatch: {expected_rows:,} != {len(metadata):,}")

    text_paths = sorted((index_dir / "text").glob("*.parquet"))
    if len(text_paths) != int(stats["text_parts"]):
        raise RuntimeError("Unexpected active text shard count")

    manifests: list[dict] = []
    pending: list[pd.DataFrame] = []
    pending_rows = 0
    packed_rows = 0
    shard_number = 0
    source_docs = 0
    empty_chunks = 0

    for source_number, text_path in enumerate(text_paths):
        texts = pd.read_parquet(text_path, columns=["doc_id", "text"])
        if texts.doc_id.duplicated().any():
            raise RuntimeError(f"Duplicate doc_id in {text_path.name}")
        source_docs += len(texts)
        joined = metadata.merge(texts, on="doc_id", how="inner", validate="many_to_one", sort=False)
        if joined.empty:
            continue
        values = joined.text.astype(str).tolist()
        starts = joined.start.to_numpy(dtype=np.int64)
        ends = joined.end.to_numpy(dtype=np.int64)
        bad = (starts < 0) | (ends <= starts) | np.fromiter(
            (end > len(text) for text, end in zip(values, ends, strict=True)),
            dtype=bool,
            count=len(joined),
        )
        if bad.any():
            sample = joined.loc[bad, ["row_id", "doc_id", "start", "end"]].head().to_dict("records")
            raise RuntimeError(f"Invalid source offsets in {text_path.name}: {sample}")
        chunk_text = [
            text[int(start):int(end)]
            for text, start, end in zip(values, starts, ends, strict=True)
        ]
        blank = np.fromiter((not value.strip() for value in chunk_text), dtype=bool, count=len(chunk_text))
        empty_chunks += int(blank.sum())
        frame = joined[["row_id", "chunk_id", "doc_id"]].copy()
        frame["chunk_text"] = chunk_text
        frame["is_blank"] = blank
        frame["row_id"] = frame.row_id.astype(np.int64)
        frame["chunk_id"] = frame.chunk_id.astype(np.int64)
        frame["doc_id"] = frame.doc_id.astype(np.int64)

        if args.max_rows is not None:
            remaining = args.max_rows - packed_rows - pending_rows
            if remaining <= 0:
                break
            frame = frame.iloc[:remaining].copy()
        pending.append(frame)
        pending_rows += len(frame)

        while pending_rows >= args.rows_per_shard:
            combined = pd.concat(pending, ignore_index=True)
            current = combined.iloc[: args.rows_per_shard].copy()
            remainder = combined.iloc[args.rows_per_shard :].copy()
            manifests.append(write_shard(current, output_dir, shard_number))
            shard_number += 1
            packed_rows += len(current)
            pending = [remainder] if not remainder.empty else []
            pending_rows = len(remainder)
            print(f"packed {packed_rows:,} chunks -> {shard_number} shards", flush=True)

        if args.max_rows is not None and packed_rows + pending_rows >= args.max_rows:
            break

    if pending_rows:
        current = pd.concat(pending, ignore_index=True)
        manifests.append(write_shard(current, output_dir, shard_number))
        packed_rows += len(current)
        print(f"packed {packed_rows:,} chunks -> {shard_number + 1} shards", flush=True)

    queries = pd.read_parquet(QUERY_SOURCE, columns=["id", "query", "query_en", "query_zh"])
    if len(queries) != 1_200 or queries.id.duplicated().any():
        raise RuntimeError("Expected exactly 1,200 unique competition queries")
    query_path = output_dir / "queries.parquet"
    queries.to_parquet(query_path, index=False, compression="zstd")

    expected_packed = min(expected_rows, args.max_rows or expected_rows)
    if packed_rows != expected_packed:
        raise RuntimeError(f"Packed row mismatch: {packed_rows:,} != {expected_packed:,}")
    total_bytes = sum(item["bytes"] for item in manifests)
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Private Kaggle input for secondary Qwen3 chunk embedding",
        "active_index": str(index_dir),
        "active_index_stats": stats,
        "chunks_metadata_sha256": sha256(metadata_path),
        "query_source_sha256": sha256(QUERY_SOURCE),
        "rows": packed_rows,
        "full_corpus": packed_rows == expected_rows,
        "rows_per_shard": args.rows_per_shard,
        "shards": manifests,
        "shard_bytes": total_bytes,
        "queries": {
            "file": query_path.name,
            "rows": len(queries),
            "bytes": query_path.stat().st_size,
            "sha256": sha256(query_path),
        },
        "invariants": {
            "row_id_equals_chunk_id": True,
            "document_instruction": None,
            "query_languages": ["vi", "en", "zh"],
            "empty_chunks": empty_chunks,
        },
    }
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": packed_rows,
        "shards": len(manifests),
        "bytes": total_bytes,
        "bytes_per_row": round(total_bytes / packed_rows, 2),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
