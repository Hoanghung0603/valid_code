"""Validate downloaded Kaggle Qwen3 shards and build a separate FAISS FlatIP index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from streaming_index_rebuild import validate_flat, write_flat


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_INDEX = ROOT / "data/indices/new_dataset/index"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        type=Path,
        nargs="+",
        help="One complete Kaggle output directory per partition",
    )
    parser.add_argument("--active-index", type=Path, default=ACTIVE_INDEX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = [path.resolve() for path in args.artifact]
    active_index = args.active_index.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    bundles = []
    for artifact in artifacts:
        manifest_path = artifact / "output_manifest.json"
        run_config_path = artifact / "run_config.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        for key, value in run_config.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Manifest/run-config mismatch for {key} in {artifact}")
        bundles.append((artifact, manifest_path, manifest, run_config))

    reference = bundles[0][2]
    common_keys = (
        "format_version", "model", "revision", "output_dim", "max_length",
        "query_instruction", "document_instruction", "pooling", "input_manifest_sha256",
    )
    for artifact, _, manifest, _ in bundles[1:]:
        for key in common_keys:
            if manifest.get(key) != reference.get(key):
                raise RuntimeError(f"Cross-partition config mismatch for {key}: {artifact}")
    partition_count = int(reference.get("partition_count", 1))
    partition_indices = {int(manifest.get("partition_index", 0)) for _, _, manifest, _ in bundles}
    if len(bundles) != partition_count or partition_indices != set(range(partition_count)):
        raise RuntimeError(
            f"Expected partitions 0..{partition_count - 1}, got {sorted(partition_indices)}"
        )
    if reference.get("pooling") != "last_token_then_slice_then_l2_normalize":
        raise RuntimeError("Unexpected pooling policy")
    if reference.get("document_instruction") is not None:
        raise RuntimeError("Documents must have no query instruction")

    active_stats = json.loads((active_index / "stats.json").read_text(encoding="utf-8"))
    rows = int(active_stats["total_chunks"])
    dim = int(reference["output_dim"])
    artifact_rows = sum(int(manifest["rows"]) for _, _, manifest, _ in bundles)
    if artifact_rows != rows:
        raise RuntimeError(f"Partition/active index row mismatch: {artifact_rows:,} != {rows:,}")
    if dim != 1024:
        raise RuntimeError(f"First secondary index must use 1024 dimensions, got {dim}")

    staging_path = output / "vectors.float16.mmap"
    seen_path = output / "seen.uint8.mmap"
    staging = np.memmap(staging_path, mode="w+", dtype=np.float16, shape=(rows, dim))
    seen = np.memmap(seen_path, mode="w+", dtype=np.uint8, shape=(rows,))
    seen[:] = 0
    checked_shards = []
    total_shards = sum(len(manifest["shards"]) for _, _, manifest, _ in bundles)
    number = 0
    for artifact, _, manifest, _ in bundles:
        partition_index = int(manifest.get("partition_index", 0))
        for item in manifest["shards"]:
            number += 1
            source_name = item["source"]
            tag = Path(source_name).stem.removeprefix("chunks_")
            emb_path = artifact / f"emb_{tag}.npy"
            row_path = artifact / f"rows_{tag}.npy"
            if sha256(emb_path) != item["embedding_sha256"] or sha256(row_path) != item["rows_sha256"]:
                raise RuntimeError(f"Downloaded shard checksum mismatch: {tag}")
            vectors = np.load(emb_path, mmap_mode="r", allow_pickle=False)
            row_ids = np.load(row_path, allow_pickle=False)
            if vectors.shape != (len(row_ids), dim) or vectors.dtype != np.float16:
                raise RuntimeError(f"Unexpected vector layout in {emb_path.name}: {vectors.shape}/{vectors.dtype}")
            if row_ids.dtype.kind not in "iu" or (row_ids < 0).any() or (row_ids >= rows).any():
                raise RuntimeError(f"Invalid row IDs in {row_path.name}")
            if len(np.unique(row_ids)) != len(row_ids) or seen[row_ids].any():
                raise RuntimeError(f"Duplicate row IDs in/across shard {tag}")
            zero_count = 0
            for start in range(0, len(row_ids), 4096):
                block = np.asarray(vectors[start : start + 4096], dtype=np.float32)
                if not np.isfinite(block).all():
                    raise RuntimeError(f"Non-finite vectors in {emb_path.name}")
                norms = np.linalg.norm(block, axis=1)
                normalized = np.isclose(norms, 1.0, rtol=2e-3, atol=2e-3)
                zeros = norms == 0
                zero_count += int(zeros.sum())
                if not np.all(normalized | zeros):
                    raise RuntimeError(f"Non-unit embeddings in {emb_path.name}")
            if zero_count != int(item.get("blank_rows", 0)):
                raise RuntimeError(f"Blank/zero-vector count mismatch in {emb_path.name}")
            staging[row_ids] = vectors
            seen[row_ids] = 1
            checked_shards.append({
                "partition_index": partition_index,
                "tag": tag,
                "rows": len(row_ids),
                "embedding_sha256": item["embedding_sha256"],
            })
            print(f"validated {number}/{total_shards}: {len(row_ids):,} rows", flush=True)
    staging.flush()
    seen.flush()
    if int(seen.sum()) != rows or not bool(seen.all()):
        raise RuntimeError(f"Incomplete row coverage: {int(seen.sum()):,}/{rows:,}")

    def vector_blocks():
        for start in range(0, rows, 4096):
            yield np.asarray(staging[start : start + 4096], dtype=np.float32)

    faiss_path = output / "faiss_index_flat_ip.bin"
    payload_sha256 = write_flat(faiss_path, rows, dim, vector_blocks())
    validate_flat(faiss_path, rows, dim, payload_sha256)
    np.save(output / "chunk_ids.npy", np.arange(rows, dtype=np.int64), allow_pickle=False)
    query_sources = [artifact / "query_embeddings.npz" for artifact, _, _, _ in bundles]
    with np.load(query_sources[0], allow_pickle=False) as first:
        reference_queries = {key: first[key].copy() for key in first.files}
    for source in query_sources[1:]:
        with np.load(source, allow_pickle=False) as candidate:
            if set(candidate.files) != set(reference_queries):
                raise RuntimeError(f"Query embedding keys differ across partitions: {source}")
            for key, expected in reference_queries.items():
                actual = candidate[key]
                if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-3, atol=1e-3):
                    raise RuntimeError(f"Query embeddings differ across partitions for {key}: {source}")
    shutil.copy2(query_sources[0], output / "query_embeddings.npz")

    result = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": reference["model"],
        "revision": reference["revision"],
        "rows": rows,
        "dim": dim,
        "max_length": reference["max_length"],
        "pooling": reference["pooling"],
        "query_instruction": reference["query_instruction"],
        "document_instruction": None,
        "active_index_stats_sha256": sha256(active_index / "stats.json"),
        "active_chunks_metadata_sha256": sha256(active_index / "chunks_metadata.parquet"),
        "partitions": partition_count,
        "kaggle_output_manifest_sha256": [
            sha256(manifest_path) for _, manifest_path, _, _ in bundles
        ],
        "query_embeddings_sha256": sha256(output / "query_embeddings.npz"),
        "faiss_payload_sha256": payload_sha256,
        "faiss_file_sha256": sha256(faiss_path),
        "shards": checked_shards,
        "retrieval_contract": "Use active index chunks_metadata/text; row i is active chunk_id i",
    }
    (output / "index_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del staging, seen
    if not args.keep_staging:
        staging_path.unlink()
        seen_path.unlink()
    print(json.dumps({
        "status": "complete",
        "model": result["model"],
        "rows": rows,
        "dim": dim,
        "faiss_bytes": faiss_path.stat().st_size,
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
