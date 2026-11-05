"""Transactionally merge an embedded ZIP artifact into the active index.

Only documents present in the official BTC corpus and absent from the active
index are retained. Existing shard files are hard-linked into a staging index,
new shards are appended with contiguous chunk IDs, and all global search
artifacts are rebuilt and validated before an atomic directory swap.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = PROJECT_ROOT / "data" / "indices" / "new_dataset"
ACTIVE_INDEX = INDEX_ROOT / "index"
OFFICIAL_PATH = (
    PROJECT_ROOT / "data" / "raw" / "btc_provided" / "links_corpus_vi_en_zh.parquet"
)
DEFAULT_SOURCE = INDEX_ROOT / (
    "index_backup_unembedded_samples_0-100_3-20261017T081518Z-1-001.zip"
)
STAGING_INDEX = INDEX_ROOT / "index_merge_artifact_staging"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def zip_root(zf: zipfile.ZipFile) -> str:
    names = [name for name in zf.namelist() if name.endswith("/stats.json")]
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one stats.json in ZIP, found {names}")
    return names[0].rsplit("/", 1)[0]


def zip_npy(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    return np.load(BytesIO(zf.read(member)), allow_pickle=False)


def read_custom_sparse_bytes(raw: bytes, rows: int, sparse_dim: int) -> sp.csr_matrix:
    with np.load(BytesIO(raw), allow_pickle=False) as packed:
        values = packed["vals"].astype(np.float32)
        indices = packed["ids"].astype(np.int32, copy=False)
        indptr = packed["ptr"].astype(np.int64, copy=False)
    if len(indptr) != rows + 1:
        raise RuntimeError("Sparse row pointer does not match source row count")
    return sp.csr_matrix((values, indices, indptr), shape=(rows, sparse_dim))


def save_custom_sparse(path: Path, matrix: sp.csr_matrix) -> None:
    matrix = matrix.tocsr()
    np.savez_compressed(
        path,
        vals=matrix.data.astype(np.float32, copy=False),
        ids=matrix.indices.astype(np.int32, copy=False),
        ptr=matrix.indptr.astype(np.int64, copy=False),
    )


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_staging() -> None:
    if not ACTIVE_INDEX.is_dir():
        raise FileNotFoundError(f"Active index not found: {ACTIVE_INDEX}")
    if STAGING_INDEX.exists():
        raise FileExistsError(
            f"Staging path already exists; inspect it before retrying: {STAGING_INDEX}"
        )
    for name in ("emb_dense", "chunks", "text"):
        (STAGING_INDEX / name).mkdir(parents=True, exist_ok=False)
        for source in sorted((ACTIVE_INDEX / name).glob("*")):
            if source.is_file():
                link_or_copy(source, STAGING_INDEX / name / source.name)


def load_active_state() -> tuple[dict, set[int], int, int, int, int]:
    stats = json.loads((ACTIVE_INDEX / "stats.json").read_text(encoding="utf-8"))
    metadata = pq.read_table(
        ACTIVE_INDEX / "chunks_metadata.parquet", columns=["chunk_id", "doc_id"]
    )
    chunk_ids = np.asarray(metadata["chunk_id"], dtype=np.int64)
    active_docs = set(map(int, metadata["doc_id"].to_pylist()))
    total_chunks = len(metadata)
    if not np.array_equal(chunk_ids, np.arange(total_chunks, dtype=np.int64)):
        raise RuntimeError("Active chunk IDs are not contiguous and ordered")
    if len(active_docs) != int(stats["total_docs"]):
        raise RuntimeError("Active metadata document count disagrees with stats.json")
    if total_chunks != int(stats["total_chunks"]):
        raise RuntimeError("Active metadata chunk count disagrees with stats.json")
    emb_parts = len(list((ACTIVE_INDEX / "emb_dense").glob("ids_*.npy")))
    chunk_parts = len(list((ACTIVE_INDEX / "chunks").glob("part_*.parquet")))
    text_parts = len(list((ACTIVE_INDEX / "text").glob("part_*.parquet")))
    if emb_parts != chunk_parts:
        raise RuntimeError("Active dense and chunk shard counts differ")
    return stats, active_docs, total_chunks, emb_parts, text_parts, len(metadata)


def audit_source(
    source_zip: Path, official_ids: set[int], active_docs: set[int]
) -> tuple[dict, str, pa.Table, set[int], set[int], set[int]]:
    with zipfile.ZipFile(source_zip) as zf:
        root = zip_root(zf)
        source_stats = json.loads(zf.read(f"{root}/stats.json").decode("utf-8"))
        chunk_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{root}/chunks/part_") and name.endswith(".parquet")
        )
        tables = [
            pq.read_table(
                BytesIO(zf.read(name)), columns=["chunk_id", "doc_id", "start", "end"]
            )
            for name in chunk_members
        ]
    chunks = pa.concat_tables(tables)
    source_docs = set(map(int, chunks["doc_id"].to_pylist()))
    outside = source_docs - official_ids
    overlap = source_docs & active_docs
    keep_docs = source_docs - outside - overlap
    if not keep_docs:
        raise RuntimeError("Artifact has no new official documents to merge")
    return source_stats, root, chunks, source_docs, outside, overlap


def append_source_shards(
    source_zip: Path,
    root: str,
    source_chunks: pa.Table,
    keep_docs: set[int],
    chunk_offset: int,
    output_part: int,
    dense_dim: int,
    sparse_dim: int,
) -> tuple[int, int]:
    with zipfile.ZipFile(source_zip) as zf:
        id_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{root}/emb_dense/ids_") and name.endswith(".npy")
        )
        source_id_parts = [zip_npy(zf, name) for name in id_members]
        source_ids = np.concatenate(source_id_parts)
        chunk_values = np.asarray(source_chunks["chunk_id"], dtype=np.int64)
        if not np.array_equal(source_ids, chunk_values):
            raise RuntimeError("Source vector and chunk order differ")

        cursor = 0
        added_chunks = 0
        for source_index, source_ids_part in enumerate(source_id_parts):
            rows = len(source_ids_part)
            part = source_chunks.slice(cursor, rows)
            doc_ids = np.asarray(part["doc_id"], dtype=np.int64)
            keep = np.fromiter(
                (int(doc_id) in keep_docs for doc_id in doc_ids),
                dtype=np.bool_,
                count=rows,
            )
            kept = int(keep.sum())
            cursor += rows
            if kept == 0:
                continue

            tag = f"{output_part:04d}"
            source_tag = f"{source_index:04d}"
            new_ids = np.arange(chunk_offset, chunk_offset + kept, dtype=np.int64)
            np.save(STAGING_INDEX / "emb_dense" / f"ids_{tag}.npy", new_ids)

            vectors = zip_npy(zf, f"{root}/emb_dense/vec_{source_tag}.npy")
            if vectors.shape != (rows, dense_dim):
                raise RuntimeError(f"Unexpected vector shape: {vectors.shape}")
            np.save(STAGING_INDEX / "emb_dense" / f"vec_{tag}.npy", vectors[keep])

            sparse = read_custom_sparse_bytes(
                zf.read(f"{root}/emb_dense/sp_{source_tag}.npz"), rows, sparse_dim
            )
            save_custom_sparse(STAGING_INDEX / "emb_dense" / f"sp_{tag}.npz", sparse[keep])

            kept_indices = np.flatnonzero(keep)
            filtered = part.take(pa.array(kept_indices, type=pa.int64())).to_pandas()
            filtered["chunk_id"] = new_ids
            filtered.to_parquet(
                STAGING_INDEX / "chunks" / f"part_{tag}.parquet",
                index=False,
                compression="snappy",
            )
            print(f"  source part {source_tag}: kept {kept:,}/{rows:,} chunks")
            chunk_offset += kept
            added_chunks += kept
            output_part += 1
            del vectors, sparse, filtered, new_ids
            gc.collect()

        if cursor != len(source_chunks):
            raise RuntimeError("Not all source chunks were consumed")
    return added_chunks, output_part


def append_source_text(
    source_zip: Path, root: str, keep_docs: set[int], output_part: int
) -> int:
    found: list[int] = []
    schema = pa.schema([pa.field("doc_id", pa.int64()), pa.field("text", pa.large_string())])
    with zipfile.ZipFile(source_zip) as zf:
        members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{root}/text/part_") and name.endswith(".parquet")
        )
        for member in members:
            table = pq.read_table(BytesIO(zf.read(member)), columns=["doc_id", "text"])
            ids = np.asarray(table["doc_id"], dtype=np.int64)
            keep = np.fromiter(
                (int(doc_id) in keep_docs for doc_id in ids),
                dtype=np.bool_,
                count=len(ids),
            )
            if not keep.any():
                continue
            filtered = table.filter(pa.array(keep)).cast(schema)
            pq.write_table(
                filtered,
                STAGING_INDEX / "text" / f"part_{output_part:04d}.parquet",
                compression="snappy",
            )
            found.extend(map(int, filtered["doc_id"].to_pylist()))
            output_part += 1
    if len(found) != len(set(found)) or set(found) != keep_docs:
        raise RuntimeError("Filtered source text does not exactly match kept documents")
    return output_part


def rebuild_dense_and_metadata(total_chunks: int, dense_dim: int) -> int:
    ids_files = sorted((STAGING_INDEX / "emb_dense").glob("ids_*.npy"))
    chunk_files = sorted((STAGING_INDEX / "chunks").glob("part_*.parquet"))
    if len(ids_files) != len(chunk_files):
        raise RuntimeError("Staging dense and chunk shard counts differ")

    index = faiss.IndexFlatIP(dense_dim)
    all_ids: list[np.ndarray] = []
    chunk_tables: list[pa.Table] = []
    for number, (ids_path, chunks_path) in enumerate(zip(ids_files, chunk_files), 1):
        ids = np.load(ids_path, allow_pickle=False)
        vectors = np.load(
            ids_path.with_name(ids_path.name.replace("ids_", "vec_")),
            mmap_mode="r",
            allow_pickle=False,
        )
        chunks = pq.read_table(chunks_path, columns=["chunk_id", "doc_id", "start", "end"])
        chunk_ids = np.asarray(chunks["chunk_id"], dtype=np.int64)
        if vectors.shape != (len(ids), dense_dim) or not np.array_equal(ids, chunk_ids):
            raise RuntimeError(f"Dense/chunk mismatch in shard {ids_path.name}")
        index.add(np.asarray(vectors, dtype=np.float32))
        all_ids.append(ids)
        chunk_tables.append(chunks)
        print(f"  FAISS {number:02d}/{len(ids_files):02d}: {len(ids):,} rows")
        del vectors, chunks, chunk_ids
        gc.collect()

    concatenated = np.concatenate(all_ids)
    if len(concatenated) != total_chunks or not np.array_equal(
        concatenated, np.arange(total_chunks, dtype=np.int64)
    ):
        raise RuntimeError("Final chunk IDs are not contiguous and ordered")
    metadata = pa.concat_tables(chunk_tables)
    if not np.array_equal(concatenated, np.asarray(metadata["chunk_id"], dtype=np.int64)):
        raise RuntimeError("Global metadata order differs from dense vector order")
    np.save(STAGING_INDEX / "chunk_ids.npy", concatenated)
    pq.write_table(metadata, STAGING_INDEX / "chunks_metadata.parquet", compression="snappy")
    if index.ntotal != total_chunks:
        raise RuntimeError("FAISS row count mismatch")
    faiss.write_index(index, str(STAGING_INDEX / "faiss_index_flat_ip.bin"))
    del index, all_ids, chunk_tables, metadata, concatenated
    gc.collect()
    return len(ids_files)


def rebuild_sparse(total_chunks: int, sparse_dim: int) -> None:
    sparse_files = sorted((STAGING_INDEX / "emb_dense").glob("sp_*.npz"))
    ids_files = sorted((STAGING_INDEX / "emb_dense").glob("ids_*.npy"))
    row_counts: list[int] = []
    nnz_counts: list[int] = []
    total_nnz = 0
    for sparse_path, ids_path in zip(sparse_files, ids_files):
        rows = len(np.load(ids_path, mmap_mode="r", allow_pickle=False))
        with np.load(sparse_path, allow_pickle=False) as packed:
            ptr = packed["ptr"]
            if len(ptr) != rows + 1:
                raise RuntimeError(f"Sparse pointer mismatch in {sparse_path.name}")
            nnz = int(ptr[-1])
        row_counts.append(rows)
        nnz_counts.append(nnz)
        total_nnz += nnz
    if len(sparse_files) != len(ids_files) or sum(row_counts) != total_chunks:
        raise RuntimeError("Sparse shard count or row total mismatch")

    values = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(total_chunks + 1, dtype=np.int64)
    indptr[0] = 0
    row_cursor = 0
    nnz_cursor = 0
    for number, (path, rows, nnz) in enumerate(
        zip(sparse_files, row_counts, nnz_counts), 1
    ):
        with np.load(path, allow_pickle=False) as packed:
            values[nnz_cursor : nnz_cursor + nnz] = packed["vals"]
            indices[nnz_cursor : nnz_cursor + nnz] = packed["ids"]
            indptr[row_cursor + 1 : row_cursor + rows + 1] = packed["ptr"][1:] + nnz_cursor
        row_cursor += rows
        nnz_cursor += nnz
        print(f"  sparse {number:02d}/{len(sparse_files):02d}: {rows:,} rows")
    matrix = sp.csr_matrix((values, indices, indptr), shape=(total_chunks, sparse_dim))
    sp.save_npz(STAGING_INDEX / "sparse_csr.npz", matrix, compressed=True)
    del matrix, values, indices, indptr
    gc.collect()


def validate(
    expected_docs: set[int], total_chunks: int, dense_dim: int, sparse_dim: int
) -> None:
    metadata = pq.read_table(STAGING_INDEX / "chunks_metadata.parquet")
    chunk_ids = np.asarray(metadata["chunk_id"], dtype=np.int64)
    docs = set(map(int, metadata["doc_id"].to_pylist()))
    if len(metadata) != total_chunks or docs != expected_docs:
        raise RuntimeError("Final metadata does not match expected rows/documents")
    if not np.array_equal(chunk_ids, np.arange(total_chunks, dtype=np.int64)):
        raise RuntimeError("Final metadata chunk IDs are not contiguous")

    text_ids: list[int] = []
    for path in sorted((STAGING_INDEX / "text").glob("part_*.parquet")):
        text_ids.extend(map(int, pq.read_table(path, columns=["doc_id"])["doc_id"].to_pylist()))
    if len(text_ids) != len(expected_docs) or set(text_ids) != expected_docs:
        raise RuntimeError("Final text store does not exactly match expected documents")

    index = faiss.read_index(
        str(STAGING_INDEX / "faiss_index_flat_ip.bin"), faiss.IO_FLAG_MMAP
    )
    if index.ntotal != total_chunks or index.d != dense_dim:
        raise RuntimeError("Final FAISS shape mismatch")
    sparse = sp.load_npz(STAGING_INDEX / "sparse_csr.npz")
    if sparse.shape != (total_chunks, sparse_dim):
        raise RuntimeError("Final sparse matrix shape mismatch")
    del index, sparse
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--no-swap", action="store_true")
    args = parser.parse_args()
    source_zip = args.source.resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)

    started = time.time()
    official_values = pq.read_table(OFFICIAL_PATH, columns=["id"])["id"].to_pylist()
    official_ids = set(map(int, official_values))
    if len(official_values) != len(official_ids):
        raise RuntimeError("Official corpus contains duplicate IDs")
    stats, active_docs, chunk_offset, output_part, text_part, _ = load_active_state()
    dense_dim = int(stats["dense_dim"])
    sparse_dim = int(stats["vocab_size"])
    source_stats, root, source_chunks, source_docs, outside, overlap = audit_source(
        source_zip, official_ids, active_docs
    )
    keep_docs = source_docs - outside - overlap
    print(f"Official corpus: {len(official_ids):,} docs")
    print(f"Active index: {len(active_docs):,} docs / {chunk_offset:,} chunks")
    print(f"Artifact: {len(source_docs):,} docs / {len(source_chunks):,} chunks")
    print(f"Outside official corpus: {len(outside):,} docs")
    print(f"Already in active index: {len(overlap):,} docs")
    print(f"New official docs to merge: {len(keep_docs):,}")

    prepare_staging()
    added_chunks, final_parts = append_source_shards(
        source_zip,
        root,
        source_chunks,
        keep_docs,
        chunk_offset,
        output_part,
        dense_dim,
        sparse_dim,
    )
    final_text_parts = append_source_text(source_zip, root, keep_docs, text_part)
    total_chunks = chunk_offset + added_chunks
    final_docs = active_docs | keep_docs
    built_parts = rebuild_dense_and_metadata(total_chunks, dense_dim)
    if built_parts != final_parts:
        raise RuntimeError("Unexpected final dense shard count")
    rebuild_sparse(total_chunks, sparse_dim)

    now = datetime.now().astimezone().isoformat()
    new_stats = {
        **stats,
        "index_name": "ViBioMIR_Consolidated_Index_Clean",
        "total_docs": len(final_docs),
        "total_chunks": total_chunks,
        "emb_parts": final_parts,
        "chunks_parts": final_parts,
        "text_parts": final_text_parts,
        "updated_at": now,
        "source": f"{stats.get('source', '')}; merged {source_zip.name}",
        "latest_added_valid_docs": len(keep_docs),
        "latest_added_valid_chunks": added_chunks,
        "latest_filtered_outside_docs": len(outside),
        "latest_skipped_overlap_docs": len(overlap),
    }
    (STAGING_INDEX / "stats.json").write_text(
        json.dumps(new_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "built_at": now,
        "official_list": str(OFFICIAL_PATH.relative_to(PROJECT_ROOT)),
        "official_list_sha256": sha256_file(OFFICIAL_PATH),
        "source_artifact": str(source_zip.relative_to(PROJECT_ROOT)),
        "source_artifact_sha256": sha256_file(source_zip),
        "source_stats": source_stats,
        "previous_index_stats": stats,
        "audit": {
            "artifact_docs": len(source_docs),
            "artifact_chunks": len(source_chunks),
            "outside_official_docs": len(outside),
            "overlap_active_docs": len(overlap),
            "merged_docs": len(keep_docs),
            "merged_chunks": added_chunks,
        },
        "result": new_stats,
    }
    (STAGING_INDEX / "rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validate(final_docs, total_chunks, dense_dim, sparse_dim)

    backup = None
    if not args.no_swap:
        backup = INDEX_ROOT / (
            "index_backup_pre_artifact3_merge_" + datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        ACTIVE_INDEX.rename(backup)
        try:
            STAGING_INDEX.rename(ACTIVE_INDEX)
        except Exception:
            if not ACTIVE_INDEX.exists() and backup.exists():
                backup.rename(ACTIVE_INDEX)
            raise

    print("=" * 80)
    print(f"MERGE COMPLETE in {(time.time() - started) / 60:.1f} minutes")
    print(f"Final index: {len(final_docs):,} docs / {total_chunks:,} chunks")
    print(f"Filtered outside corpus: {len(outside):,} docs")
    print(f"Skipped existing overlap: {len(overlap):,} docs")
    print(f"Recoverable backup: {backup if backup else 'no swap requested'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
