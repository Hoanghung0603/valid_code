"""Rebuild the consolidated ViBioMIR index from valid existing vectors.

This script performs one transactional rebuild:
1. Filters every current index row by the current BTC official doc-id list.
2. Appends the fully embedded 99,448-document artifact that was never merged.
3. Reassigns contiguous chunk ids and rebuilds FAISS, sparse CSR, and metadata.
4. Validates all cross-artifact invariants.
5. Atomically swaps the validated staging index into place and keeps the old
   index as a timestamped backup.

No embedding model is run. Existing dense and sparse vectors are reused.
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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_ROOT = DATA_DIR / "indices" / "new_dataset"
CURRENT_INDEX = INDEX_ROOT / "index"
STAGING_INDEX = INDEX_ROOT / "index_rebuild_clean_staging"
OFFICIAL_PATH = DATA_DIR / "raw" / "btc_provided" / "links_corpus_vi_en_zh.parquet"
SOURCE_ZIP = INDEX_ROOT / "index_backup_unembedded_samples_0-100-20261014T033450Z-1-001.zip"

DENSE_DIM = 1024
SPARSE_DIM = 250_005

EXPECTED_CURRENT_DOCS = 1_056_901
EXPECTED_CURRENT_VALID_DOCS = 1_016_285
EXPECTED_INVALID_DOCS = 40_616
EXPECTED_INVALID_CHUNKS = 56_609
EXPECTED_SOURCE_DOCS = 99_448
EXPECTED_SOURCE_CHUNKS = 146_128
EXPECTED_FINAL_DOCS = 1_115_733
EXPECTED_FINAL_CHUNKS = 1_590_459


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def zip_root(zf: zipfile.ZipFile) -> str:
    stats = [name for name in zf.namelist() if name.endswith("/stats.json")]
    if len(stats) != 1:
        raise RuntimeError(f"Expected one stats.json in source ZIP, found {stats}")
    return stats[0].rsplit("/", 1)[0]


def zip_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(BytesIO(zf.read(name)), allow_pickle=False)


def extract_zip_member(zf: zipfile.ZipFile, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)


def read_custom_sparse(path: Path, rows: int) -> sp.csr_matrix:
    with np.load(path, allow_pickle=False) as packed:
        # BGE-M3 shard weights are stored as float16 to save disk. SciPy CSR
        # cannot operate on float16, so promote only the parts that need row
        # filtering. Unchanged sparse parts are copied byte-for-byte.
        values = packed["vals"].astype(np.float32)
        indices = packed["ids"]
        indptr = packed["ptr"]
        matrix = sp.csr_matrix((values, indices, indptr), shape=(rows, SPARSE_DIM))
        return matrix


def save_custom_sparse(path: Path, matrix: sp.csr_matrix) -> None:
    matrix = matrix.tocsr()
    np.savez_compressed(
        path,
        vals=matrix.data,
        ids=matrix.indices.astype(np.int32, copy=False),
        ptr=matrix.indptr.astype(np.int64, copy=False),
    )


def load_official_ids() -> tuple[set[int], np.ndarray]:
    table = pq.read_table(OFFICIAL_PATH, columns=["id"])
    values = np.asarray(table["id"], dtype=np.int64)
    if len(values) != len(np.unique(values)):
        raise RuntimeError("Official BTC doc-id list contains duplicates")
    return set(values.tolist()), np.sort(values)


def prepare_staging() -> None:
    if not CURRENT_INDEX.is_dir():
        raise FileNotFoundError(f"Current index not found: {CURRENT_INDEX}")
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError(f"Embedded source artifact not found: {SOURCE_ZIP}")
    if STAGING_INDEX.exists():
        raise FileExistsError(
            f"Staging path already exists; inspect it before removing or resuming: {STAGING_INDEX}"
        )
    for child in ("emb_dense", "chunks", "text"):
        (STAGING_INDEX / child).mkdir(parents=True, exist_ok=False)


def build_filtered_parts(
    official_ids: set[int], official_sorted: np.ndarray
) -> tuple[set[int], int, int]:
    current_ids_files = sorted((CURRENT_INDEX / "emb_dense").glob("ids_*.npy"))
    current_chunk_files = sorted((CURRENT_INDEX / "chunks").glob("part_*.parquet"))
    if len(current_ids_files) != len(current_chunk_files):
        raise RuntimeError("Current embedding/chunk part counts do not match")

    current_docs: set[int] = set()
    invalid_docs: set[int] = set()
    invalid_chunks = 0
    global_chunk_offset = 0
    output_part = 0

    print("[1/7] Filtering current dense/sparse/chunk shards...")
    for ids_path, chunks_path in zip(current_ids_files, current_chunk_files):
        ids = np.load(ids_path, allow_pickle=False)
        chunks = pd.read_parquet(
            chunks_path, columns=["chunk_id", "doc_id", "start", "end"]
        )
        if len(ids) != len(chunks):
            raise RuntimeError(f"Row mismatch in {ids_path.name} and {chunks_path.name}")
        if not np.array_equal(ids, chunks["chunk_id"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Chunk-id order mismatch in {ids_path.name}")

        doc_values = chunks["doc_id"].to_numpy(dtype=np.int64)
        keep = np.isin(doc_values, official_sorted, assume_unique=False)
        kept_rows = int(keep.sum())
        removed_rows = len(keep) - kept_rows
        invalid_chunks += removed_rows
        current_docs.update(doc_values[keep].tolist())
        invalid_docs.update(doc_values[~keep].tolist())

        if kept_rows == 0:
            print(f"  - {ids_path.name}: skipped empty valid part")
            continue

        tag = f"{output_part:04d}"
        new_ids = np.arange(
            global_chunk_offset, global_chunk_offset + kept_rows, dtype=np.int64
        )
        np.save(STAGING_INDEX / "emb_dense" / f"ids_{tag}.npy", new_ids)

        source_vec = ids_path.with_name(ids_path.name.replace("ids_", "vec_"))
        target_vec = STAGING_INDEX / "emb_dense" / f"vec_{tag}.npy"
        if kept_rows == len(keep):
            shutil.copy2(source_vec, target_vec)
        else:
            vectors = np.load(source_vec, mmap_mode="r", allow_pickle=False)
            np.save(target_vec, np.asarray(vectors[keep]))
            del vectors

        source_sparse = ids_path.with_name(
            ids_path.name.replace("ids_", "sp_").replace(".npy", ".npz")
        )
        target_sparse = STAGING_INDEX / "emb_dense" / f"sp_{tag}.npz"
        if kept_rows == len(keep):
            shutil.copy2(source_sparse, target_sparse)
        else:
            sparse_matrix = read_custom_sparse(source_sparse, len(keep))
            save_custom_sparse(target_sparse, sparse_matrix[keep])
            del sparse_matrix

        filtered_chunks = chunks.loc[keep, ["chunk_id", "doc_id", "start", "end"]].copy()
        filtered_chunks["chunk_id"] = new_ids
        filtered_chunks.to_parquet(
            STAGING_INDEX / "chunks" / f"part_{tag}.parquet",
            index=False,
            compression="snappy",
        )

        print(
            f"  - {ids_path.name}: kept {kept_rows:,}, removed {removed_rows:,}, "
            f"new range {new_ids[0]:,}..{new_ids[-1]:,}"
        )
        global_chunk_offset += kept_rows
        output_part += 1
        del ids, chunks, keep, new_ids, filtered_chunks
        gc.collect()

    if len(current_docs | invalid_docs) != EXPECTED_CURRENT_DOCS:
        raise RuntimeError("Unexpected current document count")
    if len(current_docs) != EXPECTED_CURRENT_VALID_DOCS:
        raise RuntimeError(f"Unexpected current valid docs: {len(current_docs):,}")
    if len(invalid_docs) != EXPECTED_INVALID_DOCS:
        raise RuntimeError(f"Unexpected invalid docs: {len(invalid_docs):,}")
    if invalid_chunks != EXPECTED_INVALID_CHUNKS:
        raise RuntimeError(f"Unexpected invalid chunks: {invalid_chunks:,}")
    if not current_docs.issubset(official_ids) or invalid_docs & official_ids:
        raise RuntimeError("Official-id filtering invariant failed")

    return current_docs, global_chunk_offset, output_part


def append_embedded_artifact(
    official_ids: set[int], current_docs: set[int], chunk_offset: int, output_part: int
) -> tuple[set[int], int, int]:
    print("[2/7] Appending the unmerged 99,448-document embedded artifact...")
    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        root = zip_root(zf)
        chunk_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{root}/chunks/part_") and name.endswith(".parquet")
        )
        source_chunk_tables = [
            pq.read_table(
                BytesIO(zf.read(name)), columns=["chunk_id", "doc_id", "start", "end"]
            )
            for name in chunk_members
        ]
        source_chunks = pa.concat_tables(source_chunk_tables).to_pandas()
        source_docs = set(source_chunks["doc_id"].astype("int64").tolist())

        if len(source_chunks) != EXPECTED_SOURCE_CHUNKS:
            raise RuntimeError(f"Unexpected source chunks: {len(source_chunks):,}")
        if len(source_docs) != EXPECTED_SOURCE_DOCS:
            raise RuntimeError(f"Unexpected source docs: {len(source_docs):,}")
        if not source_docs.issubset(official_ids):
            raise RuntimeError("The source artifact contains non-official document IDs")
        if source_docs & current_docs:
            raise RuntimeError("The source artifact overlaps the current valid index")

        id_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{root}/emb_dense/ids_") and name.endswith(".npy")
        )
        source_ids_parts = [zip_npy(zf, name) for name in id_members]
        source_ids = np.concatenate(source_ids_parts)
        if not np.array_equal(
            source_ids, source_chunks["chunk_id"].to_numpy(dtype=np.int64)
        ):
            raise RuntimeError("Source vector/chunk order mismatch")

        source_cursor = 0
        for source_part_index, source_ids_part in enumerate(source_ids_parts):
            part_rows = len(source_ids_part)
            part_chunks = source_chunks.iloc[
                source_cursor : source_cursor + part_rows
            ].copy()
            tag = f"{output_part:04d}"
            new_ids = np.arange(
                chunk_offset, chunk_offset + part_rows, dtype=np.int64
            )
            np.save(STAGING_INDEX / "emb_dense" / f"ids_{tag}.npy", new_ids)

            source_tag = f"{source_part_index:04d}"
            extract_zip_member(
                zf,
                f"{root}/emb_dense/vec_{source_tag}.npy",
                STAGING_INDEX / "emb_dense" / f"vec_{tag}.npy",
            )
            extract_zip_member(
                zf,
                f"{root}/emb_dense/sp_{source_tag}.npz",
                STAGING_INDEX / "emb_dense" / f"sp_{tag}.npz",
            )

            part_chunks = part_chunks[["chunk_id", "doc_id", "start", "end"]]
            part_chunks["chunk_id"] = new_ids
            part_chunks.to_parquet(
                STAGING_INDEX / "chunks" / f"part_{tag}.parquet",
                index=False,
                compression="snappy",
            )
            print(
                f"  - source part {source_tag}: added {part_rows:,} chunks, "
                f"new range {new_ids[0]:,}..{new_ids[-1]:,}"
            )
            source_cursor += part_rows
            chunk_offset += part_rows
            output_part += 1

        if source_cursor != len(source_chunks):
            raise RuntimeError("Not all source chunks were consumed")

    if chunk_offset != EXPECTED_FINAL_CHUNKS:
        raise RuntimeError(f"Unexpected final chunk count: {chunk_offset:,}")
    return source_docs, chunk_offset, output_part


def rebuild_text_store(official_ids: set[int], source_docs: set[int]) -> int:
    print("[3/7] Filtering current text store and appending source text...")
    output_part = 0
    current_text_docs: set[int] = set()

    for source_path in sorted((CURRENT_INDEX / "text").glob("part_*.parquet")):
        frame = pd.read_parquet(source_path, columns=["doc_id", "text"])
        keep = frame["doc_id"].isin(official_ids)
        filtered = frame.loc[keep].copy()
        if not filtered.empty:
            filtered.to_parquet(
                STAGING_INDEX / "text" / f"part_{output_part:04d}.parquet",
                index=False,
                compression="snappy",
            )
            current_text_docs.update(filtered["doc_id"].astype("int64").tolist())
            output_part += 1
        del frame, keep, filtered

    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        root = zip_root(zf)
        text_members = sorted(
            name
            for name in zf.namelist()
            if name.startswith(f"{root}/text/part_") and name.endswith(".parquet")
        )
        artifact_text_docs: set[int] = set()
        for member in text_members:
            raw = zf.read(member)
            table = pq.read_table(BytesIO(raw), columns=["doc_id"])
            artifact_text_docs.update(table["doc_id"].to_pylist())
            target = STAGING_INDEX / "text" / f"part_{output_part:04d}.parquet"
            with target.open("wb") as handle:
                handle.write(raw)
            output_part += 1

    if artifact_text_docs != source_docs:
        raise RuntimeError("Source text documents do not match source chunk documents")
    if current_text_docs & source_docs:
        raise RuntimeError("Source text overlaps current valid text")
    all_text_docs = current_text_docs | artifact_text_docs
    if len(all_text_docs) != EXPECTED_FINAL_DOCS:
        raise RuntimeError(f"Unexpected final text docs: {len(all_text_docs):,}")
    print(f"  - wrote {output_part} text parts for {len(all_text_docs):,} documents")
    return output_part


def build_global_metadata_and_faiss(total_chunks: int) -> int:
    print("[4/7] Building global chunk metadata and FAISS IndexFlatIP...")
    ids_files = sorted((STAGING_INDEX / "emb_dense").glob("ids_*.npy"))
    chunk_files = sorted((STAGING_INDEX / "chunks").glob("part_*.parquet"))
    if len(ids_files) != len(chunk_files):
        raise RuntimeError("Staging embedding/chunk part counts do not match")

    all_ids: list[np.ndarray] = []
    chunk_tables: list[pa.Table] = []
    index = faiss.IndexFlatIP(DENSE_DIM)

    for number, (ids_path, chunks_path) in enumerate(zip(ids_files, chunk_files), 1):
        ids = np.load(ids_path, allow_pickle=False)
        vectors_path = ids_path.with_name(ids_path.name.replace("ids_", "vec_"))
        vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (len(ids), DENSE_DIM):
            raise RuntimeError(f"Unexpected vector shape in {vectors_path.name}: {vectors.shape}")
        index.add(np.asarray(vectors, dtype=np.float32))
        all_ids.append(ids)
        chunk_tables.append(pq.read_table(chunks_path))
        print(f"  - FAISS part {number:02d}/{len(ids_files):02d}: {len(ids):,} rows")
        del vectors, ids
        gc.collect()

    concatenated_ids = np.concatenate(all_ids)
    if len(concatenated_ids) != total_chunks:
        raise RuntimeError("Unexpected concatenated chunk-id count")
    if not np.array_equal(concatenated_ids, np.arange(total_chunks, dtype=np.int64)):
        raise RuntimeError("Chunk IDs are not contiguous and ordered")
    np.save(STAGING_INDEX / "chunk_ids.npy", concatenated_ids)

    metadata = pa.concat_tables(chunk_tables)
    metadata_ids = np.asarray(metadata["chunk_id"], dtype=np.int64)
    if not np.array_equal(concatenated_ids, metadata_ids):
        raise RuntimeError("Global chunk metadata order does not match vector order")
    pq.write_table(
        metadata,
        STAGING_INDEX / "chunks_metadata.parquet",
        compression="snappy",
    )
    del chunk_tables, metadata, metadata_ids, all_ids, concatenated_ids

    if index.ntotal != total_chunks:
        raise RuntimeError(f"FAISS rows {index.ntotal:,} != {total_chunks:,}")
    faiss.write_index(index, str(STAGING_INDEX / "faiss_index_flat_ip.bin"))
    del index
    gc.collect()
    return len(ids_files)


def build_global_sparse(total_chunks: int) -> None:
    print("[5/7] Building global sparse CSR without holding duplicate matrices...")
    sparse_files = sorted((STAGING_INDEX / "emb_dense").glob("sp_*.npz"))
    ids_files = sorted((STAGING_INDEX / "emb_dense").glob("ids_*.npy"))
    if len(sparse_files) != len(ids_files):
        raise RuntimeError("Staging sparse/id part counts do not match")

    total_nnz = 0
    row_counts: list[int] = []
    nnz_counts: list[int] = []
    for sparse_path, ids_path in zip(sparse_files, ids_files):
        rows = len(np.load(ids_path, mmap_mode="r", allow_pickle=False))
        with np.load(sparse_path, allow_pickle=False) as packed:
            ptr = packed["ptr"]
            if len(ptr) != rows + 1:
                raise RuntimeError(f"Sparse row pointer mismatch in {sparse_path.name}")
            nnz = int(ptr[-1])
        row_counts.append(rows)
        nnz_counts.append(nnz)
        total_nnz += nnz

    if sum(row_counts) != total_chunks:
        raise RuntimeError("Sparse part row total does not match final chunk count")

    values = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(total_chunks + 1, dtype=np.int64)
    indptr[0] = 0
    row_cursor = 0
    nnz_cursor = 0

    for number, (sparse_path, rows, nnz) in enumerate(
        zip(sparse_files, row_counts, nnz_counts), 1
    ):
        with np.load(sparse_path, allow_pickle=False) as packed:
            part_values = packed["vals"]
            part_indices = packed["ids"]
            part_ptr = packed["ptr"]
            values[nnz_cursor : nnz_cursor + nnz] = part_values
            indices[nnz_cursor : nnz_cursor + nnz] = part_indices
            indptr[row_cursor + 1 : row_cursor + rows + 1] = part_ptr[1:] + nnz_cursor
        row_cursor += rows
        nnz_cursor += nnz
        print(
            f"  - sparse part {number:02d}/{len(sparse_files):02d}: "
            f"{rows:,} rows, {nnz:,} non-zeros"
        )

    matrix = sp.csr_matrix(
        (values, indices, indptr), shape=(total_chunks, SPARSE_DIM)
    )
    sp.save_npz(STAGING_INDEX / "sparse_csr.npz", matrix, compressed=True)
    del matrix, values, indices, indptr
    gc.collect()


def write_stats_and_manifest(
    emb_parts: int, text_parts: int, total_chunks: int
) -> None:
    print("[6/7] Writing stats and reproducibility manifest...")
    now = datetime.now().astimezone().isoformat()
    stats = {
        "index_name": "ViBioMIR_Consolidated_Index_Clean",
        "total_docs": EXPECTED_FINAL_DOCS,
        "total_chunks": total_chunks,
        "dense_dim": DENSE_DIM,
        "vocab_size": SPARSE_DIM,
        "model": "BAAI/bge-m3",
        "emb_parts": emb_parts,
        "chunks_parts": emb_parts,
        "text_parts": text_parts,
        "created_at": now,
        "source": (
            "Official-ID-filtered rebuild of previous consolidated index plus "
            "index_backup_unembedded_samples_0-100-20261014T033450Z-1-001.zip"
        ),
        "removed_invalid_docs": EXPECTED_INVALID_DOCS,
        "removed_invalid_chunks": EXPECTED_INVALID_CHUNKS,
        "added_valid_docs": EXPECTED_SOURCE_DOCS,
        "added_valid_chunks": EXPECTED_SOURCE_CHUNKS,
    }
    (STAGING_INDEX / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (CURRENT_INDEX / "stats.json").open("r", encoding="utf-8") as handle:
        previous_stats = json.load(handle)
    manifest = {
        "built_at": now,
        "official_list": str(OFFICIAL_PATH.relative_to(PROJECT_ROOT)),
        "official_list_sha256": sha256_file(OFFICIAL_PATH),
        "source_artifact": str(SOURCE_ZIP.relative_to(PROJECT_ROOT)),
        "source_artifact_sha256": sha256_file(SOURCE_ZIP),
        "previous_index_stats": previous_stats,
        "result": stats,
    }
    (STAGING_INDEX / "rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_staging(
    official_ids: set[int], current_valid_docs: set[int], source_docs: set[int]
) -> None:
    print("[7/7] Validating the complete staging index...")
    metadata = pq.read_table(
        STAGING_INDEX / "chunks_metadata.parquet",
        columns=["chunk_id", "doc_id", "start", "end"],
    )
    chunk_ids = np.asarray(metadata["chunk_id"], dtype=np.int64)
    doc_ids = np.asarray(metadata["doc_id"], dtype=np.int64)
    final_docs = set(doc_ids.tolist())
    if len(metadata) != EXPECTED_FINAL_CHUNKS:
        raise RuntimeError("Final metadata chunk count mismatch")
    if len(final_docs) != EXPECTED_FINAL_DOCS:
        raise RuntimeError("Final metadata document count mismatch")
    if not final_docs.issubset(official_ids):
        raise RuntimeError("Final index still contains non-official document IDs")
    if final_docs != current_valid_docs | source_docs:
        raise RuntimeError("Final document set does not match expected union")
    if not np.array_equal(chunk_ids, np.arange(EXPECTED_FINAL_CHUNKS, dtype=np.int64)):
        raise RuntimeError("Final metadata chunk IDs are not contiguous")
    if np.any(np.asarray(metadata["start"]) < 0):
        raise RuntimeError("Negative chunk start detected")
    if np.any(np.asarray(metadata["end"]) <= np.asarray(metadata["start"])):
        raise RuntimeError("Invalid chunk span detected")

    dense_rows = 0
    dense_ids: list[np.ndarray] = []
    for ids_path in sorted((STAGING_INDEX / "emb_dense").glob("ids_*.npy")):
        ids = np.load(ids_path, allow_pickle=False)
        vectors = np.load(
            ids_path.with_name(ids_path.name.replace("ids_", "vec_")),
            mmap_mode="r",
            allow_pickle=False,
        )
        if len(ids) != len(vectors) or vectors.shape[1] != DENSE_DIM:
            raise RuntimeError(f"Dense shape mismatch in {ids_path.name}")
        dense_rows += len(ids)
        dense_ids.append(ids)
        del vectors
    if dense_rows != EXPECTED_FINAL_CHUNKS:
        raise RuntimeError("Dense row total mismatch")
    if not np.array_equal(np.concatenate(dense_ids), chunk_ids):
        raise RuntimeError("Dense chunk-id order mismatch")
    del dense_ids

    sparse = sp.load_npz(STAGING_INDEX / "sparse_csr.npz")
    if sparse.shape != (EXPECTED_FINAL_CHUNKS, SPARSE_DIM):
        raise RuntimeError(f"Sparse matrix shape mismatch: {sparse.shape}")
    del sparse
    gc.collect()

    index = faiss.read_index(
        str(STAGING_INDEX / "faiss_index_flat_ip.bin"), faiss.IO_FLAG_MMAP
    )
    if index.ntotal != EXPECTED_FINAL_CHUNKS or index.d != DENSE_DIM:
        raise RuntimeError("FAISS shape mismatch")
    probe = np.zeros((1, DENSE_DIM), dtype=np.float32)
    probe[0, 0] = 1.0
    scores, rows = index.search(probe, 5)
    if rows.shape != (1, 5) or scores.shape != (1, 5):
        raise RuntimeError("FAISS smoke search failed")
    del index, probe, scores, rows
    gc.collect()

    text_docs: list[int] = []
    for text_path in sorted((STAGING_INDEX / "text").glob("part_*.parquet")):
        text_docs.extend(
            pq.read_table(text_path, columns=["doc_id"])["doc_id"].to_pylist()
        )
    if len(text_docs) != EXPECTED_FINAL_DOCS:
        raise RuntimeError("Final text row count mismatch")
    if len(set(text_docs)) != EXPECTED_FINAL_DOCS:
        raise RuntimeError("Duplicate document IDs detected in final text store")
    if set(text_docs) != final_docs:
        raise RuntimeError("Final text and metadata document sets differ")

    print(
        f"  - VALID: {len(final_docs):,} official docs, "
        f"{len(metadata):,} chunks, dense/sparse/FAISS/text all aligned"
    )


def atomic_swap() -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = INDEX_ROOT / f"index_backup_pre_clean_rebuild_{timestamp}"
    if backup.exists():
        raise FileExistsError(f"Backup path unexpectedly exists: {backup}")

    print(f"Swapping validated staging index into place; backup: {backup.name}")
    CURRENT_INDEX.rename(backup)
    try:
        STAGING_INDEX.rename(CURRENT_INDEX)
    except Exception:
        if not CURRENT_INDEX.exists() and backup.exists():
            backup.rename(CURRENT_INDEX)
        raise
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="Build and validate staging only; do not replace the active index.",
    )
    args = parser.parse_args()

    started = time.time()
    prepare_staging()
    official_ids, official_sorted = load_official_ids()
    print(f"Loaded {len(official_ids):,} official BTC document IDs")

    current_docs, chunk_offset, output_part = build_filtered_parts(
        official_ids, official_sorted
    )
    source_docs, total_chunks, output_part = append_embedded_artifact(
        official_ids, current_docs, chunk_offset, output_part
    )
    text_parts = rebuild_text_store(official_ids, source_docs)
    emb_parts = build_global_metadata_and_faiss(total_chunks)
    if emb_parts != output_part:
        raise RuntimeError("Output embedding part count changed unexpectedly")
    build_global_sparse(total_chunks)
    write_stats_and_manifest(emb_parts, text_parts, total_chunks)
    validate_staging(official_ids, current_docs, source_docs)

    backup = None
    if not args.no_swap:
        backup = atomic_swap()

    elapsed = time.time() - started
    print("=" * 80)
    print(f"CLEAN REBUILD COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"Active index: {CURRENT_INDEX if backup else STAGING_INDEX}")
    if backup:
        print(f"Recoverable backup: {backup}")
    print(f"Documents: {EXPECTED_FINAL_DOCS:,}")
    print(f"Chunks: {EXPECTED_FINAL_CHUNKS:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
