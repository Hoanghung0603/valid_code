"""Bounded-memory builders for the existing FAISS FlatIP and SciPy CSR formats.

FlatIP layout is guarded against the installed FAISS serializer before use:
https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_write.cpp
No approximate index, quantization, normalization or ranking change is made.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import struct
import zipfile

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HEADER = struct.Struct('<4siqqq?iQ')
BLOCK = 2 * 1024 * 1024


def check(condition, message):
    if not condition:
        raise ValueError(message)


def flat_header(rows, dim):
    """Fail closed if the installed FAISS version uses a different FlatIP layout."""
    for count in (0, 2):
        sample = np.arange(count * dim, dtype=np.float32).reshape(count, dim)
        native = faiss.IndexFlatIP(dim)
        native.add(sample)
        expected = HEADER.pack(b'IxFI', dim, count, 1 << 20, 1 << 20, True, 0, count * dim)
        check(faiss.serialize_index(native).tobytes() == expected + sample.tobytes(),
              'Unsupported installed FAISS serialization format')
    return HEADER.pack(b'IxFI', dim, rows, 1 << 20, 1 << 20, True, 0, rows * dim)


def write_flat(path, rows, dim, vector_blocks):
    sha = hashlib.sha256()
    actual = 0
    with Path(path).open('wb') as handle:
        handle.write(flat_header(rows, dim))
        for block in vector_blocks:
            block = np.asarray(block, dtype='<f4', order='C')
            check(block.ndim == 2 and block.shape[1] == dim and np.isfinite(block).all(), 'Invalid dense block')
            raw = block.tobytes()
            handle.write(raw)
            sha.update(raw)
            actual += len(block)
    check(actual == rows, 'Dense row total mismatch')
    return sha.hexdigest()


def validate_flat(path, rows, dim, payload_sha256):
    check(Path(path).stat().st_size == HEADER.size + rows * dim * 4, 'FAISS file size mismatch')
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        check(handle.read(HEADER.size) == flat_header(rows, dim), 'FAISS header mismatch')
        sample = handle.read(min(rows, 64) * dim * 4)
        sha.update(sample)
        while raw := handle.read(BLOCK):
            sha.update(raw)
    check(sha.hexdigest() == payload_sha256, 'FAISS payload readback mismatch')
    # Native reader/search test without loading the multi-GB full index into RAM.
    count = len(sample) // (dim * 4)
    native = faiss.deserialize_index(np.frombuffer(flat_header(count, dim) + sample, dtype='uint8').copy())
    expected = np.frombuffer(sample, dtype='<f4').reshape(count, dim)
    check(native.ntotal == count and np.array_equal(native.reconstruct_n(0, count), expected), 'FAISS reconstruction mismatch')
    if count:
        distances, ids = native.search(expected[:1], min(count, 5))
        scores = expected @ expected[0]
        check(np.allclose(distances[0], np.sort(scores)[::-1][:len(ids[0])], rtol=2e-5, atol=2e-5), 'FAISS search mismatch')


def rebuild_dense(index_dir, total_chunks, dim):
    index_dir = Path(index_dir)
    metadata = []
    all_ids = []
    cursor = 0
    pairs = sorted((index_dir / 'emb_dense').glob('ids_*.npy'))

    def blocks():
        nonlocal cursor
        for number, path in enumerate(pairs, 1):
            ids = np.load(path, allow_pickle=False)
            tag = path.stem.removeprefix('ids_')
            vectors = np.load(path.with_name(f'vec_{tag}.npy'), mmap_mode='r', allow_pickle=False)
            table = pq.read_table(index_dir / 'chunks' / f'part_{tag}.parquet', columns=['chunk_id', 'doc_id', 'start', 'end'])
            check(vectors.shape == (len(ids), dim), 'Dense shape mismatch')
            check(np.array_equal(ids, np.arange(cursor, cursor + len(ids))), 'Shard chunk IDs not contiguous')
            check(np.array_equal(ids, np.asarray(table['chunk_id'])), 'Vector/metadata order mismatch')
            all_ids.append(ids)
            metadata.append(table)
            for start in range(0, len(ids), 2048):
                yield vectors[start:start + 2048]
            cursor += len(ids)
            print(f'  dense {number}/{len(pairs)}: {cursor:,} rows', flush=True)

    sha = write_flat(index_dir / 'faiss_index_flat_ip.bin', total_chunks, dim, blocks())
    check(cursor == total_chunks, 'Global dense count mismatch')
    np.save(index_dir / 'chunk_ids.npy', np.concatenate(all_ids))
    pq.write_table(pa.concat_tables(metadata), index_dir / 'chunks_metadata.parquet', compression='snappy')
    validate_flat(index_dir / 'faiss_index_flat_ip.bin', total_chunks, dim, sha)
    return {'parts': len(pairs), 'rows': total_chunks, 'payload_sha256': sha,
            'validation': 'full header/size/payload readback; native FAISS reader/reconstruction/search on bounded sample'}


def npz_blocks(path, key, dtype):
    """Decode a one-dimensional NPY member in small blocks, not a whole shard."""
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as handle:
        version = np.lib.format.read_magic(handle)
        check(version in ((1, 0), (2, 0)), 'Unsupported NPY header version')
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, stored = reader(handle)
        check(len(shape) == 1 and not fortran and stored.kind in 'iuf', 'Unsupported sparse array')
        remaining = shape[0]
        while remaining:
            count = min(remaining, max(1, BLOCK // stored.itemsize))
            raw = handle.read(count * stored.itemsize)
            check(len(raw) == count * stored.itemsize, 'Truncated NPY array')
            yield np.frombuffer(raw, dtype=stored).astype(dtype, copy=False)
            remaining -= count
        check(not handle.read(1), 'Unexpected NPY trailing bytes')


def write_npz_array(archive, name, count, dtype, blocks, validate=None):
    sha = hashlib.sha256()
    seen = 0
    dtype = np.dtype(dtype)
    with archive.open(name + '.npy', 'w', force_zip64=True) as handle:
        np.lib.format.write_array_header_1_0(handle, dict(descr=dtype.str, fortran_order=False, shape=(count,)))
        for block in blocks:
            block = np.asarray(block, dtype=dtype)
            check(block.ndim == 1, 'Expected flat sparse block')
            if validate:
                validate(block)
            raw = block.tobytes()
            handle.write(raw)
            sha.update(raw)
            seen += len(block)
    check(seen == count, f'{name} length mismatch')
    return sha.hexdigest()


def sparse_layout(index_dir, total_chunks):
    result = []
    for path in sorted((Path(index_dir) / 'emb_dense').glob('sp_*.npz')):
        ids_path = path.with_name(path.name.replace('sp_', 'ids_').replace('.npz', '.npy'))
        rows = len(np.load(ids_path, mmap_mode='r', allow_pickle=False))
        with np.load(path, allow_pickle=False) as packed:
            ptr = packed['ptr']
            check(ptr.dtype.kind in 'iu' and ptr.shape == (rows + 1,) and ptr[0] == 0 and (np.diff(ptr) >= 0).all(), 'Bad source CSR ptr')
            result.append((path, rows, int(ptr[-1])))
    check(sum(x[1] for x in result) == total_chunks, 'Sparse row total mismatch')
    return result


def rebuild_sparse(index_dir, total_chunks, vocab):
    layout = sparse_layout(index_dir, total_chunks)
    total_nnz = sum(x[2] for x in layout)
    path = Path(index_dir) / 'sparse_csr.npz'
    hashes = {}

    def parts(key, dtype):
        for source, _, nnz in layout:
            seen = 0
            for block in npz_blocks(source, key, dtype):
                seen += len(block)
                yield block
            check(seen == nnz, f'Source {key} NNZ mismatch')

    def pointers():
        yield np.array([0], dtype='<i8')
        offset = 0
        for source, _, nnz in layout:
            with np.load(source, allow_pickle=False) as packed:
                yield packed['ptr'][1:].astype('<i8') + offset
            offset += nnz

    def valid_data(a):
        check(np.isfinite(a).all(), 'Nonfinite sparse values')

    def valid_indices(a):
        check((a >= 0).all() and (a < vocab).all(), 'Sparse index outside vocabulary')

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for key, value in [('format', np.array(b'csr')), ('shape', np.array([total_chunks, vocab], dtype='<i8'))]:
            raw = io.BytesIO()
            np.save(raw, value, allow_pickle=False)
            archive.writestr(key + '.npy', raw.getvalue())
        print(f'  sparse: {total_chunks:,} rows / {total_nnz:,} nonzeros', flush=True)
        hashes['data'] = write_npz_array(archive, 'data', total_nnz, '<f4', parts('vals', '<f4'), valid_data)
        print('  sparse data complete', flush=True)
        hashes['indices'] = write_npz_array(archive, 'indices', total_nnz, '<i4', parts('ids', '<i4'), valid_indices)
        print('  sparse indices complete', flush=True)
        hashes['indptr'] = write_npz_array(archive, 'indptr', total_chunks + 1, '<i8', pointers())
    with np.load(path, allow_pickle=False) as packed:
        check(packed['format'].item() == b'csr' and packed['shape'].tolist() == [total_chunks, vocab], 'Sparse container metadata mismatch')
    for key, dtype in [('data', '<f4'), ('indices', '<i4'), ('indptr', '<i8')]:
        sha = hashlib.sha256()
        count = 0
        for block in npz_blocks(path, key, dtype):
            sha.update(block.tobytes())
            count += len(block)
        check(count == (total_chunks + 1 if key == 'indptr' else total_nnz), 'Sparse output length mismatch')
        check(sha.hexdigest() == hashes[key], f'Sparse readback mismatch: {key}')
        print(f'  sparse verified: {key}', flush=True)
    return {'shape': [total_chunks, vocab], 'nnz': total_nnz, 'array_sha256': hashes}
