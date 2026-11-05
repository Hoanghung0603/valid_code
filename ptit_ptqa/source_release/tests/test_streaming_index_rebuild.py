import io
import json
from pathlib import Path
import sys
import zipfile

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import streaming_index_rebuild as stream
from scripts import merge_recovered_artifacts as runner


def test_flat_stream_bytes_equal_native_faiss_and_native_search(tmp_path):
    vectors = np.random.default_rng(42).normal(size=(103, 8)).astype('float16')
    path = tmp_path / 'flat.bin'
    sha = stream.write_flat(path, len(vectors), 8, [vectors[:3], vectors[3:64], vectors[64:]])
    stream.validate_flat(path, len(vectors), 8, sha)
    native = faiss.IndexFlatIP(8)
    native.add(vectors.astype('float32'))
    assert path.read_bytes() == faiss.serialize_index(native).tobytes()
    read = faiss.read_index(str(path))
    assert read.ntotal == len(vectors)
    np.testing.assert_array_equal(read.reconstruct_n(0, len(vectors)), vectors.astype('float32'))
    with path.open('r+b') as handle:
        handle.seek(-1, 2)
        handle.write(b'\xff')
    with pytest.raises(ValueError, match='readback'):
        stream.validate_flat(path, len(vectors), 8, sha)


def test_sparse_stream_roundtrips_with_scipy_without_changing_values(tmp_path):
    (tmp_path / 'emb_dense').mkdir()
    parts = [sp.csr_matrix([[0, .25, 0, 1], [0, 0, 0, 0]], dtype='float32'),
             sp.csr_matrix([[2, 0, 3, 0]], dtype='float32')]
    offset = 0
    for i, part in enumerate(parts):
        np.save(tmp_path / 'emb_dense' / f'ids_{i:04d}.npy', np.arange(offset, offset + part.shape[0]))
        np.savez_compressed(tmp_path / 'emb_dense' / f'sp_{i:04d}.npz',
                            ids=part.indices, vals=part.data.astype('float16'), ptr=part.indptr)
        offset += part.shape[0]
    report = stream.rebuild_sparse(tmp_path, 3, 4)
    result = sp.load_npz(tmp_path / 'sparse_csr.npz')
    np.testing.assert_array_equal(result.toarray(), sp.vstack(parts).toarray())
    assert report['shape'] == [3, 4] and report['nnz'] == 4


def make_source(path, doc_id, text):
    with zipfile.ZipFile(path, 'w') as archive:
        stats = dict(docs=1, kept=1, dropped=0, failed=0, chunks=1, rows_docs=1, rows_text=1, rows_chunks=1)
        archive.writestr('index/stats.json', json.dumps(stats))
        frames = {
            'docs': pd.DataFrame([dict(doc_id=doc_id, url=f'https://a/{doc_id}', n_chunks=1)]),
            'text': pd.DataFrame([dict(doc_id=doc_id, text=text)]),
            'chunks': pd.DataFrame([dict(chunk_id=0, doc_id=doc_id, start=0, end=len(text), n_chars=len(text))]),
        }
        for kind, f in frames.items():
            raw = io.BytesIO()
            f.to_parquet(raw, index=False)
            archive.writestr(f'index/{kind}/part_0000.parquet', raw.getvalue())
            raw = io.BytesIO()
            np.savez(raw, key=np.array([0 if kind == 'chunks' else doc_id]), part=np.array([0]), row=np.array([0]))
            archive.writestr(f'index/{kind}/_index.npz', raw.getvalue())
        for name, values in [('ids', np.array([0])), ('vec', np.ones((1, 1024), dtype='float16'))]:
            raw = io.BytesIO()
            np.save(raw, values)
            archive.writestr(f'index/emb_dense/{name}_0000.npy', raw.getvalue())
        raw = io.BytesIO()
        np.savez(raw, ids=np.array([1], dtype='int32'), vals=np.array([.5], dtype='float16'), ptr=np.array([0, 1]))
        archive.writestr('index/emb_dense/sp_0000.npz', raw.getvalue())


@pytest.mark.parametrize('registry_failure', [False, True, 'after_commit'])
def test_two_artifact_transaction_avoids_duplicate_registry_rows_and_rolls_back(tmp_path, monkeypatch, registry_failure):
    index = tmp_path / 'index'
    for directory in ('chunks', 'emb_dense', 'text'):
        (index / directory).mkdir(parents=True)
    stats = dict(total_docs=1, total_chunks=1, model='BAAI/bge-m3', dense_dim=1024,
                 vocab_size=250005, emb_parts=1, chunks_parts=1, text_parts=1)
    (index / 'stats.json').write_text(json.dumps(stats))
    chunks = pd.DataFrame([dict(chunk_id=0, doc_id=1, start=0, end=3)])
    chunks.to_parquet(index / 'chunks_metadata.parquet', index=False)
    chunks.to_parquet(index / 'chunks/part_0000.parquet', index=False)
    pd.DataFrame([dict(doc_id=1, text='old')]).to_parquet(index / 'text/part_0000.parquet', index=False)
    np.save(index / 'emb_dense/ids_0000.npy', np.array([0]))
    np.save(index / 'emb_dense/vec_0000.npy', np.ones((1, 1024), dtype='float16'))
    np.savez(index / 'emb_dense/sp_0000.npz', ids=np.array([1], dtype='int32'), vals=np.array([.5]), ptr=np.array([0, 1]))
    official = pd.DataFrame({'id': [1, 2, 3, 4], 'url': [f'https://a/{i}' for i in range(1, 5)]})
    official_path = tmp_path / 'official.parquet'
    official.to_parquet(official_path, index=False)
    frames = {
        'embedded': official.iloc[:2].assign(text=['old', 'two']).reset_index(drop=True),
        'unembedded': official.iloc[[2]].assign(text='three').reset_index(drop=True),
        'remaining': official.iloc[[3]].reset_index(drop=True),
        'excluded': official.iloc[:0].copy(),
    }
    paths = {n: tmp_path / (n + '.parquet') for n in frames}
    for name, frame in frames.items():
        frame.to_parquet(paths[name], index=False)
    source1, source2 = tmp_path / 'two.zip', tmp_path / 'five.zip'
    make_source(source1, 2, 'two')
    make_source(source2, 3, 'three')
    old_hashes = {n: runner.digest(p) for n, p in paths.items()}
    old_index_snapshot = runner.snapshot(index)
    monkeypatch.setattr(runner, 'REGISTRIES', paths)
    monkeypatch.setattr(runner, 'OFFICIAL', official_path)
    monkeypatch.setattr(runner.merge, 'INDEX_ROOT', tmp_path)
    monkeypatch.setattr(runner.merge, 'ACTIVE_INDEX', index)
    monkeypatch.setattr(runner.merge, 'STAGING_INDEX', tmp_path / 'unused')
    out = tmp_path / 'audit'
    monkeypatch.setattr(sys, 'argv', ['merge', '--source', str(source1), '--source', str(source2), '--out-dir', str(out), '--apply'])
    if registry_failure:
        def fail(*args):
            raise OSError('simulated registry failure')
        if registry_failure == 'after_commit':
            monkeypatch.setattr(runner, 'validate_partition', fail)
        else:
            monkeypatch.setattr(runner, 'commit_registry_moves', fail)
        with pytest.raises(OSError):
            runner.main()
        assert runner.snapshot(index) == old_index_snapshot
        assert {n: runner.digest(p) for n, p in paths.items()} == old_hashes
    else:
        runner.main()
        assert pd.read_parquet(paths['embedded']).id.tolist() == [1, 2, 3]
        assert len(pd.read_parquet(paths['unembedded'])) == 0
        result = json.loads((index / 'stats.json').read_text())
        assert result['total_docs'] == 3 and result['total_chunks'] == 3
        assert faiss.read_index(str(index / 'faiss_index_flat_ip.bin')).ntotal == 3
        assert sp.load_npz(index / 'sparse_csr.npz').shape == (3, 250005)
        assert runner.digest(paths['remaining']) == old_hashes['remaining']
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['registry_rows_to_move'] == 1
        assert runner.snapshot(Path(manifest['backup_index'])) == old_index_snapshot
