import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import sync_embedded_zip_registry as sync


def make_artifact(path, bad_vector=False, bad_csr=False):
    with zipfile.ZipFile(path, 'w') as z:
        stats = dict(docs=3, kept=1, dropped=2, failed=0, chunks=1,
                     rows_docs=1, rows_text=1, rows_chunks=1)
        z.writestr('index/stats.json', json.dumps(stats))
        frames = {
            'docs': pd.DataFrame([dict(doc_id=2, url='https://a/2', n_chunks=1)]),
            'text': pd.DataFrame([dict(doc_id=2, text='article body')]),
            'chunks': pd.DataFrame([dict(chunk_id=0, doc_id=2, start=0, end=12, n_chars=12)]),
        }
        for kind, frame in frames.items():
            b = io.BytesIO()
            frame.to_parquet(b, index=False)
            z.writestr(f'index/{kind}/part_0000.parquet', b.getvalue())
            b = io.BytesIO()
            np.savez(b, key=np.array([0 if kind == 'chunks' else 2]), part=np.array([0]), row=np.array([0]))
            z.writestr(f'index/{kind}/_index.npz', b.getvalue())
        for name, value in [('ids', np.array([0])), ('vec', np.ones((1, 1024), dtype='float16'))]:
            if name == 'vec' and bad_vector:
                value[0, 0] = np.nan
            b = io.BytesIO()
            np.save(b, value)
            z.writestr(f'index/emb_dense/{name}_0000.npy', b.getvalue())
        b = io.BytesIO()
        np.savez(b, ids=np.array([5]), vals=np.array([0.1]), ptr=np.array([0, 2 if bad_csr else 1]))
        z.writestr('index/emb_dense/sp_0000.npz', b.getvalue())


@pytest.mark.parametrize('bad_vector,bad_csr', [(False, False), (True, False), (False, True)])
def test_artifact_requires_actual_finite_aligned_vectors(tmp_path, bad_vector, bad_csr):
    path = tmp_path / 'artifact.zip'
    make_artifact(path, bad_vector, bad_csr)
    if bad_vector or bad_csr:
        with pytest.raises(ValueError):
            sync.read_artifact(path)
    else:
        docs, text, stats = sync.read_artifact(path)
        assert docs.doc_id.tolist() == [2]  # Input docs=3 must not move three rows.
        assert len(text) == 1 and stats['dropped'] == 2


def fixtures():
    official = pd.DataFrame({'id': [1, 2, 3, 4], 'url': [f'https://a/{i}' for i in range(1, 5)]})
    frames = {
        'embedded': official.iloc[[0]].assign(text='old embedded').reset_index(drop=True),
        'unembedded': official.iloc[[1, 2]].assign(text=['article body', 'dropped input']).reset_index(drop=True),
        'remaining': official.iloc[[3]].reset_index(drop=True),
        'excluded': official.iloc[:0].copy(),
    }
    return official, frames


def test_select_only_vector_backed_rows_and_reject_text_or_url_mismatch(tmp_path):
    official, frames = fixtures()
    docs = pd.DataFrame([dict(doc_id=2, url='https://a/2')])
    texts = pd.DataFrame([dict(doc_id=2, text='article body')])
    identities = {n: f[['id', 'url']] for n, f in frames.items()}
    moved, pending, _ = sync.select_moves(docs, texts, official, identities, frames['unembedded'], tmp_path / 'unused')
    assert moved.id.tolist() == [2] and pending.id.tolist() == [3]
    with pytest.raises(ValueError, match='text mismatch'):
        sync.select_moves(docs, texts.assign(text='different'), official, identities, frames['unembedded'], tmp_path / 'unused')
    with pytest.raises(ValueError, match='official'):
        sync.select_moves(docs.assign(url='https://wrong'), texts, official, identities, frames['unembedded'], tmp_path / 'unused')


@pytest.mark.parametrize('fail_second_replace', [False, True])
def test_streaming_transaction_preserves_content_and_rolls_back(tmp_path, monkeypatch, fail_second_replace):
    _, frames = fixtures()
    paths = {n: tmp_path / (n + '.parquet') for n in frames}
    for n, f in frames.items():
        f.to_parquet(paths[n], index=False)
    hashes = {n: sync.digest(p) for n, p in paths.items()}
    out = tmp_path / 'audit'
    out.mkdir()
    moved = frames['unembedded'].iloc[:1]
    pending = frames['unembedded'].iloc[1:].reset_index(drop=True)
    if fail_second_replace:
        replace = sync.os.replace

        def fail(src, dst):
            if dst == paths['unembedded']:
                raise OSError('simulated failure')
            return replace(src, dst)

        monkeypatch.setattr(sync.os, 'replace', fail)
        with pytest.raises(OSError):
            sync.commit_registry_moves(paths, moved, pending, hashes, out)
        assert {n: sync.digest(p) for n, p in paths.items()} == hashes
    else:
        sync.commit_registry_moves(paths, moved, pending, hashes, out)
        assert pd.read_parquet(paths['embedded']).id.tolist() == [1, 2]
        assert pd.read_parquet(paths['unembedded']).id.tolist() == [3]
        for n in ('embedded', 'unembedded'):
            assert sync.digest(out / 'backups' / paths[n].name) == hashes[n]
        for n in ('remaining', 'excluded'):
            assert sync.digest(paths[n]) == hashes[n]


def test_noop_resync_is_idempotent(tmp_path):
    official, frames = fixtures()
    embedded = pd.concat([frames['embedded'], frames['unembedded'].iloc[:1]], ignore_index=True)
    path = tmp_path / 'embedded.parquet'
    embedded.to_parquet(path, index=False)
    frames['embedded'] = embedded
    frames['unembedded'] = frames['unembedded'].iloc[1:].reset_index(drop=True)
    moved, pending, _ = sync.select_moves(
        pd.DataFrame([dict(doc_id=2, url='https://a/2')]),
        pd.DataFrame([dict(doc_id=2, text='article body')]), official,
        {n: f[['id', 'url']] for n, f in frames.items()}, frames['unembedded'], path)
    assert len(moved) == 0 and pending.id.tolist() == [3]
