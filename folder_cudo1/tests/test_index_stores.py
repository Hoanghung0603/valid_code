import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from medret.chunk.run import CHUNK_SCHEMA
from medret.index.export import EXPORT_SCHEMA
from medret.index.lexical import BM25Index, build_bm25, vi_terms, zh_terms
from medret.parse.run import DOC_SCHEMA


def _write(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


DOCS = {
    1: ("vi", "Đái tháo đường", "Đái tháo đường\nBệnh đái tháo đường type 2 cần kiểm soát đường huyết và HbA1c."),
    2: ("vi", "Phẫu thuật", "Phẫu thuật\nTháo bỏ băng gạc sau phẫu thuật cắt amidan ở trẻ."),
    3: ("zh", "糖尿病", "糖尿病\n糖尿病患者应控制血糖，常用二甲双胍。"),
    4: ("zh", "感冒", "感冒\n感冒发烧可以多喝水，注意休息。"),
}


@pytest.fixture()
def corpus(tmp_path):
    docs, chunks, inputs = tmp_path / "documents", tmp_path / "chunks", tmp_path / "embed_inputs"
    drows, crows, irows = [], [], []
    for i, (doc_id, (lang, title, text)) in enumerate(DOCS.items()):
        drows.append(dict(doc_id=doc_id, url=f"u{doc_id}", host="h", lang=lang, title=title, text=text,
                          blocks=[], n_chars=len(text), text_sha1="x", dup_group=None, extractor_version="t"))
        crows.append(dict(chunk_id=doc_id * 10_000, doc_id=doc_id, lang=lang, start=0, end=len(text), heading_path="",
                          n_tokens=10, chunker_version="t", embed_sha1=f"h{doc_id}"))
        irows.append(dict(input_id=i, embed_sha1=f"h{doc_id}", text=text, n_tokens=10))
    # a second doc carrying exactly the same input as doc 1 (shares its vector)
    drows.append({**drows[0], "doc_id": 5, "url": "u5"})
    crows.append({**crows[0], "chunk_id": 50_000, "doc_id": 5})
    _write(docs / "s/a.parquet", drows, DOC_SCHEMA)
    _write(chunks / "s/a.parquet", crows, CHUNK_SCHEMA)
    _write(inputs / "shard-00000.parquet", irows, EXPORT_SCHEMA)
    _write(tmp_path / "aliases.parquet", [{"id": 9, "fetched_id": 1}, {"id": 1, "fetched_id": 1}],
           pa.schema([("id", pa.int64()), ("fetched_id", pa.int64())]))
    return tmp_path


def test_tokenizers():
    assert "đái_tháo" in vi_terms("Đái tháo đường") and "tháo_đường" in vi_terms("Đái tháo đường")
    assert zh_terms("服用二甲双胍500mg") == ["服用", "用二", "二甲", "甲双", "双胍", "500mg"]


def test_bm25_per_language(corpus):
    for lang in ("vi", "zh"):
        n = build_bm25(corpus / "documents", corpus / "chunks", corpus / "bm25" / lang, lang, workers=1,
                       heap_bytes=50_000_000, log=lambda *_: None)
        assert n == (3 if lang == "vi" else 2)
    assert BM25Index(corpus / "bm25" / "vi").search("tiểu đường hay đái tháo đường", k=1)[0][0] in (10_000, 50_000)
    assert BM25Index(corpus / "bm25" / "zh").search("糖尿病用什么药", k=1)[0][0] == 30_000


def test_meta_store(corpus):
    from medret.index.meta import MetaStore, build_meta

    db = corpus / "meta.duckdb"
    stats = build_meta(db, corpus / "documents", corpus / "chunks", corpus / "embed_inputs",
                       corpus / "aliases.parquet", memory_limit="1GB", log=lambda *_: None)
    assert stats == {"documents": 5, "chunks": 5, "chunks_without_input": 0}
    m = MetaStore(db)
    c = m.chunks([30_000]).iloc[0]
    assert c.chunk_text == DOCS[3][2] and c.input_id == 2 and c.title == "糖尿病"
    assert sorted(m.chunks_of_inputs([0])["doc_id"]) == [1, 5]  # one vector, two docs
    assert sorted(m.twins([1])["id"]) == [1, 9]


def test_dense_store_partial_and_exact_rescoring(tmp_path):
    from medret.index.dense import DenseIndex, assemble_vectors, build_ivfpq

    rng = np.random.default_rng(0)
    n, d = 3000, 32
    allv = rng.standard_normal((n, d)).astype(np.float32)
    allv /= np.linalg.norm(allv, axis=1, keepdims=True)
    emb = tmp_path / "emb"
    emb.mkdir()
    for s, (a, b) in enumerate([(0, 1000), (1000, 2000)]):  # third shard not embedded yet
        np.save(emb / f"shard-{s}.vec.npy", allv[a:b].astype(np.float16))
        np.save(emb / f"shard-{s}.ids.npy", np.arange(a, b, dtype=np.int64))
        (emb / f"shard-{s}.done.json").write_text(json.dumps({"shard": f"shard-{s}"}))
    out = tmp_path / "dense"
    assert assemble_vectors(emb, out, n, dim=d, log=lambda *_: None) == 2000
    assert assemble_vectors(emb, out, n, dim=d, log=lambda *_: None) == 0  # idempotent
    build_ivfpq(out, factory="IVF16,PQ8", train_size=2000, log=lambda *_: None)
    idx = DenseIndex(out, nprobe=16)
    hits = idx.search(allv[[5, 1500, 2500]], k=3, k_coarse=200)
    assert hits[0][0][0] == 5 and hits[1][0][0] == 1500  # exact re-scoring puts the vector itself first
    assert abs(hits[0][0][1] - 1.0) < 1e-2
    assert all(i < 2000 for i, _ in hits[2])  # rows not yet embedded are never returned


def test_dense_store_grows_when_inputs_are_appended(tmp_path):
    from medret.index.dense import PRESENT, VECTORS, assemble_vectors

    rng = np.random.default_rng(1)
    v = rng.standard_normal((10, 8)).astype(np.float16)
    emb = tmp_path / "emb"
    emb.mkdir()
    np.save(emb / "shard-0.vec.npy", v)
    np.save(emb / "shard-0.ids.npy", np.arange(10, dtype=np.int64))
    (emb / "shard-0.done.json").write_text(json.dumps({"shard": "shard-0"}))
    out = tmp_path / "dense"
    assemble_vectors(emb, out, 10, dim=8, log=lambda *_: None)
    assert assemble_vectors(emb, out, 15, dim=8, log=lambda *_: None) == 0  # grown, nothing new to copy
    grown = np.load(out / VECTORS)
    assert grown.shape == (15, 8) and np.array_equal(grown[:10], v)
    assert np.load(out / PRESENT).tolist() == [True] * 10 + [False] * 5


def test_dense_store_grows_in_place_and_falls_back_to_copy(tmp_path):
    from medret.index.dense import _grow, _grow_in_place

    v = np.arange(9 * 4, dtype=np.float16).reshape(9, 4)
    p, pres = tmp_path / "v.npy", tmp_path / "p.npy"
    np.save(p, v)
    np.save(pres, np.ones(9, dtype=bool))
    size = p.stat().st_size
    _grow(p, pres, 12, 4, log=lambda *_: None)  # 9 -> 12 rows: one more digit, the header padding absorbs it
    g = np.load(p)
    assert g.shape == (12, 4) and np.array_equal(g[:9], v) and not g[9:].any()
    assert p.stat().st_size == size + 3 * 4 * 2  # grown in place, no copy
    assert np.load(pres).tolist() == [True] * 9 + [False] * 3
    np.save(p, np.asfortranarray(v))  # not C order: in-place growth refused, the copy path is used
    assert not _grow_in_place(p, 12, 4)
