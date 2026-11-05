import pyarrow as pa
import pyarrow.parquet as pq

from medret.chunk.run import CHUNK_SCHEMA
from medret.index.export import export_embed_inputs
from medret.parse.run import DOC_SCHEMA


def _write(path, rows, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _doc(doc_id, text):
    return dict(doc_id=doc_id, url="u", host="h", lang="vi", title="Tiêu đề", text=text, blocks=[],
                n_chars=len(text), text_sha1="x", dup_group=None, extractor_version="t")


def _chunk(doc_id, end, sha):
    return dict(chunk_id=doc_id * 10_000, doc_id=doc_id, lang="vi", start=0, end=end, heading_path="", n_tokens=5,
                chunker_version="t", embed_sha1=sha)


def test_export_writes_each_hash_once_and_resumes(tmp_path):
    docs, chunks, out = tmp_path / "documents", tmp_path / "chunks", tmp_path / "embed_inputs"
    _write(docs / "s/a.parquet", [_doc(1, "Nội dung A"), _doc(2, "Nội dung A")], DOC_SCHEMA)
    _write(chunks / "s/a.parquet", [_chunk(1, 10, "h1"), _chunk(2, 10, "h1")], CHUNK_SCHEMA)
    stats = export_embed_inputs(docs, chunks, out, log=lambda *_: None)
    assert stats["unique"] == 1
    rows = pq.read_table(out / "shard-00000.parquet").to_pylist()
    assert rows == [{"input_id": 0, "embed_sha1": "h1", "text": "Tiêu đề Nội dung A", "n_tokens": 5}]

    # crawl increment: only the new hash is exported, the input_id sequence continues in a new shard
    _write(docs / "s/b.parquet", [_doc(3, "Nội dung B")], DOC_SCHEMA)
    _write(chunks / "s/b.parquet", [_chunk(3, 10, "h2")], CHUNK_SCHEMA)
    stats = export_embed_inputs(docs, chunks, out, log=lambda *_: None)
    assert stats["unique"] == 1 and len(list(out.glob("shard-*.parquet"))) == 2
    assert pq.read_table(out / "shard-00001.parquet").column("input_id").to_pylist() == [1]
