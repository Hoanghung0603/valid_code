import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from combine_sub import combine_sub  # noqa: E402


def test_keeps_doc_order():
    docs_sub = [{"id": 1, "relevant_docs": [3, 1, 4], "relevant_chunks": []}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 4, "chunk_text": "a"}]}]
    out = combine_sub(docs_sub, chunks_sub)
    assert out[0]["relevant_docs"] == [3, 1, 4]


def test_appends_missing_chunk_docs_without_duplicates():
    docs_sub = [{"id": 2, "relevant_docs": [1, 2], "relevant_chunks": []}]
    chunks = [
        {"doc_id": 9, "chunk_text": "x"},
        {"doc_id": 2, "chunk_text": "y"},
        {"doc_id": 7, "chunk_text": "z"},
        {"doc_id": 9, "chunk_text": "w"},
    ]
    chunks_sub = [{"id": 2, "relevant_docs": [], "relevant_chunks": chunks}]
    out = combine_sub(docs_sub, chunks_sub)
    assert out[0]["relevant_docs"] == [1, 2, 9, 7]
    assert len(out[0]["relevant_docs"]) == len({str(d) for d in out[0]["relevant_docs"]})


def test_chunks_identical():
    chunks = [{"doc_id": 5, "chunk_text": "a", "chunk_order": 0}, {"doc_id": 6, "chunk_text": "b"}]
    docs_sub = [{"id": 3, "relevant_docs": [5], "relevant_chunks": [{"doc_id": 0, "chunk_text": "old"}]}]
    chunks_sub = [{"id": 3, "relevant_docs": [9], "relevant_chunks": chunks}]
    out = combine_sub(docs_sub, chunks_sub)
    assert out[0]["relevant_chunks"] == chunks
    assert out[0]["relevant_chunks"] is chunks
    assert out[0]["relevant_docs"] == [5, 6]
