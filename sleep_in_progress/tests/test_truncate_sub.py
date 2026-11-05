import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from truncate_sub import truncate_sub  # noqa: E402


def test_truncate_keeps_rank_order():
    sub = [{"id": 1, "relevant_docs": [3, 1, 4, 1, 5], "relevant_chunks": []}]
    out, outside = truncate_sub(sub, 3)
    assert out[0]["id"] == 1
    assert out[0]["relevant_docs"] == [3, 1, 4]
    assert outside == []


def test_truncate_keeps_chunks():
    chunks = [{"doc_id": 3, "chunk_text": "a"}, {"doc_id": 1, "chunk_text": "b"}]
    sub = [{"id": 7, "relevant_docs": [3, 1, 4, 5], "relevant_chunks": chunks}]
    out, outside = truncate_sub(sub, 2)
    assert out[0]["relevant_chunks"] == chunks
    assert outside == []


def test_detects_chunks_outside_kept_docs():
    chunks = [{"doc_id": 3, "chunk_text": "a"}, {"doc_id": 5, "chunk_text": "b"}, {"doc_id": 9, "chunk_text": "c"}]
    sub = [{"id": 7, "relevant_docs": [3, 1, 4, 5], "relevant_chunks": chunks}]
    out, outside = truncate_sub(sub, 2)
    assert outside == [(7, 5), (7, 9)]
    assert out[0]["relevant_chunks"] == chunks
    assert out[0]["relevant_docs"] == [3, 1]


def test_n_docs_over_len_keeps_all():
    sub = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": [{"doc_id": 2, "chunk_text": "x"}]}]
    out, outside = truncate_sub(sub, 60)
    assert out[0]["relevant_docs"] == [1, 2]
    assert outside == []
