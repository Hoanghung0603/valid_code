from __future__ import annotations

import pytest

from scripts.build_best_axis_recombination import compose_row


def test_compose_uses_only_requested_axes() -> None:
    docs = {"id": 1, "relevant_docs": [2, 3], "relevant_chunks": [{"doc_id": 2, "chunk_text": "old"}]}
    chunks = {"id": 1, "relevant_docs": [9], "relevant_chunks": [{"doc_id": 9, "chunk_text": "new"}]}
    assert compose_row(docs, chunks) == {
        "id": 1, "relevant_docs": [2, 3],
        "relevant_chunks": [{"doc_id": 9, "chunk_text": "new"}],
    }


def test_compose_rejects_bad_ids_and_duplicate_docs() -> None:
    with pytest.raises(ValueError):
        compose_row({"id": 1, "relevant_docs": [2], "relevant_chunks": []},
                    {"id": 2, "relevant_docs": [], "relevant_chunks": [{"doc_id": 2, "chunk_text": "x"}]})
    with pytest.raises(ValueError):
        compose_row({"id": 1, "relevant_docs": [2, 2], "relevant_chunks": []},
                    {"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 2, "chunk_text": "x"}]})
