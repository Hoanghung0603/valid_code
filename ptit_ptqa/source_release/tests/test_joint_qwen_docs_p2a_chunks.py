from __future__ import annotations

import pytest

from scripts.build_joint_qwen_docs_p2a_chunks import is_subsequence, stable_joint_docs


def test_subsequence_preserves_exact_items_and_order() -> None:
    source = [{"doc_id": 1, "chunk_text": "a"}, {"doc_id": 2, "chunk_text": "b"},
              {"doc_id": 1, "chunk_text": "c"}]
    assert is_subsequence([source[0], source[2]], source)
    assert not is_subsequence([source[2], source[1]], source)


def test_joint_docs_keep_rank_head_and_filtered_owners_in_base_order() -> None:
    base = [10, 11, 12, 13, 14]
    ranked = [14, 13, 12, 11, 10]
    assert stable_joint_docs(base, ranked, {10}, 2) == [10, 13, 14]


def test_joint_docs_reject_unknown_owner_or_bad_coverage() -> None:
    with pytest.raises(ValueError):
        stable_joint_docs([1, 2, 3], [3, 2, 1], {9}, 1)
    with pytest.raises(ValueError):
        stable_joint_docs([1, 2, 3], [3, 2], {1}, 1)
