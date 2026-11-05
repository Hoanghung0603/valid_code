import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vi41c import merge_row, use_chunk_source


def row(docs, chunks):
    return {'id': 1, 'relevant_docs': docs,
            'relevant_chunks': [{'doc_id': d, 'chunk_text': text} for d, text in chunks]}


def test_no_missing_preserves_candidate_docs_and_exact_chunks():
    ref = row([1, 2, 3], [(1, 'á\n中'), (1, 'b')])
    cand = row([3, 1, 4], [(4, 'không dùng')])
    originals = copy.deepcopy((ref, cand))
    merged, m, short = merge_row(ref, cand)
    assert merged == row([3, 1, 4], [(1, 'á\n中'), (1, 'b')])
    assert (m, short) == (0, 0)
    assert (ref, cand) == originals


def test_missing_appended_in_ref_doc_order_removes_tail_unprotected():
    ref = row([5, 1, 4, 2, 3], [(4, 'first'), (5, 'second'), (2, 'third')])
    cand = row([8, 2, 7, 6, 9], [])
    merged, m, short = merge_row(ref, cand)
    assert merged['relevant_docs'] == [8, 2, 7, 5, 4]
    assert merged['relevant_chunks'] == ref['relevant_chunks']
    assert (m, short) == (2, 0)


def test_insufficient_unprotected_docs_can_grow():
    ref = row([1, 2, 3], [(1, 'a'), (2, 'b'), (3, 'c')])
    merged, m, short = merge_row(ref, row([1], []))
    assert merged['relevant_docs'] == [1, 2, 3]
    assert (m, short) == (2, 2)


def test_protected_never_removed_even_when_late():
    ref = row([1, 2, 3, 4], [(3, 'z'), (4, 'x')])
    merged, m, short = merge_row(ref, row([9, 8, 3, 7], []))
    assert merged['relevant_docs'] == [9, 8, 3, 4]
    assert (m, short) == (1, 0)


def test_reject_invalid_inputs():
    with pytest.raises(ValueError, match='duplicate doc'):
        merge_row(row([1], []), row([1, '1'], []))
    with pytest.raises(ValueError, match='outside docs'):
        merge_row(row([1], [(2, 'x')]), row([1], []))


def test_alternate_chunks_keep_reference_order_and_reject_changed_doc_set():
    ref = row([3, 1, 2], [(1, 'old'), (3, 'old')])
    source = row([1, 3], [(3, 'mới'), (1, 'a'), (3, 'bridge')])
    merged, missing, short = merge_row(use_chunk_source(ref, source), row([2, 4, 1], []))
    assert merged == row([2, 1, 3], [(3, 'mới'), (1, 'a'), (3, 'bridge')])
    assert (missing, short) == (1, 0)
    with pytest.raises(ValueError, match='chunk document set differs'):
        use_chunk_source(ref, row([1, 2, 3], [(1, 'x'), (2, 'bridge'), (3, 'y')]))


def test_changed_chunk_docs_protect_new_docs_in_reference_order():
    ref = row([5, 1, 4, 2, 3], [(1, 'old')])
    source = row([5, 1, 4, 2, 3], [(4, 'new-first'), (5, 'new-second')])
    merged, missing, short = merge_row(
        use_chunk_source(ref, source, allow_changed_chunk_docs=True), row([8, 1, 7, 6, 9], []))
    assert merged == row([8, 1, 7, 5, 4], [(4, 'new-first'), (5, 'new-second')])
    assert (missing, short) == (2, 0)
    with pytest.raises(ValueError, match='outside reference docs'):
        use_chunk_source(ref, row([99], [(99, 'bad')]), allow_changed_chunk_docs=True)
