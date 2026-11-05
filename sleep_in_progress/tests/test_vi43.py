import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi43 as v


def fixture():
    a = [{'id': q, 'relevant_docs': [1, 2, 3], 'relevant_chunks': [
        {'doc_id': 1, 'chunk_text': 'old'}]} for q in range(1, 1201)]
    b = [{**x, 'relevant_chunks': x['relevant_chunks'] + [
        {'doc_id': 1, 'chunk_text': 'bridge'}]} for x in a]
    c = [{'id': q, 'relevant_docs': [4, 1, 2]} for q in range(1, 1201)]
    return a, b, c


def test_combine_preserves_chunks_order_and_vi41c_docs():
    a, b, c = fixture()
    out, rep = v.combine(a, b[::-1], c)
    assert out[0]['id'] == 1200
    assert all(x['relevant_docs'] == [4, 1, 2] for x in out)
    assert out[0]['relevant_chunks'] == b[-1]['relevant_chunks']
    assert rep['missing_chunk_docs_appended'] == 0
    assert rep['chunks_equal_vi42'] == rep['chunk_bearing_doc_sets_equal_vi37'] == 1200


def test_combine_rejects_new_chunk_bearing_document():
    a, b, c = fixture()
    b[0]['relevant_chunks'].append({'doc_id': 2, 'chunk_text': 'not allowed'})
    with pytest.raises(AssertionError, match='chunk-bearing docs changed'):
        v.combine(a, b, c)


def test_combine_rejects_wrong_candidate_count_and_missing_protected_docs():
    a, b, c = fixture()
    c[0]['relevant_docs'] = [4, 1]
    with pytest.raises(AssertionError):
        v.combine(a, b, c)
    c[0]['relevant_docs'] = [4, 2, 3]
    with pytest.raises(AssertionError):
        v.combine(a, b, c)
