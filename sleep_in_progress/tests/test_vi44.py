import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi44 as v


def test_all_ranks_and_top50_exact_reproduction():
    text = ' '.join(f'từ{i}' for i in range(2200))
    chunks, _ = v.v.bridge_geometry(text)
    docs = list(range(1, 123))
    row = {'id': 1, 'relevant_docs': docs, 'relevant_chunks': [
        {'doc_id': d, 'chunk_text': chunks[0].text} for d in (1, 50, 51, 100, 101, 122)]}
    texts = {d: text for d in docs}
    wins = {str(d): [[0, len(text), d/100]] for d in docs}
    got = v.select_bridges(row, texts, wins)
    old = v.v.select_bridges(row, texts, wins)
    assert [b for b in got if b['rank'] <= 50] == old
    assert [b['rank'] for b in got] == [122, 101, 100, 51, 50, 1]
    assert v.rank_buckets({'1': got}) == {'1-50': 2, '51-100': 2, '101+': 2}
    assert v.rank_buckets({'1': got}, 1) == {'1-50': 0, '51-100': 1, '101+': 2}
    assert v.ordered_subsequence(old, got)


@pytest.mark.parametrize('a,b,want', [([], [], True), ([1, 3], [1, 2, 3], True),
    ([1, 3], [3, 1], False), ([1, 1], [1], False)])
def test_ordered_subsequence_not_unordered_subset(a, b, want):
    assert v.ordered_subsequence(a, b) is want


def test_relationships_1200_and_nonprefix_vi42():
    ref = [{'id': q, 'relevant_docs': [1, 2], 'relevant_chunks': [
        {'doc_id': 1, 'chunk_text': 'old'}, {'doc_id': 2, 'chunk_text': 'other'}]} for q in range(1, 1201)]
    bs = {str(q): [{'doc_id': 2, 'chunk_text': 'new', 'score': 2},
                   {'doc_id': 1, 'chunk_text': 'bridge42', 'score': 1}] for q in range(1, 1201)}
    oldbs = {q: rows[1:] for q, rows in bs.items()}
    ref42 = v.v.append_bridges(ref, oldbs)
    sub = v.v.append_bridges(ref, bs)
    assert v.check_relations(sub[::-1], ref, ref42, bs, 0)['vi42_ordered_subsequence'] == 1200
    with pytest.raises(AssertionError):
        v.check_relations(v.v.append_bridges(ref, bs, 2), ref, ref42, bs, 2)


def test_final_checks_real_validate_and_zip_1200(tmp_path, monkeypatch):
    rows = [{'id': q, 'relevant_docs': [7], 'relevant_chunks': [
        {'doc_id': 7, 'chunk_text': 'nguyên văn'}]} for q in range(1, 1201)]
    zp = tmp_path/'submission_test.zip'
    v.v.write_reordered_zip(rows, zp)
    monkeypatch.setattr(v.v.dw, 'read', lambda path: [] if path.name in {'bad_ids.json', 'new_ids.json'}
                        else v.json.loads(path.read_text(encoding='utf-8')))
    monkeypatch.setattr(v.v.vi29, 'load_corpus_ids', lambda path: {7})
    report = v.final_checks(rows, zp, 'test', tmp_path)
    assert report['validate'] == report['check_zip'] == report['crc'] == 0
    assert all(n == 0 for k, n in report['assert_clean'].items() if k != 'cau')
    monkeypatch.setattr(v.v.vi29, 'load_corpus_ids', lambda path: {8})
    with pytest.raises(AssertionError):
        v.final_checks(rows, zp, 'test', tmp_path)


def test_deep_rank_gate_inclusive_zero_and_missing_cache():
    text = ' '.join(f'từ{i}' for i in range(2200))
    chunks, _ = v.v.bridge_geometry(text)
    row = {'id': 1, 'relevant_docs': list(range(1, 102)), 'relevant_chunks': [
        {'doc_id': 101, 'chunk_text': chunks[0].text}]}
    assert v.select_bridges(row, {101: text}, {'101': [[0, len(text), -0.001]]}) == []
    result = v.select_bridges(row, {101: text}, {'101': [[0, len(text), 0]]})
    assert len(result) == 1 and result[0]['rank'] == 101 and result[0]['score'] == 0
    assert len(row['relevant_chunks']) == 1
    with pytest.raises(AssertionError, match='missing cached windows'):
        v.select_bridges(row, {101: text}, {})
