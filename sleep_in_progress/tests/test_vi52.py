import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import vi50
import vi51
import vi52


def example():
    data = {'pool': {'1': [[1, 0, 5], [2, 1, 4], [3, 2, 3], [4, 3, 2]]},
            'q8b': {'1': [[2, 1, 2], [1, 0, 1]]},
            'dense': {'1': [[3, 2, 1], [4, 3, 0]]},
            'hyde': {'1': [[4, 3, 1], [1, 0, 0]]}}
    return data, {'1': {'1': 0.4, '3': 0.4}}, {'1': {'2': 1., '4': 0.}}, {'1': {'1': 1., '2': 1., '3': 0., '4': -1.}}


def test_ablation_exactly_reuses_original_rules():
    data, bge, d8w, q3e = example()
    assert vi52.fused(data, bge, d8w=d8w) == vi50.fused(data, bge, d8w)
    assert vi52.fused(data, bge, q3e=q3e) == vi51.fused(data, bge, q3e)


def test_both_sources_and_missing_score_contributions():
    data, bge, d8w, q3e = example()
    both = vi52.fused(data, bge, d8w, q3e)['1']
    one = {h[0]: h[2] for h in vi52.fused(data, bge, d8w)['1']}
    for rank, h in enumerate([1, 2, 3, 4], 1):
        score = next(row[2] for row in both if row[0] == h)
        assert abs(score - one[h] - 1 / (60 + rank)) < 1.1e-8


def test_protection_and_counts():
    base = [{'id': 1, 'relevant_docs': [1, 2], 'relevant_chunks': [{'doc_id': 2, 'chunk_text': 'x'}]}]
    actual = vi52.doclists(base, {'1': [[3, 0, 3], [4, 0, 2], [2, 0, 1]]})
    assert actual == {1: [3, 2]}
    assert base[0]['relevant_chunks'] == [{'doc_id': 2, 'chunk_text': 'x'}]


def test_compare_order_and_set():
    result = vi52.compare({1: [1, 2], 2: [1, 3]}, {1: [2, 1], 2: [1, 2]})
    assert result == {'queries': 2, 'exact': 0, 'changed_sets': 1, 'docs_entering': 1, 'docs_entering_per_query': .5}


def test_swappable_cosine_uses_same_fusion_and_missing_scores():
    data, bge, d8w, _ = example()
    sparse_cosine = {'1': {'3': .7, '1': .3}}
    assert vi52.fused(data, bge, q3e=sparse_cosine) == vi51.fused(data, bge, sparse_cosine)
    assert vi52.fused(data, bge, d8w, sparse_cosine) != vi52.fused(data, bge, d8w)


def test_build_defaults_keep_vi52_sources():
    defaults = vi52.build.__defaults__
    assert defaults[0] == vi52.TEMP / 'vi51/cosines_vi51a.json'
    assert defaults[1] == vi52.TEMP / 'vi51a/submission_vi51a.zip'
    assert defaults[2:] == ('vi52', None, None)
