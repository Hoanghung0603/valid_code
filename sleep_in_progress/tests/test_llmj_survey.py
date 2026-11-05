import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from llmj_survey import describe, regions


def test_regions_protect_chunks_and_do_not_duplicate_doclist():
    hits = [[i, i, 0.] for i in range(1, 141)]
    dl = list(range(1, 111)) + [115]
    r = regions(hits, dl, {100, 101}, 110)
    assert r['a_100_nq_nochunk'] == list(range(102, 111))
    assert 115 not in r['below_nq_30']
    assert len(r['c_swap30']) == len(set(r['c_swap30']))
    assert len(r['pool']) == 140


def test_describe_empty_and_quantiles():
    assert describe([]) == {'n': 0}
    assert describe([5, 1, 3])['p50'] == 3


def test_no_donors_when_nq_below_boundary():
    hits = [[i, i, 0.] for i in range(1, 141)]
    r = regions(hits, list(range(1, 41)), set(), 40)
    assert len(r['below_nq_30']) == 30
    assert r['c_swap30'] == []
