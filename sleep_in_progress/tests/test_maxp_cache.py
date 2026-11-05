import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from maxp_cache import build_maxp, doc_max_scores, overlap_stats, overlap_topk  # noqa: E402


def test_doc_max_scores():
    wins = {"1": [[0, 5, -1.0], [5, 9, 2.5]], "2": [[0, 5, 0.5]], "3": []}
    assert doc_max_scores(wins) == {1: 2.5, 2: 0.5}


def test_build_maxp_sorts_by_max_logit():
    dw = {"7": {"10": [[0, 5, 0.1]], "20": [[0, 5, 3.0]], "30": [[0, 5, 1.0]]}}
    out = build_maxp(dw)
    assert out["7"] == [[20, 0, 3.0], [30, 0, 1.0], [10, 0, 0.1]]


def test_build_maxp_tie_break_by_cands_idx():
    dw = {"7": {"10": [[0, 5, 1.0]], "20": [[0, 5, 1.0]], "30": [[0, 5, 1.0]]}}
    cands = {"7": [[20, 9, 0], [30, 8, 0], [10, 7, 0]]}
    out = build_maxp(dw, cands)
    assert [r[0] for r in out["7"]] == [20, 30, 10]
    assert [r[1] for r in out["7"]] == [9, 8, 7]


def test_build_maxp_skips_docs_without_windows():
    dw = {"7": {"10": [], "20": [[0, 5, 1.0]]}}
    out = build_maxp(dw)
    assert [r[0] for r in out["7"]] == [20]


def test_overlap_topk():
    a = [[1, 0, 3.0], [2, 0, 2.0], [3, 0, 1.0], [4, 0, 0.5]]
    b = [[3, 0, 3.0], [2, 0, 2.0], [9, 0, 1.0]]
    assert overlap_topk(a, b, 3) == 2
    assert overlap_topk(a, b, 1) == 0


def test_overlap_stats():
    maxp = {"1": [[1, 0, 1.0], [2, 0, 0.5]], "2": [[5, 0, 1.0]]}
    ref = {"1": [[2, 0, 1.0], [9, 0, 0.5]], "2": [[5, 0, 1.0], [6, 0, 0.5]]}
    st = overlap_stats(maxp, ref, 30)
    assert st["câu"] == 2
    assert st["trung bình"] == 1.0
    assert st["min"] == 1 and st["max"] == 1
