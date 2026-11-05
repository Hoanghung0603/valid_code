import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zhnew_report import fix_idx, load_id2idx, sub_sanity, top30_overlap  # noqa: E402


def test_load_id2idx_and_fix_idx(tmp_path):
    np.save(tmp_path / "ids.npy", np.array([100, 200, 200, 300], np.int64))
    m = load_id2idx(str(tmp_path / "ids.npy"))
    assert m == {100: 0, 200: 1, 300: 3}
    cache = {"1": [[100, -1, 0.5], [200, -1, 0.4], [999, -1, 0.3]]}
    fixed, st = fix_idx(cache, m)
    assert fixed["1"] == [[100, 0, 0.5], [200, 1, 0.4], [999, -1, 0.3]]
    assert st == {"n_queries": 1, "n_hits": 3, "n_missing_idx": 1}


def test_top30_overlap():
    a = {"1": [[1, -1, 1], [2, -1, 1], [3, -1, 1]], "2": [[5, -1, 1]]}
    b = {"1": [[2, -1, 1], [3, -1, 1], [4, -1, 1]], "2": [[6, -1, 1]], "3": [[7, -1, 1]]}
    r = top30_overlap(a, b, n=3)
    assert r["n_queries"] == 2
    assert r["mean"] == 1.0
    assert r["min"] == 0 and r["max"] == 2
    assert top30_overlap({}, {}, n=3) == {"n_queries": 0}


def test_sub_sanity_full():
    sub = [
        {"id": 1, "relevant_docs": [7, 8], "relevant_chunks": [{"doc_id": 7, "chunk_text": "x"},
                                                              {"doc_id": 9, "chunk_text": "y"}]},
        {"id": 2, "relevant_docs": [8], "relevant_chunks": []},
    ]
    s = sub_sanity(sub, allowed={7, 8})
    assert s["n_queries"] == 2
    assert s["n_missing_query_ids"] == 1198
    assert s["chunk_doc_not_in_relevant_docs"] == 1
    assert s["n_chunks"] == 2
    assert s["doc_in_new_zh_frac"] == 1.0
    assert s["doc_per_q_mean"] == 1.5
    assert s["chunk_per_q_mean"] == 1.0


def test_sub_sanity_empty_allowed_none():
    s = sub_sanity([{"id": 1, "relevant_docs": [], "relevant_chunks": []}])
    assert s["n_queries"] == 1
    assert "doc_in_new_zh_frac" not in s
    assert s["doc_per_q_min"] == 0 and s["doc_per_q_max"] == 0
