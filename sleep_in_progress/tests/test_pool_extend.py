import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pool_extend import collect_new, extend_pairs, per_query_sizes


def test_collect_new_order_exclusion_and_dedup():
    dense = [[1, 10, 0.9], [2, 20, 0.8], [3, 30, 0.7], [4, 40, 0.6]]
    bm25 = [[3, 30, 5.0], [5, 50, 4.0], [2, 20, 3.0], [6, 60, 2.0]]
    new, mm = collect_new(dense, bm25, have={2}, dense_k=4, bm_k=4)
    assert new == [[1, 10], [3, 30], [4, 40], [5, 50], [6, 60]]
    assert mm == []
    new, _ = collect_new(dense, bm25, have={2}, dense_k=2, bm_k=1)
    assert new == [[1, 10], [3, 30]]


def test_collect_new_idx_first_source_and_mismatch():
    dense = [[7, 1, 0.5], [8, 2, 0.4]]
    bm25 = [[7, 99, 1.0], [9, 3, 0.9]]
    new, mm = collect_new(dense, bm25, have=set(), dense_k=2, bm_k=2)
    assert new == [[7, 1], [8, 2], [9, 3]]
    assert mm == [(7, 1, 99)]
    new, _ = collect_new([], [[7, 99, 1.0]], have=set(), dense_k=2, bm_k=2)
    assert new == [[7, 99]] and _ == []


def test_extend_pairs_without_fallback():
    dense = {"1": [[1, 1, 0.9], [2, 2, 0.8], [3, 3, 0.7]]}
    bm25 = {"1": [[4, 4, 9.0], [2, 2, 8.0]]}
    old = {"1": [[2, 2, 1.0]]}
    pairs, n0, used_fb, n1, nmis = extend_pairs(dense, bm25, old, dense_k=3, bm_k=2,
                                                max_pairs=100, dense_k_fallback=1)
    assert pairs == {"1": [[1, 1], [3, 3], [4, 4]]}
    assert (n0, used_fb, n1, nmis) == (3, False, 3, 0)


def test_extend_pairs_fallback_when_over_max():
    dense = {"1": [[i, i, 0.9] for i in range(1, 5)]}
    bm25 = {"1": [[10 + i, 10 + i, 9.0] for i in range(3)]}
    old = {"1": []}
    pairs, n0, used_fb, n1, nmis = extend_pairs(dense, bm25, old, dense_k=4, bm_k=3,
                                                max_pairs=5, dense_k_fallback=2)
    assert used_fb and n0 == 7 and n1 == 5
    assert pairs == {"1": [[1, 1], [2, 2], [10, 10], [11, 11], [12, 12]]}
    pairs, n0, used_fb, n1, _ = extend_pairs(dense, bm25, old, dense_k=4, bm_k=3,
                                             max_pairs=7, dense_k_fallback=2)
    assert not used_fb and n0 == n1 == 7
    assert len(pairs["1"]) == 7


def test_per_query_sizes():
    lo, avg, hi = per_query_sizes({"1": [[1, 1]], "2": [], "3": [[2, 2], [3, 3]]})
    assert (lo, hi) == (0, 2)
    assert avg == 1.0


def test_main_writes_cands_and_reports(tmp_path, capsys, monkeypatch):
    from pool_extend import main
    dense = {"1": [[1, 1, 0.9], [2, 2, 0.8]], "2": [[5, 5, 0.7]]}
    bm25 = {"1": [[3, 3, 9.0]], "2": [[5, 5, 8.0], [6, 6, 7.0]]}
    old = {"1": [[2, 2, 0.1]], "2": []}
    p = tmp_path
    (p / "dense.json").write_text(json.dumps(dense), encoding="utf-8")
    (p / "bm25.json").write_text(json.dumps(bm25), encoding="utf-8")
    (p / "old.json").write_text(json.dumps(old), encoding="utf-8")
    out = p / "cands.json"
    monkeypatch.setattr(sys, "argv", [
        "pool_extend.py", "--dense", str(p / "dense.json"), "--bm25", str(p / "bm25.json"),
        "--old", str(p / "old.json"), "--out", str(out),
        "--dense-k", "2", "--bm-k", "2", "--max-pairs", "1", "--dense-k-fallback", "1",
    ])
    main()
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res == {"1": [[1, 1], [3, 3]], "2": [[5, 5], [6, 6]]}
    txt = capsys.readouterr().out
    assert "VƯỢT --max-pairs" in txt and "4 cặp" in txt
