import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zh_probe_stats import (compare_top1, examples, host_counts, percentiles, pick_example_qids,
                            rank_scores, score_dist, top_docs)

ZH = {
    "1": [[10, 0, 0.6], [11, 1, 0.5], [12, 2, 0.4]],
    "2": [[11, 1, 0.3], [13, 3, 0.2], [10, 0, 0.1]],
    "3": [[14, 4, 0.5]],
}
VI = {"1": [[100, 0, 0.7]], "2": [[101, 1, 0.2]], "9": [[102, 2, 0.9]]}


def test_percentiles_basic_and_empty():
    p = percentiles([0.1, 0.2, 0.3, 0.4, 0.5])
    assert p["p0"] == 0.1 and p["p50"] == 0.3 and p["p100"] == 0.5
    assert abs(p["mean"] - 0.3) < 1e-9
    assert percentiles([]) == {}


def test_rank_scores_skips_short_lists():
    assert rank_scores(ZH, 1) == {"1": 0.6, "2": 0.3, "3": 0.5}
    assert rank_scores(ZH, 3) == {"1": 0.4, "2": 0.1}


def test_score_dist_keys():
    d = score_dist(ZH, 3)
    assert set(d) == {"top1", "top3"}
    assert d["top1"]["p100"] == 0.6


def test_compare_top1_common_only():
    c = compare_top1(ZH, VI)
    assert c["n"] == 2
    assert c["zh_gt_vi"] == 1  # câu 2: 0.3 > 0.2; câu 1: 0.6 < 0.7


def test_top_docs_and_hosts():
    assert top_docs(ZH, 2) == {10: 0, 11: 1, 13: 3, 14: 4}
    hosts = {10: "a.cn", 11: "a.cn", 13: "b.cn"}
    hc = host_counts(ZH, hosts, 2)
    assert hc[0] == ("a.cn", 3, 0.6)
    assert ("?", 1, 0.2) in hc  # doc 14 không có host


def test_pick_example_qids_spread_and_examples():
    q = pick_example_qids(ZH, 3)
    assert q == ["1", "3", "2"]  # cao, giữa, thấp
    ex = examples(ZH, {"1": "câu một"}, {10: "标题"}, ["1"], 2)
    assert ex[0]["query"] == "câu một"
    assert ex[0]["docs"] == [[10, 0.6, "标题"], [11, 0.5, ""]]
    assert pick_example_qids({}, 3) == []
