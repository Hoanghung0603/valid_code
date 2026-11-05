import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dwx_merge import (  # noqa: E402
    dist3,
    dq_docs_per_q,
    filter_new_rows,
    find_new_file,
    impact_stats,
    merge_q,
    new_doc_stats,
    pct,
    rankwise_logits,
    sort_rows,
    write_zst_stream,
)


def test_pct_and_dist3():
    vals = [3.0, 1.0, 2.0, 4.0]
    assert pct(vals, 0.0) == 1.0
    assert pct(vals, 0.5) == 3.0
    assert pct(vals, 1.0) == 4.0
    d = dist3(vals)
    assert d == {"p25": 2.0, "p50": 3.0, "p75": 3.0}
    import math
    assert math.isnan(dist3([])["p50"])


def test_sort_rows_logit_desc_tiebreak():
    rows = [
        [9, 2, 1, 1.0, "b"],
        [8, 1, 5, 5.0, "a"],
        [7, 3, 0, 1.0, "c"],
        [6, 2, 0, 1.0, "d"],
    ]
    out = sort_rows(rows)
    assert [r[4] for r in out] == ["a", "d", "b", "c"]


def test_merge_q_keeps_all_sorted():
    old = [[1, 1, 0, 2.0, "x"], [1, 1, 1, 4.0, "y"]]
    new = [[2, 30, 0, 3.0, "z"]]
    out = merge_q(old, new)
    assert [r[4] for r in out] == ["y", "z", "x"]
    assert len(out) == 3


def test_find_new_file(tmp_path):
    d = tmp_path / "ext"
    d.mkdir()
    try:
        find_new_file(d)
        assert False, "phải lỗi khi chưa có file"
    except FileNotFoundError:
        pass
    (d / "chunk_dw_u900.json").write_text("{}", encoding="utf-8")
    assert find_new_file(d).name == "chunk_dw_u900.json"
    (d / "chunk_dw_u900.json.zst").write_bytes(b"x")
    assert find_new_file(d).name == "chunk_dw_u900.json.zst"


def test_write_zst_stream_roundtrip(tmp_path):
    import zstandard
    data = {"1": [[10, 1, 0, 1.5, "tiếng Việt"]], "2": []}
    p = write_zst_stream(tmp_path / "out.json.zst", data)
    with open(p, "rb") as f:
        raw = zstandard.ZstdDecompressor().stream_reader(f).read()
    assert json.loads(raw) == data


def test_rankwise_logits():
    rows_by_q = {
        "1": [[1, 1, 0, 0.5, "a"], [2, 2, 0, 1.5, "b"]],
        "2": [[3, 1, 0, 2.5, "c"]],
    }
    out = rankwise_logits(rows_by_q, ranks=[1, 2])
    assert out[1] == [1.5, 2.5]
    assert out[2] == [0.5]


def test_new_doc_stats():
    new = {"1": [[10, 1, 0, 1.0, "a"], [10, 1, 1, 0.9, "b"], [11, 2, 0, 0.8, "c"]],
           "2": []}
    nc, nd = new_doc_stats(new)
    assert nc == 1.5
    assert nd == 1.0
    nc, nd = new_doc_stats(new, nq=4)
    assert nc == 0.75
    assert nd == 0.5


def test_impact_stats_counts_new_in_topk():
    merged = {"1": [[10, 1, 0, 5.0, "cu a"], [20, 2, 0, 4.0, "moi b"], [21, 3, 0, 3.0, "moi c"]]}
    new = {"1": [[20, 2, 0, 4.0, "moi b"], [21, 3, 0, 3.0, "moi c"]]}
    fused = {"1": [[10, 0, 1.0], [20, 0, 10.0], [21, 0, 10.0]]}
    ref = [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}] * 2}]
    rep = impact_stats(merged, new, fused, ref, wd=4.0)
    # zc = [1.22, 0, -1.22], zd = [-1.41, 0.71, 0.71] -> s = [-4.43, 2.83, 1.60]
    # top-2 theo s = hai chunk mới
    assert rep["TB k_q (số chunk bài cũ)"] == 2.0
    assert rep["TB chunk mới lọt top-k/câu"] == 2.0
    assert rep["câu có chunk mới lọt"] == 1
    assert rep["phân bố số mới lọt"]["max"] == 2


def test_impact_stats_k1_picks_best_new():
    merged = {"1": [[10, 1, 0, 5.0, "cu a"], [20, 2, 0, 4.0, "moi b"], [21, 3, 0, 3.0, "moi c"]]}
    new = {"1": [[20, 2, 0, 4.0, "moi b"], [21, 3, 0, 3.0, "moi c"]]}
    fused = {"1": [[10, 0, 1.0], [20, 0, 10.0], [21, 0, 10.0]]}
    ref = [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}]}]
    rep = impact_stats(merged, new, fused, ref, wd=4.0)
    assert rep["TB chunk mới lọt top-k/câu"] == 1.0
    assert rep["TB chunk mới lọt / k_q"] == 1.0


def test_impact_stats_old_wins_without_fused_boost():
    merged = {"1": [[10, 1, 0, 5.0, "cu a"], [20, 2, 0, 4.0, "moi b"]]}
    new = {"1": [[20, 2, 0, 4.0, "moi b"]]}
    fused = {"1": [[10, 0, 5.0], [20, 0, 4.0]]}
    ref = [{"id": 1, "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}]}]
    rep = impact_stats(merged, new, fused, ref, wd=4.0)
    assert rep["TB chunk mới lọt top-k/câu"] == 0.0
    assert rep["câu có chunk mới lọt"] == 0


def test_dq_docs_per_q_excludes_pool_top_n():
    cands = {"1": [[50, 1], [10, 2], [99, 3]], "2": [[20, 4], [21, 5]]}
    pool = {"1": [[10, 2, 1.0], [11, 6, 0.9]], "2": [[20, 4, 1.0]]}
    dq = dq_docs_per_q(cands, pool, top_n=1)
    assert dq == {"1": {"50", "99"}, "2": {"21"}}


def test_filter_new_rows_keeps_only_dq():
    new = {
        "1": [[50, 1, 0, 1.0, "a"], [10, 2, 0, 2.0, "b"], [50, 1, 1, 0.5, "c"]],
        "2": [[20, 1, 0, 3.0, "d"]],
    }
    dq = {"1": {"50", "99"}, "2": {"21"}}
    kept, n_keep, n_drop = filter_new_rows(new, dq)
    assert kept == {"1": [[50, 1, 0, 1.0, "a"], [50, 1, 1, 0.5, "c"]]}
    assert n_keep == 2
    assert n_drop == 2


def test_filter_new_rows_unknown_query_dropped():
    kept, n_keep, n_drop = filter_new_rows({"9": [[1, 1, 0, 1.0, "x"]]}, {"1": set()})
    assert kept == {}
    assert (n_keep, n_drop) == (0, 1)
