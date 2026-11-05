import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pool3 import (  # noqa: E402
    build_pool3,
    cmp_cache,
    cmp_sub,
    cmplogit,
    collect_idx,
    doclist,
    extend_source,
    hit_map,
    load_cache,
    missing_pairs,
    sanity_pairs,
)


def test_load_cache_str_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({1: [[2, 0, 0.5]]}), encoding="utf-8")
    assert load_cache(p) == {"1": [[2, 0, 0.5]]}


def test_hit_map_first_occurrence_wins():
    hits = [[5, 1, 0.5], [3, 2, 0.4], [5, 9, 0.9]]
    m = hit_map(hits)
    assert m[5] == [5, 1, 0.5]
    assert m[3] == [3, 2, 0.4]
    assert hit_map([]) == {}


def test_collect_idx_ok_and_conflict():
    a = {"1": [[5, 10, 0.5], [6, 11, 0.4]]}
    b = {"2": [[6, 11, 0.3], [7, 12, 0.2]]}
    idx_of, bad = collect_idx([a, b])
    assert idx_of == {5: 10, 6: 11, 7: 12}
    assert bad == []
    _, bad = collect_idx([a, {"1": [[6, 99, 0.1]]}])
    assert bad == [6]


def test_missing_pairs_counts_and_no_idx():
    new = {"1": [10, 11, 12, 13], "2": [20]}
    known = {"1": {10: [10, 1, 0.5], 11: [11, 2, None]}}
    idx_of = {10: 1, 11: 2, 12: 3, 20: 7}
    cands_m, cands_a, rep = missing_pairs(new, known, idx_of)
    assert cands_m == {"1": [[11, 2], [12, 3]], "2": [[20, 7]]}
    assert cands_a == {"1": [[10, 1], [11, 2], [12, 3]], "2": [[20, 7]]}
    assert rep["cap_thieu_logit"] == 3
    assert rep["cap_new"] == 4
    assert rep["da_co_logit"] == 1
    assert rep["so_thieu_idx"] == 1
    assert rep["thieu_idx_mau"] == [13]
    assert rep["so_cau"] == 2


def test_sanity_pairs_old_and_new():
    pool = {"1": [[1, 10, 0.5], [2, 11, 0.4], [3, 12, None]]}
    p2 = {"1": [[9, 20, 0.3], [1, 10, 0.5], [8, 21, 0.2]]}
    pairs, ref = sanity_pairs(pool, p2, n_q=5, per=1)
    assert pairs == {"1": [[1, 10], [9, 20]]}
    assert ref == {"1": {"1": 0.5, "9": 0.3}}


def test_build_pool3_merge_prefers_known_then_rerank():
    pool = {"1": [[1, 10, 2.0], [2, 11, 1.0]]}
    new = {"1": [3, 4, 5]}
    known = {"1": {3: [3, 30, -0.5], 4: [4, 40, None]}}
    rerank = {"1": [[4, 40, -1.5]]}
    idx_of = {3: 30, 4: 40, 5: 50}
    out, rep = build_pool3(pool, new, known, rerank, idx_of)
    assert out["1"] == [[1, 10, 2.0], [2, 11, 1.0], [3, 30, -0.5], [4, 40, -1.5],
                        [5, 50, None]]
    assert rep["new_thieu_logit_None"] == 1
    assert rep["TB_pool_cu"] == 2.0
    assert rep["TB_new_them"] == 3.0
    assert rep["so_loi"] == 0


def test_build_pool3_flags_dup_with_pool():
    pool = {"1": [[1, 10, 2.0]]}
    out, rep = build_pool3(pool, {"1": [1, 2]}, {}, {}, {2: 20})
    assert out["1"] == [[1, 10, 2.0], [2, 20, None]]
    assert rep["so_loi"] == 1
    assert rep["loi"][0].startswith("cau 1")


def test_extend_source_keeps_base_and_appends_sorted():
    base = {"1": [[1, 10, 0.9], [2, 11, 0.5]]}
    new_scores = {"1": [[3, 30, 0.7], [4, 40, 0.1], [1, 10, 9.9], [5, 50, 0.3]]}
    out, rep = extend_source(base, new_scores)
    assert out["1"] == [[1, 10, 0.9], [2, 11, 0.5], [3, 30, 0.7], [5, 50, 0.3], [4, 40, 0.1]]
    assert rep["TB_new_noi_duoi"] == 3.0
    assert rep["new_chen_len_tren_goc"] == 1
    assert rep["mau_chen"][0][0] == "1"


def test_extend_source_none_score_last():
    base = {"1": [[1, 10, 0.9]]}
    out, rep = extend_source(base, {"1": [[2, 20, None], [3, 30, 0.2]]})
    assert out["1"] == [[1, 10, 0.9], [3, 30, 0.2], [2, 20, None]]
    assert rep["new_chen_len_tren_goc"] == 0


def test_doclist_n_q_and_new_stats():
    pool3 = {"1": [[1, 1, -1.0], [2, 2, -2.5], [3, 3, None]],
             "2": [[4, 4, -0.5]]}
    rank = {"1": [[2, 2, 0.9], [1, 1, 0.8], [3, 3, 0.7]],
            "2": [[4, 4, 0.6], [5, 5, 0.5]]}
    new_ids = {"1": {2, 3}, "2": {5}}
    meta = {"1": {2: ["hyde"], 3: ["dense", "bm25"]}, "2": {5: ["pool2"]}}
    out, rep = doclist(pool3, rank, new_ids, meta, thr=-2.0, lo=2, hi=10)
    assert out == {"1": [2, 1], "2": [4, 5]}
    assert rep["TB_doc_cau"] == 2.0
    assert rep["TB_new_cau"] == 1.0
    assert rep["phan_bo_nguon_new"] == {"hyde": 0.5, "pool2": 0.5}
    assert rep["TB_so_nguon_moi_new"] == 1.0
    assert rep["thr"] == -2.0


def test_doclist_clamp_bounds():
    pool3 = {"1": [[i, i, -0.5] for i in range(1, 8)]}
    rank = {"1": [[i, i, 1.0 - i * 0.01] for i in range(1, 8)]}
    out, rep = doclist(pool3, rank, {"1": set()}, {"1": {}}, thr=-2.0, lo=3, hi=5)
    assert out["1"] == [1, 2, 3, 4, 5]
    assert rep["min_doc"] == 5 and rep["max_doc"] == 5
    out, rep = doclist(pool3, rank, {"1": set()}, {"1": {}}, thr=2.0, lo=3, hi=5)
    assert out["1"] == [1, 2, 3]
    assert rep["min_doc"] == 3 and rep["max_doc"] == 3


def test_cmp_cache_exact_and_partial():
    a = {"1": [[1, 0, 1.0], [2, 0, 0.9]], "2": [[5, 0, 1.0]]}
    b = {"1": [[1, 0, 1.0], [2, 0, 0.9]], "2": [[6, 0, 1.0], [5, 0, 0.8]]}
    rep = cmp_cache(a, a, top=2)
    assert rep["ty_le_khop"] == 1.0 and rep["so_cau_khop"] == 2
    rep = cmp_cache(a, b, top=2)
    assert rep["ty_le_khop"] == 0.5
    assert rep["lech_dau_trung_vi"] == 0
    rep = cmp_cache(a, {"1": [[1, 0, 1.0]]}, top=2)
    assert rep["cau_thieu"] == 1


def test_cmp_cache_plain_doc_id_lists():
    a = {"1": [1, 2, 3]}
    b = {"1": [[1, 0, 1.0], [2, 0, 0.9], [3, 0, 0.8]]}
    assert cmp_cache(a, b, top=3)["ty_le_khop"] == 1.0


def test_cmp_sub_exact_ratio_and_missing():
    a = [{"id": 1, "relevant_docs": [7, 8]}, {"id": 2, "relevant_docs": [9]}, {"id": 3}]
    b = [{"id": 1, "relevant_docs": [7, 8]}, {"id": 2, "relevant_docs": [9, 10]}]
    rep = cmp_sub(a, b)
    assert rep["so_cau"] == 3
    assert rep["so_cau_khop_tuyet_doi"] == 1
    assert rep["ty_le_khop"] == 0.5
    assert rep["TB_ti_le_giao"] == pytest.approx((1.0 + 0.5) / 2)
    assert rep["cau_thieu_trong_ref"] == 1


def test_cmplogit_ok_and_bad():
    ref = {"1": {"10": 0.5, "11": -1.0}}
    got = {"1": [[10, 1, 0.505], [11, 2, -1.004]]}
    rep = cmplogit(got, ref, tol=1e-2)
    assert rep["ok"] is True
    assert rep["so_cap_khop"] == 2
    assert rep["max_sai"] == pytest.approx(0.005)
    rep = cmplogit({"1": [[10, 1, 0.6], [11, 2, -1.0]]}, ref, tol=1e-2)
    assert rep["ok"] is False
    assert rep["cap_sai_nhat"][1] == 10


def test_cmplogit_absent_and_require_full():
    ref = {"1": {"10": 0.5, "11": -1.0}}
    got = {"1": [[10, 1, 0.5]]}
    assert cmplogit(got, ref)["ok"] is True
    assert cmplogit(got, ref)["so_cap_thieu_trong_got"] == 1
    assert cmplogit(got, ref, require_full=True)["ok"] is False
    assert cmplogit({"1": []}, ref)["ok"] is False


def _write_cache(p, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_cmd_pairs_end_to_end(tmp_path):
    from pool3 import main
    d = tmp_path
    _write_cache(d / "pool.json", {"1": [[1, 10, 0.5], [2, 11, 0.4]]})
    _write_cache(d / "pool2.json", {"1": [[9, 20, 0.3], [1, 10, 0.5], [8, 21, None]]})
    _write_cache(d / "new.json", {"1": [9, 8, 7]})
    _write_cache(d / "src.json", {"1": [[8, 21, 0.2], [7, 22, 0.1]]})
    argv = ["pool3.py", "pairs", "--pool", str(d / "pool.json"), "--pool2", str(d / "pool2.json"),
            "--new", str(d / "new.json"), "--idx-from", str(d / "src.json"),
            "--out-cands", str(d / "cands.json"), "--out-new-all", str(d / "all.json"),
            "--out-sanity", str(d / "sanity.json"), "--out-sanity-ref", str(d / "ref.json"),
            "--sanity-q", "5", "--sanity-per", "1"]
    monkey_argv = argv
    old = sys.argv
    sys.argv = monkey_argv
    try:
        with pytest.raises(SystemExit) as e:
            main()
    finally:
        sys.argv = old
    assert e.value.code == 0
    cands = json.loads((d / "cands.json").read_text(encoding="utf-8"))
    assert cands == {"1": [[8, 21], [7, 22]]}
    allp = json.loads((d / "all.json").read_text(encoding="utf-8"))
    assert allp == {"1": [[9, 20], [8, 21], [7, 22]]}
    sanity = json.loads((d / "sanity.json").read_text(encoding="utf-8"))
    ref = json.loads((d / "ref.json").read_text(encoding="utf-8"))
    assert sanity == {"1": [[1, 10], [9, 20]]}
    assert ref == {"1": {"1": 0.5, "9": 0.3}}


def test_cmd_pairs_reports_thieu_idx(tmp_path):
    from pool3 import main
    d = tmp_path
    _write_cache(d / "pool.json", {"1": [[1, 10, 0.5]]})
    _write_cache(d / "pool2.json", {"1": [[1, 10, 0.5]]})
    _write_cache(d / "new.json", {"1": [55]})
    old = sys.argv
    sys.argv = ["pool3.py", "pairs", "--pool", str(d / "pool.json"), "--pool2", str(d / "pool2.json"),
                "--new", str(d / "new.json"),
                "--out-cands", str(d / "c.json"), "--out-new-all", str(d / "a.json"),
                "--out-sanity", str(d / "s.json"), "--out-sanity-ref", str(d / "r.json")]
    try:
        with pytest.raises(SystemExit) as e:
            main()
    finally:
        sys.argv = old
    assert e.value.code == 3


def test_cmd_doclist_end_to_end(tmp_path):
    from pool3 import main
    d = tmp_path
    _write_cache(d / "pool3.json", {"1": [[1, 1, -1.0], [2, 2, -3.0]]})
    _write_cache(d / "rank.json", {"1": [[2, 2, 0.9], [1, 1, 0.8]]})
    _write_cache(d / "new.json", {"1": [2]})
    (d / "meta.json").write_text(
        json.dumps({"1": [{"doc_id": 2, "n_src": 1, "sources": [["hyde", 3]], "rrf": 0.1}]}),
        encoding="utf-8")
    old = sys.argv
    sys.argv = ["pool3.py", "doclist", "--pool3", str(d / "pool3.json"), "--rank", str(d / "rank.json"),
                "--new", str(d / "new.json"), "--meta", str(d / "meta.json"),
                "--thr", "-2.0", "--lo", "2", "--hi", "10", "--out", str(d / "dl.json")]
    try:
        with pytest.raises(SystemExit) as e:
            main()
    finally:
        sys.argv = old
    assert e.value.code == 0
    out = json.loads((d / "dl.json").read_text(encoding="utf-8"))
    assert out == {"1": [2, 1]}


def test_cmd_pool3_end_to_end(tmp_path):
    from pool3 import main
    d = tmp_path
    _write_cache(d / "pool.json", {"1": [[1, 10, 0.5]]})
    _write_cache(d / "pool2.json", {"1": [[2, 20, 0.4]]})
    _write_cache(d / "new.json", {"1": [2, 3]})
    _write_cache(d / "rerank.json", {"1": [[3, 30, -1.0]]})
    _write_cache(d / "src.json", {"1": [[3, 30, 0.0]]})
    old = sys.argv
    sys.argv = ["pool3.py", "pool3", "--pool", str(d / "pool.json"), "--pool2", str(d / "pool2.json"),
                "--new", str(d / "new.json"), "--rerank", str(d / "rerank.json"),
                "--idx-from", str(d / "src.json"), "--out", str(d / "out.json")]
    try:
        with pytest.raises(SystemExit) as e:
            main()
    finally:
        sys.argv = old
    assert e.value.code == 0
    out = json.loads((d / "out.json").read_text(encoding="utf-8"))
    assert out["1"] == [[1, 10, 0.5], [2, 20, 0.4], [3, 30, -1.0]]
