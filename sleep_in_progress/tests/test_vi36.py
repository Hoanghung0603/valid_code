import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi36  # noqa: E402


def test_constants():
    assert vi36.MAX_ZIP == 99_000_000
    assert vi36.DELTA0 == 56 and vi36.SANITY_DELTA == 56
    assert vi36.EXP_ADD == 1793 and vi36.N_PKG == 1793 and vi36.N_BAD == 16079
    assert vi36.W8F == 2.0
    assert vi36.OUT == vi36.ROOT / "data/sub/vi36"
    assert vi36.IX == vi36.ROOT / "data/index/vi36"
    assert vi36.Q8_V33.as_posix().endswith("diag33/q8b_v33.json")
    assert vi36.CHUNK_V34.parent.name == "vi34"
    assert vi36.PKG_FILES == ["vi-topup-00000.jsonl.zst"]


def test_select_new_ids_disjoint():
    idx = {1, 2, 3}
    pkg = {4, 5, 6}
    sel = vi36.select_new_ids(pkg, idx)
    assert sel["add"] == {4, 5, 6}
    assert sel["dup"] == set()


def test_select_new_ids_bat_trung():
    sel = vi36.select_new_ids({1, 4}, {1, 2})
    assert sel["add"] == {4}
    assert sel["dup"] == {1}


def test_verify_index_ids():
    r = vi36.verify_index_ids([1, 2, 3, 4, 5], [1, 2, 3], [4, 5])
    assert r["ok"] and r["mong_doi"] == 5
    assert not vi36.verify_index_ids([1, 1, 2], [1], [2])["ok"]      # trung
    assert not vi36.verify_index_ids([1, 2], [1, 2, 3], [])["ok"]    # thieu 3


def test_filter_rows():
    rows = [{"id": 1}, {"id": 3}, {"id": 4, "x": 1}, {"title": "khong id"}]
    assert [r.get("id") for r in vi36.filter_rows(rows, {3, 4})] == [1, None]


def test_build_excl():
    assert vi36.build_excl({3, 4, 9}, [3, 5]) == {4, 9}


def test_merge_topk_hits():
    base = [[1, 0, 0.9], [2, 1, 0.5]]
    new = [[3, 2, 0.7], [1, 3, 0.4]]
    out, st = vi36.merge_topk_hits(base, new, 3)
    assert [h[0] for h in out] == [1, 3, 2]
    assert st["so_moi"] == 1 and st["so_cu"] == 2


def test_select_8b_targets_doclist_plus_margin():
    doclist = {"1": [10, 20]}
    fhyde = {"1": [[10, 0, 0.9], [20, 1, 0.8], [30, 2, 0.7], [40, 3, 0.6], [50, 4, 0.5]]}
    q8b = {"1": [[10, 0, 0.5]]}
    out = vi36.select_8b_targets(doclist, fhyde, q8b, margin=2)
    # 20 (doclist thieu 8B) + 30,40 (margin top-4 thieu 8B); 10 da co, 50 ngoai top
    assert out == {"1": [20, 30, 40]}


def test_select_8b_targets_bo_da_co_diem():
    doclist = {"1": [10, 20]}
    fhyde = {"1": [[10, 0, 0.9], [20, 1, 0.8], [30, 2, 0.7]]}
    q8b = {"1": [[10, 0, 0.5], [20, 1, 0.4], [30, 2, 0.3]]}
    assert vi36.select_8b_targets(doclist, fhyde, q8b, margin=1) == {}


def test_check_wincheck():
    vi36.check_wincheck({"so_cap": 300, "so_mau": 300, "max_sai": 0.01})
    with pytest.raises(SystemExit):
        vi36.check_wincheck({"so_cap": 299, "so_mau": 300, "max_sai": 0.01})


def test_input_spec_sanity_vs_build():
    sp = vi36.input_spec(vi36.Q8_V33, vi36.CHUNK_V34)
    assert {n for n, _, _ in sp} == set(vi36.FILE_NAMES.values())
    ss = vi36.sanity_spec()
    assert {n for n, _, _ in ss} == set(vi36.FILE_NAMES.values())
    # sanity dung nen vi34 (q8b_v33 + chunk_v34)
    d = {n: s for n, s, _ in ss}
    assert d[vi36.FILE_NAMES["q8b"]] == vi36.Q8_V33
    assert d[vi36.FILE_NAMES["cands"]] == vi36.CHUNK_V34


def test_q8b_select_no_self_lock():
    # q8b_select chay q8b31.cmd_select tu giu heavy_job.lock -> main phai bo qua khoa
    assert "q8b_select" in vi36.NO_SELF_LOCK
    assert vi36.NO_SELF_LOCK <= set(vi36.B660_STAGES)
    for st in vi36.B660_STAGES:
        if st not in vi36.NO_SELF_LOCK:
            assert st not in vi36.NO_SELF_LOCK
