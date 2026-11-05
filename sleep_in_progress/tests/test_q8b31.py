import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from q8b31 import (bench_sample, check_merged, corr, merge_q8b, new_rank_dist,  # noqa: E402
                   select_pairs, split_shards)


def test_select_pairs_pool_bo_cu_top_trung():
    fhyde = {"1": [[5, 0, 1], [6, 0, 1], [7, 0, 1], [6, 0, 1], [8, 0, 1], [9, 0, 1]]}
    pool = {"1": [[5, 0, 0], [6, 0, 0], [8, 0, 0], [9, 0, 0]]}
    q8b = {"1": [[5, 0, 2.0]]}
    assert select_pairs(fhyde, pool, q8b, top=5) == {"1": [6, 8]}  # 7 ngoai pool, 9 ngoai top-5


def test_select_pairs_cau_khong_co_q8b_va_khoa_int():
    out = select_pairs({1: [[5, 0, 1]]}, {"1": [[5, 0, 0]]}, {})
    assert out == {"1": [5]}


def test_bench_sample_chi_cap_co_diem():
    fhyde = {"1": [[5, 0, 1], [6, 0, 1]], "2": [[7, 0, 1]]}
    q8b = {"1": [[5, 0, 2.0], [6, 0, None]], "2": [[7, 0, -1.0]]}
    b = bench_sample(fhyde, q8b, 10)
    assert sorted(b) == [("1", 5, 2.0), ("2", 7, -1.0)]
    assert len(bench_sample(fhyde, q8b, 1)) == 1


def test_split_shards_can_bang_bo_rong():
    pairs = {"1": [1] * 5, "2": [1] * 3, "3": [1] * 2, "4": []}
    s = split_shards(pairs, 2)
    assert sorted(sum(map(len, x.values())) for x in s) == [5, 5]
    assert "4" not in s[0] and "4" not in s[1]
    assert list(s[0]) == sorted(s[0], key=int)


def test_merge_q8b_hop_sap_giu_cu_bo_none():
    old = {"1": [[5, 0, 3.0], [6, 0, 1.0]], "2": [[9, 0, 0.5]]}
    new = {"1": [[7, 1, 2.0], [5, 0, 9.0], [8, 2, None]]}
    out, st = merge_q8b(old, new)
    assert out["1"] == [[5, 0, 3.0], [7, 1, 2.0], [6, 0, 1.0]]
    assert out["2"] == [[9, 0, 0.5]]
    assert st == {"trung_bo": 1, "none_bo": 1}


def test_check_merged_va_rank_dist():
    old = {str(q): [[1, 0, 0.0]] for q in range(1, 4)}
    new = {"1": [[2, 0, 1.0]], "2": [[3, 0, -1.0]]}
    out, _ = merge_q8b(old, new)
    pool = {"1": [[2, 0, 0]], "2": [[3, 0, 0]]}
    chk = check_merged(out, new, pool, n_q=3)
    assert chk["ok"]
    assert not check_merged(out, new, {"1": [[2, 0, 0]]}, n_q=3)["ok"]
    d = new_rank_dist(out, new, [2])
    assert d["new_ids"]["1-10"] == 1 and d["cu_thieu"]["1-10"] == 1


def test_corr_trung():
    r = corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.5])
    assert r["spearman"] == 1.0 and r["lech_max"] == 0.5
