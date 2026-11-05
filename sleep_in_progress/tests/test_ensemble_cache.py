import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ensemble_cache import (  # noqa: E402
    ensemble_cache,
    ens_query,
    fill_min,
    rank_map,
    top30_overlap,
    z_map,
)


def test_fill_min_none_lay_min_cau():
    assert fill_min({"a": 1.0, "b": None, "c": -3.0}, ["a", "b", "c"]) == {"a": 1.0, "b": -3.0, "c": -3.0}


def test_fill_min_tat_ca_none_thi_0():
    assert fill_min({"a": None, "b": None}, ["a", "b"]) == {"a": 0.0, "b": 0.0}


def test_rank_map_giam_dan_1_based():
    assert rank_map(["a", "b", "c"], {"a": 1.0, "b": 3.0, "c": 2.0}) == {"b": 1, "c": 2, "a": 3}


def test_z_map_coban():
    z = z_map(["a", "b"], {"a": 0.0, "b": 1.0})
    assert z["a"] == pytest.approx(-1.0)
    assert z["b"] == pytest.approx(1.0)


def test_z_map_sigma_0_thi_z_0():
    assert z_map(["a", "b", "c"], {"a": 5.0, "b": 5.0, "c": 5.0}) == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_rrf_coban():
    rows_a = [[1, 10, 3.0], [2, 20, 2.0], [3, 30, 1.0]]
    rows_b = [[3, 30, 10.0], [1, 10, 5.0], [2, 20, 1.0]]
    rows, model_a, model_b = ens_query(rows_a, rows_b, "rrf", k=60)
    assert [r[0] for r in rows] == [1, 3, 2]
    assert rows[0][2] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
    assert rows[0][1] == 10
    assert model_a == [1, 2, 3]
    assert model_b == [3, 1, 2]


def test_rrf_trong_so():
    rows_a = [[1, 10, 3.0], [2, 20, 2.0]]
    rows_b = [[2, 20, 9.0], [1, 10, 1.0]]
    rows, _, _ = ens_query(rows_a, rows_b, "rrf", k=60, wa=1.0, wb=0.0)
    assert [r[0] for r in rows] == [1, 2]


def test_z_mode_coban():
    rows_a = [[1, 10, 0.0], [2, 20, 1.0]]
    rows_b = [[1, 10, 10.0], [2, 20, 20.0]]
    rows, _, _ = ens_query(rows_a, rows_b, "z")
    assert [r[0] for r in rows] == [2, 1]
    assert rows[0][2] == pytest.approx(2.0, abs=1e-6)
    assert rows[1][2] == pytest.approx(-2.0, abs=1e-6)


def test_rrf_none_dien_min_anh_huong_hang():
    rows_a = [[1, 1, 1.0], [2, 2, 2.0], [3, 3, 3.0]]
    rows_b = [[1, 1, None], [2, 2, 0.0], [3, 3, 5.0]]
    rows, _, _ = ens_query(rows_a, rows_b, "rrf", k=60, wa=0.0, wb=1.0)
    assert [r[0] for r in rows] == [3, 1, 2]


def test_chi_giao_hai_ben():
    rows_a = [[1, 1, 2.0], [2, 2, 1.0], [9, 9, 0.0]]
    rows_b = [[1, 1, 1.0], [2, 2, 2.0], [8, 8, 3.0]]
    rows, _, _ = ens_query(rows_a, rows_b, "rrf")
    assert {r[0] for r in rows} == {1, 2}


def test_hoa_giu_thu_tu_a():
    rows_a = [[1, 1, 5.0], [2, 2, 5.0], [3, 3, 5.0]]
    rows_b = [[3, 3, 5.0], [2, 2, 5.0], [1, 1, 5.0]]
    rows, _, _ = ens_query(rows_a, rows_b, "z")
    assert [r[0] for r in rows] == [1, 2, 3]


def test_idx_lay_tu_a():
    rows, _, _ = ens_query([[5, 99, 1.0]], [[5, 77, 1.0]], "rrf")
    assert rows[0][1] == 99


def test_dinh_dang_dau_ra():
    rows, _, _ = ens_query([[7, 70, 1.0], [8, 80, 2.0]], [[7, 70, 1.0], [8, 80, 2.0]], "z")
    assert all(len(r) == 3 for r in rows)
    assert all(isinstance(r[1], int) for r in rows)
    assert rows[0][2] >= rows[1][2]


def test_top30_overlap():
    out_rows = [[i, i, 0.0] for i in range(40)]
    model = list(range(20)) + list(range(100, 120))
    assert top30_overlap(out_rows, model) == pytest.approx(20 / 30)


def test_ensemble_cache_a_top_va_cau_thieu_b():
    cache_a = {"1": [[1, 1, 3.0], [2, 2, 2.0], [3, 3, 1.0]], "2": [[9, 9, 1.0]]}
    cache_b = {"1": [[1, 1, 1.0], [2, 2, 2.0], [3, 3, 3.0]]}
    out, stats = ensemble_cache(cache_a, cache_b, "rrf", a_top=2)
    assert list(out) == ["1"]
    assert [r[0] for r in out["1"]] == [1, 2]
    assert set(stats) == {"trùng top-30 a", "trùng top-30 b"}
    assert 0 <= stats["trùng top-30 a"] <= 1
    assert 0 <= stats["trùng top-30 b"] <= 1
