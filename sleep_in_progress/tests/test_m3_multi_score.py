import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from m3_multi_score import (  # noqa: E402
    colbert_score,
    dense_score,
    ensemble_score,
    rank_cache,
    sparse_score,
    sparse_token_weights,
    spearman,
)


def test_colbert_score_maxsim_trung_binh_theo_token_q():
    q = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    d = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    # token q1: max(1, 0, 0.5) = 1.0; token q2: max(0, 0, 0.5) = 0.5 -> (1.0 + 0.5) / 2
    assert colbert_score(q, d) == pytest.approx(0.75)


def test_colbert_score_max_theo_tung_token_doc():
    q = np.array([[1.0, 0.0], [1.0, 0.0]])
    d = np.array([[0.6, 0.8], [0.9, 0.4358898943540673]])
    # cả 2 token q cùng chọn max = 0.9
    assert colbert_score(q, d) == pytest.approx(0.9)


def test_colbert_score_khong_chia_dung_so_token_q():
    q = np.zeros((4, 3))
    q[:, 0] = 1.0
    d = np.array([[1.0, 0.0, 0.0]])
    assert colbert_score(q, d) == pytest.approx(1.0)


def test_colbert_score_vector_gia():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 8))
    d = rng.normal(size=(7, 8))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sims = q @ d.T
    assert colbert_score(q, d) == pytest.approx(float(sims.max(axis=1).mean()))


def test_colbert_score_bao_loi_dau_vao():
    with pytest.raises(ValueError):
        colbert_score(np.zeros((0, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        colbert_score(np.zeros((2, 3)), np.zeros((2, 4)))


def test_sparse_score_dot_tren_token_chung():
    q_w = {5: 1.0, 7: 2.0, 9: 3.0}
    d_w = {7: 0.5, 8: 4.0}
    assert sparse_score(q_w, d_w) == pytest.approx(1.0)


def test_sparse_score_doi_xung_va_rong():
    assert sparse_score({1: 2.0}, {1: 3.0}) == pytest.approx(6.0)
    assert sparse_score({1: 2.0}, {1: 3.0}) == sparse_score({1: 3.0}, {1: 2.0})
    assert sparse_score({}, {1: 3.0}) == 0.0
    assert sparse_score({2: 1.0}, {3: 1.0}) == 0.0


def test_sparse_token_weights_max_theo_vi_tri_bo_special():
    token_ids = [0, 4, 4, 7, 7, 7, 2]
    weights = [0.9, 0.5, 0.2, 0.3, 0.8, 0.0, 0.4]
    out = sparse_token_weights(token_ids, weights, unused_tokens=(0, 1, 2, 3))
    assert out == {4: 0.5, 7: 0.8}


def test_sparse_token_weights_bo_w_am_va_le():
    out = sparse_token_weights([10, 10, 11], [0.0, -1.5, 2.5], unused_tokens=())
    assert out == {11: 2.5}


def test_dense_score_cosine_vector_chuan_hoa():
    v = np.array([3.0, 4.0]) / 5.0
    assert dense_score(v, v) == pytest.approx(1.0)
    assert dense_score(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_ensemble_score_trong_so_flag_embedding():
    assert ensemble_score(1.0, 2.0, 0.5) == pytest.approx(1.0 + 0.3 * 2.0 + 0.5)


def test_spearman_hoan_hao_va_nguoc():
    assert spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_xu_ly_tie_va_ngan():
    assert spearman([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0) or math.isnan(
        spearman([1.0, 1.0], [2.0, 2.0])
    )
    assert math.isnan(spearman([1.0], [2.0]))


def test_rank_cache_giam_dan_dung_dinh_dang():
    rows = [
        [10, 0, 0.1, 5.0, 0.2],
        [11, 1, 0.9, 1.0, 0.8],
        [12, 2, 0.5, 2.0, 0.4],
    ]
    out = rank_cache(rows, lambda r: r[4])
    assert out == [[11, 1, 0.8], [12, 2, 0.4], [10, 0, 0.2]]
    out_all = rank_cache(rows, lambda r: ensemble_score(r[2], r[3], r[4]))
    assert out_all[0][0] == 11  # 0.9 + 0.3 + 0.8 = 2.0 la cao nhat
    assert all(len(r) == 3 for r in out_all)


def test_rank_cache_diem_bang_giu_thu_tu_goc():
    rows = [
        [1, 0, 0.0, 1.0, 0.5],
        [2, 1, 0.0, 1.0, 0.5],
    ]
    assert [r[0] for r in rank_cache(rows, lambda r: r[4])] == [1, 2]
