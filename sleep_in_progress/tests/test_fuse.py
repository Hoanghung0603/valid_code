import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fuse import merge_cosines, pool_rrf3, rerank_order, rrf_fuse, rrf_fuse_multi, sort_by_score  # noqa: E402


def test_rrf_promotes_doc_ranked_high_by_dense():
    bm = [[1, 10, 9.0], [2, 20, 8.0], [3, 30, 7.0], [4, 40, 6.0]]
    dense = [[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3], [4, 40, 0.9]]
    out = rrf_fuse(bm, dense)
    assert {d for d, _, _ in out} == {1, 2, 3, 4}
    assert out[0][0] in (1, 4) and out.index(next(h for h in out if h[0] == 4)) < out.index(next(h for h in out if h[0] == 3))


def test_rrf_tie_keeps_bm25_order_and_handles_missing_dense():
    bm = [[1, 10, 2.0], [2, 20, 1.0]]
    out = rrf_fuse(bm, [[1, 10, None], [2, 20, None]])
    assert [d for d, _, _ in out] == [1, 2]
    assert out[0][1] == 10


def test_rrf_only_bm25_candidates_plus_dense_extra():
    out = rrf_fuse([[1, 10, 1.0]], [[1, 10, 0.5], [5, 50, 0.9]])
    assert [d for d, _, _ in out][0] == 1 and 5 in [d for d, _, _ in out]


def test_rrf_fuse_multi_single_ranking_keeps_order():
    hits = [[1, 10, 9.0], [2, 20, 8.0], [3, 30, 7.0]]
    out = rrf_fuse_multi([(hits, 1.0)])
    assert [d for d, _, _ in out] == [1, 2, 3]


def test_rrf_three_rankings_rerank_breaks_tie():
    bm = [[1, 10, 3.0], [2, 20, 2.0], [3, 30, 1.0]]
    dn = [[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]]
    rr = [[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]]
    two = rrf_fuse(bm, dn)
    out = rrf_fuse(bm, dn, k=60, rerank_hits=rr)
    assert [d for d, _, _ in two] == [1, 3, 2]
    assert [d for d, _, _ in out] == [3, 2, 1]


def test_rrf_three_way_tie_keeps_bm25_order():
    hits = [[1, 10, 1.0], [2, 20, 1.0], [3, 30, 1.0]]
    out = rrf_fuse(hits, hits, k=60, rerank_hits=hits)
    assert [d for d, _, _ in out] == [1, 2, 3]


def test_rrf_without_rerank_equals_empty_rerank():
    bm = [[1, 10, 3.0], [2, 20, 2.0], [3, 30, 1.0]]
    dn = [[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.3]]
    assert rrf_fuse(bm, dn) == rrf_fuse(bm, dn, rerank_hits=[])


def test_rrf_w_rerank_scales_scores():
    bm = [[1, 10, 3.0], [2, 20, 2.0]]
    rr = [[1, 10, 0.1], [2, 20, 0.2]]
    a = rrf_fuse(bm, [], rerank_hits=rr, w_rerank=1.0)
    b = rrf_fuse(bm, [], rerank_hits=rr, w_rerank=2.0)
    assert [d for d, _, _ in a] != [d for d, _, _ in b]


def test_rerank_sorts_by_logit_desc_ties_keep_bm25_order():
    bm = [[1, 10, 9.0], [2, 20, 8.0], [3, 30, 7.0]]
    rr = [[1, 10, 0.5], [2, 20, 1.5], [3, 30, 1.5]]
    out = rerank_order(bm, rr, n=3)
    assert [d for d, _, _ in out] == [2, 3, 1]
    assert [s for _, _, s in out] == [1.5, 1.5, 0.5]
    assert out[0][1] == 20


def test_rerank_takes_top_n_only_and_missing_logit_last():
    bm = [[1, 10, 9.0], [2, 20, 8.0], [3, 30, 7.0], [4, 40, 6.0]]
    rr = [[3, 30, 0.1], [1, 10, None]]
    out = rerank_order(bm, rr, n=3)
    assert [d for d, _, _ in out] == [3, 1, 2]
    assert out[-1][2] is None and out[0][2] == 0.1


def test_rerank_result_shape_is_cache_format():
    out = rerank_order([[1, 10, 9.0], [2, 20, 8.0]], [[2, 20, 2.4395], [1, 10, -0.0307]], n=2)
    assert out == [[2, 20, 2.4395], [1, 10, -0.0307]]


def test_sort_by_score_desc_ties_keep_original_order():
    hits = [[1, 10, 0.5], [2, 20, 1.5], [3, 30, 1.5], [4, 40, -2.0]]
    out = sort_by_score(hits)
    assert [d for d, _, _ in out] == [2, 3, 1, 4]
    assert out[0][1] == 20


def test_sort_by_score_none_last():
    out = sort_by_score([[1, 10, None], [2, 20, 0.1], [3, 30, None]])
    assert [d for d, _, _ in out] == [2, 1, 3]


def test_merge_cosines_prefers_primary_on_overlap():
    a = [[1, 10, 0.6], [2, 20, 0.9]]
    b = [[2, 20, 0.85], [3, 30, 0.95]]
    out = merge_cosines(a, b)
    by = {d: (i, s) for d, i, s in out}
    assert set(by) == {1, 2, 3}
    assert by[2][1] == 0.9
    assert [d for d, _, _ in out] == [3, 2, 1]


def test_merge_cosines_skips_none_and_orders_by_cosine():
    out = merge_cosines([[1, 10, None], [2, 20, 0.3]], [[3, 30, 0.7]])
    assert [d for d, _, _ in out] == [3, 2]
    assert 1 not in [d for d, _, _ in out]


def test_pool_rrf3_uses_three_rankings_and_keeps_all_pool_docs():
    bm = [[1, 10, 9.0], [2, 20, 8.0]]
    dp = [[1, 10, 0.2], [2, 20, 0.3]]
    ds = [[3, 30, 0.9]]
    pool = [[1, 10, 0.1], [2, 20, 0.5], [3, 30, 0.2]]
    out = pool_rrf3(bm, dp, ds, pool, k=60)
    assert {d for d, _, _ in out} == {1, 2, 3}
    assert out[0][0] == 2


def test_pool_rrf3_doc_absent_from_bm25_gets_no_bm25_contribution():
    bm = [[1, 10, 9.0]]
    dp = [[1, 10, 0.2], [2, 20, 0.3]]
    ds = []
    pool = [[1, 10, 0.5], [2, 20, 0.5]]
    out = pool_rrf3(bm, dp, ds, pool, k=60)
    scores = {d: s for d, _, s in out}
    assert scores[1] > scores[2]
