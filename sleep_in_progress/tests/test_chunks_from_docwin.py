import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunks_from_docwin import (  # noqa: E402
    build_q_rows,
    chunk_score,
    doc_rank_dist,
    logit_at_ranks,
    overlap_chars,
    overlap_topk,
    score_doc_chunks,
    select_q_rows,
    stats,
    verify_verbatim,
)

TEXT = "\n".join("Điều trị bằng truyền dịch và theo dõi tiểu cầu ngày %d. " % i * 5 for i in range(30))


def test_overlap_chars():
    assert overlap_chars(0, 10, 5, 15) == 5
    assert overlap_chars(0, 10, 10, 15) == 0
    assert overlap_chars(0, 10, 2, 8) == 6
    assert overlap_chars(5, 6, 0, 100) == 1


def test_chunk_score_majority_inside_qualifies_and_max_pools():
    # chunk [100, 200): cửa sổ A [100, 180) 100% trong chunk; B [150, 250) 50% (>= 50%: đủ điều kiện)
    wins = [[100, 180, 1.0], [150, 250, 5.0], [0, 60, 9.0]]
    assert chunk_score(wins, 100, 200) == 5.0, "max logit các cửa sổ >= 50% ký tự trong chunk"


def test_chunk_score_below_half_excluded():
    # cửa sổ [0, 155): 55/155 ≈ 35% trong chunk [100, 200) — không đủ, dù logit cao
    wins = [[0, 155, 9.0], [100, 140, 1.5]]
    # [100, 140): 40/40 = 100% -> qualify; [0, 155) 55*2=110 < 155 -> loại
    assert chunk_score(wins, 100, 200) == 1.5


def test_chunk_score_exact_half_qualifies():
    wins = [[0, 100, 2.0]]  # overlap 50/100 = 50%
    assert chunk_score(wins, 50, 150) == 2.0


def test_chunk_score_fallback_max_overlap_then_logit():
    # chunk [100, 200): không cửa sổ nào >= 50% (mọi cửa sổ bao trùm/rìa đều < 50%)
    wins = [[0, 300, 1.0], [0, 250, 3.0], [50, 280, 2.0]]
    # overlap đều 100 → đồng "giao nhiều nhất" → logit cao hơn 3.0
    assert chunk_score(wins, 100, 200) == 3.0
    wins2 = [[0, 300, 1.0], [150, 260, 9.0]]
    # overlap 100 vs 50 (cùng < 50% độ dài cửa sổ) → [0, 300) thắng dù logit thấp hơn
    assert chunk_score(wins2, 100, 200) == 1.0


def test_chunk_score_none_and_zero_overlap():
    assert chunk_score([], 0, 10) is None
    # tất cả overlap = 0 → fallback vẫn trả logit của cửa sổ đầu (giao nhiều nhất = 0)
    wins = [[100, 200, 4.0], [300, 400, 7.0]]
    assert chunk_score(wins, 0, 50) == 7.0


def test_score_doc_chunks_verbatim_and_indices():
    from chunking import chunk_document
    wins = [[0, len(TEXT), 1.0]]
    out = score_doc_chunks(chunk_document(TEXT, "vi", max_units=30), wins)
    assert out
    for ci, cs, ce, sc, ct in out:
        assert ct in TEXT, "chunk_text phải là nguyên văn"
        assert sc == 1.0
        assert ci >= 0 and 0 <= cs < ce <= len(TEXT)


def test_score_doc_chunks_empty_wins_skipped():
    from chunking import chunk_document
    assert score_doc_chunks(chunk_document(TEXT, "vi", max_units=30), []) == []


def test_build_q_rows_format_and_doc_rank():
    docs = {10: TEXT, 20: "x y z", 30: "không có wins"}
    wins_q = {"10": [[0, len(TEXT), 2.0]], "20": [[0, 5, 3.0]]}
    cand = [[10, 1], [99, 2], [20, 3], [30, 4]]
    rows = build_q_rows(cand, docs, wins_q, max_units=30, top_docs=4)
    assert rows, "có dòng"
    for r in rows:
        assert len(r) == 5
        assert r[0] in (10, 20)
        assert r[4] in docs[r[0]], "nguyên văn"
        assert r[3] in (2.0, 3.0)
    ranks10 = [r[1] for r in rows if r[0] == 10]
    ranks20 = [r[1] for r in rows if r[0] == 20]
    assert set(ranks10) == {1} and set(ranks20) == {3}, "doc_rank giữ vị trí trong cand"


def test_build_q_rows_missing_docs_and_cache():
    docs = {1: "một hai ba bốn năm sáu bảy tám chín mười"}
    wins_q = {"1": [[0, 100, 1.0]]}
    cand = [[5, 0], [1, 1]]
    cache = {}
    rows1 = build_q_rows(cand, docs, wins_q, 4, 2, cache)
    assert rows1 and {r[0] for r in rows1} == {1}, "doc 5 thiếu text bị bỏ qua"
    assert [r[1] for r in rows1] == [2] * len(rows1), "doc_rank = 2"
    assert len(cache) == 1
    chunks_before = cache[(1, 4)]
    rows2 = build_q_rows(cand, docs, wins_q, 4, 2, cache)
    assert rows2 == rows1
    assert cache[(1, 4)] is chunks_before, "cache: không chunk lại"


def test_select_q_rows_sorts_desc_and_cuts():
    rows = [[1, 1, 0, 0.5, "a"], [2, 2, 0, 2.25, "b"], [3, 3, 0, 1.123456, "c"], [4, 4, 0, -3.0, "d"]]
    out = select_q_rows(rows, top_chunks=3)
    assert out == [[2, 2, 0, 2.25, "b"], [3, 3, 0, 1.1235, "c"], [1, 1, 0, 0.5, "a"]]
    assert select_q_rows([], 5) == []


def test_select_q_rows_tie_breaks():
    rows = [[9, 2, 1, 1.0, "x"], [8, 1, 3, 1.0, "y"], [7, 2, 0, 1.0, "z"]]
    assert select_q_rows(rows, 3) == [[8, 1, 3, 1.0, "y"], [7, 2, 0, 1.0, "z"], [9, 2, 1, 1.0, "x"]]


def test_overlap_topk():
    a = [[1, 1, 0, 3.0, "a"], [2, 2, 1, 2.0, "b"], [3, 3, 2, 1.0, "c"]]
    b = [[2, 9, 1, 5.0, "b"], [4, 8, 0, 4.0, "z"], [1, 7, 0, 3.0, "a"]]
    assert overlap_topk(a, b, 2) == 1
    assert overlap_topk(a, b, 3) == 2
    assert overlap_topk(a, [], 3) == 0


def test_verify_verbatim_detects_bad():
    docs = {10: "nguyên văn ở đây"}
    out = {"1": [[10, 1, 0, 1.0, "nguyên văn ở đây"], [10, 1, 1, 1.0, "GIẢ MẠO"]]}
    assert verify_verbatim(out, docs) == 1
    assert verify_verbatim({"1": [[10, 1, 0, 1.0, "nguyên văn ở đây"]]}, docs) == 0


def test_logit_at_ranks_and_doc_rank_dist():
    out = {
        "1": [[1, 1, 0, 10.0, "a"], [2, 30, 0, 5.0, "b"], [3, 60, 0, 1.0, "c"]],
        "2": [[4, 2, 0, 20.0, "d"], [5, 80, 0, 7.0, "e"]],
    }
    lr = logit_at_ranks(out, [1, 2, 3])
    assert lr[1] == 15.0 and lr[2] == 6.0 and lr[3] == 1.0
    dist = doc_rank_dist(out, top=40)
    assert abs(dist["1-25"] - 2 / 5) < 1e-9
    assert abs(dist["26-50"] - 1 / 5) < 1e-9
    assert abs(dist["51-75"] - 1 / 5) < 1e-9
    assert abs(dist["76-100"] - 1 / 5) < 1e-9


def test_stats_summary():
    out = {"1": [[1, 1, 0, 2.0, "một hai ba"], [2, 2, 0, 1.0, "bốn năm"]], "2": [[3, 3, 0, 5.0, "sáu"]]}
    st = stats(out, [1, 2])
    assert st["câu"] == 2
    assert st["TB chunk/câu"] == 1.5
    assert abs(st["TB từ/chunk"] - (3 + 2 + 1) / 3) < 1e-9
    assert st["logit TB hạng"][1] == 3.5
    assert "1-25" in st["doc_rank top-40"]
