import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunk_rerank import (  # noqa: E402
    build_chunk_pairs,
    build_window_pairs,
    chunk_doc_texts,
    max_pool,
    out_path,
    select_top,
    split_first_col,
    window_texts,
    write_out,
)

SENT = "Bệnh nhân sốt cao liên tục trong ba ngày thì đến khám. "
TEXT = SENT * 40 + "\n" + "Điều trị bằng truyền dịch và theo dõi tiểu cầu. " * 40


def test_chunk_doc_texts_are_verbatim_substrings():
    cts = chunk_doc_texts(TEXT, max_units=30)
    assert len(cts) >= 2
    pos = 0
    for ct in cts:
        assert ct
        assert ct in TEXT, "chunk phải là nguyên văn từ text doc"
        pos = TEXT.find(ct, pos)
        assert pos >= 0, "chunk giữ nguyên thứ tự trong doc"
        pos += len(ct)


def test_chunk_doc_texts_default_units():
    text = " ".join(f"t{i}" for i in range(620))
    cts = chunk_doc_texts(text)
    assert len(cts) == 2
    assert all(ct in text for ct in cts)
    assert chunk_doc_texts("ngắn") == ["ngắn"]


def test_build_chunk_pairs_metas_and_missing_docs():
    docs = {10: "x", 20: "y"}
    cache = {10: ["c10a", "c10b"], 20: ["c20a"]}
    cand = [[10, 100], [99, 101], [20, 102], [30, 103]]
    pairs, metas = build_chunk_pairs("q?", cand, docs, top_docs=4, chunk_cache=cache)
    assert pairs == [("q?", "c10a"), ("q?", "c10b"), ("q?", "c20a")]
    assert metas == [[10, 1, 0, "c10a"], [10, 1, 1, "c10b"], [20, 3, 0, "c20a"]]


def test_build_chunk_pairs_respects_top_docs():
    cache = {1: ["a"], 2: ["b"], 3: ["c"]}
    cand = [[1, 0], [2, 1], [3, 2]]
    pairs, metas = build_chunk_pairs("q", cand, {}, top_docs=2, chunk_cache=cache)
    assert pairs == [("q", "a"), ("q", "b")]
    assert [m[1] for m in metas] == [1, 2]


def test_build_chunk_pairs_fills_cache_and_keeps_verbatim():
    cache = {}
    pairs, metas = build_chunk_pairs("qq", [[7, 0]], {7: TEXT}, top_docs=1,
                                     max_units=30, chunk_cache=cache)
    assert cache[7] == chunk_doc_texts(TEXT, 30)
    assert [p[1] for p in pairs] == cache[7]
    assert [m[3] for m in metas] == cache[7]
    assert all(m[3] in TEXT for m in metas)
    assert [m[2] for m in metas] == list(range(len(metas)))
    # gọi lại không chunk lại: cache giữ nguyên khi docs đã đổi
    pairs2, _ = build_chunk_pairs("qq", [[7, 0]], {7: "khác"}, top_docs=1,
                                  max_units=30, chunk_cache=cache)
    assert pairs2 == pairs


def test_build_chunk_pairs_no_cache_uses_docs():
    pairs, metas = build_chunk_pairs("q", [[7, 0]], {7: TEXT}, top_docs=1, max_units=30)
    assert pairs == [("q", m[3]) for m in metas]
    assert all(m[3] in TEXT for m in metas)
    assert build_chunk_pairs("q", [[7, 0]], {}, top_docs=1) == ([], [])


def test_select_top_sorts_desc_rounds_and_cuts():
    metas = [[1, 1, 0, "a"], [2, 2, 0, "b"], [3, 3, 0, "c"], [4, 4, 0, "d"]]
    scores = [0.5, 2.25, 1.123456, -3.0]
    rows = select_top(metas, scores, top_chunks=3)
    assert rows == [[2, 2, 0, 2.25, "b"], [3, 3, 0, 1.1235, "c"], [1, 1, 0, 0.5, "a"]]
    assert select_top(metas, scores, top_chunks=40) == rows + [[4, 4, 0, -3.0, "d"]]
    assert select_top([], [], top_chunks=5) == []


def test_select_top_tie_breaks_by_rank_then_chunk():
    metas = [[9, 2, 1, "x"], [8, 1, 3, "y"], [7, 2, 0, "z"]]
    rows = select_top(metas, [1.0, 1.0, 1.0], top_chunks=3)
    assert rows == [[8, 1, 3, 1.0, "y"], [7, 2, 0, 1.0, "z"], [9, 2, 1, 1.0, "x"]]


def test_out_path_json_and_zst():
    assert out_path("a/b/out.json", 100) == Path("a/b/out.json")
    assert out_path("a/b/out.json", 10**9) == Path("a/b/out.json.zst")
    assert out_path("out", 10**9) == Path("out.zst")


def test_write_out_json_roundtrip(tmp_path):
    data = {"1": [[10, 1, 0, 0.5, "chunk văn bản"]]}
    p = write_out(str(tmp_path / "out.json"), data)
    assert p.name == "out.json"
    assert json.loads(p.read_text(encoding="utf-8")) == data


def test_write_out_zst_when_over_limit(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    data = {"1": [[10, 1, 0, 0.5, "x" * 500]]}
    p = write_out(str(tmp_path / "out.json"), data, limit=100)
    assert p.name == "out.json.zst"
    with open(p, "rb") as f:
        raw = zstandard.ZstdDecompressor().decompress(f.read(), max_output_size=1 << 20)
    assert json.loads(raw.decode("utf-8")) == data


def test_import_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    import chunk_rerank
    import importlib
    importlib.reload(chunk_rerank)
    assert chunk_rerank.PART_SIZE == 50
    assert callable(chunk_rerank.score_pairs)


def test_window_texts_covers_all_words():
    words = [f"t{i}" for i in range(100)]
    text = " ".join(words)
    for w, s in [(30, 20), (30, 30), (50, 10), (100, 50), (99, 98), (2, 1)]:
        wins = window_texts(text, w, s)
        covered: set = set()
        for win in wins:
            assert win in text, "cửa sổ phải là nguyên văn chunk"
            for i, wd in enumerate(win.split()):
                covered.add(int(wd[1:]))
        assert covered == set(range(100)), f"phủ hết từ (w={w}, s={s})"


def test_window_texts_last_window_reaches_end_and_overlap():
    text = " ".join(f"t{i}" for i in range(100))
    wins = window_texts(text, 30, 20)
    # các cửa sổ đầu đủ 30 từ, cách nhau 20 từ (chồng 10)
    assert wins[0].split() == [f"t{i}" for i in range(30)]
    assert wins[1].split() == [f"t{i}" for i in range(20, 50)]
    # cửa sổ cuối phủ tới hết chunk
    assert wins[-1].split()[-1] == "t99"
    assert all(len(w.split()) == 30 for w in wins[:-1])
    assert 1 <= len(wins[-1].split()) <= 30


def test_window_texts_short_chunk_single_window():
    for n in (1, 29, 30):
        text = " ".join(f"t{i}" for i in range(n))
        assert window_texts(text, 30, 20) == [text]
    assert window_texts("ngắn", 300, 250) == ["ngắn"]
    # W <= 0: hành vi cũ, 1 cửa sổ = nguyên chunk
    assert window_texts("a b c d e", 0, 0) == ["a b c d e"]


def test_window_texts_verbatim_with_whitespace():
    text = "Đầu  dòng\nthứ hai   và ba.\nCuối cùng!"
    wins = window_texts(text, 3, 2)
    assert all(win in text for win in wins)
    assert wins[0] == "Đầu  dòng\nthứ"
    assert wins[-1].endswith("Cuối cùng!")
    assert sum(len(w.split()) for w in wins) >= len(text.split())


def test_window_texts_bad_stride():
    import pytest
    with pytest.raises(ValueError):
        window_texts("a b c", 2, 3)
    with pytest.raises(ValueError):
        window_texts("a b c", 2, 0)


def test_max_pool_takes_max_and_first():
    scores = [1.0, 5.0, 2.0, 0.5, 4.0, 3.0]
    chunk_scores, first_scores = max_pool(scores, [3, 1, 2])
    assert chunk_scores == [5.0, 0.5, 4.0]
    assert first_scores == [1.0, 0.5, 4.0]


def test_max_pool_single_window_chunks():
    scores = [2.0, 1.0]
    chunk_scores, first_scores = max_pool(scores, [1, 1])
    assert chunk_scores == [2.0, 1.0]
    assert first_scores == [2.0, 1.0]


def test_max_pool_validates():
    import pytest
    with pytest.raises(ValueError):
        max_pool([1.0], [2])
    with pytest.raises(ValueError):
        max_pool([1.0, 2.0], [0, 2])
    assert max_pool([], []) == ([], [])
    assert max_pool([1.0, 2.0], [2]) == ([2.0], [1.0])


def test_build_window_pairs_windows_and_counts():
    text = " ".join(f"t{i}" for i in range(70))
    docs = {10: text, 20: "x y z"}
    pairs, metas, counts = build_window_pairs("q?", [[10, 0], [20, 1]], docs, top_docs=2,
                                              window_words=30, window_stride=20, max_units=60)
    assert metas == [[10, 1, 0, text], [20, 2, 0, "x y z"]]
    assert counts == [3, 1]
    assert len(pairs) == 4
    assert [p[0] for p in pairs] == ["q?"] * 4
    assert pairs[0][1] == window_texts(text, 30, 20)[0]
    assert pairs[-1] == ("q?", "x y z")
    # max_pool ghép lại đúng chunk
    chunk_scores, first_scores = max_pool([0.1, 0.9, 0.5, 0.2], counts)
    assert chunk_scores == [0.9, 0.2]
    assert first_scores == [0.1, 0.2]


def test_build_window_pairs_no_window_matches_build_chunk_pairs():
    cache1, cache2 = {}, {}
    a = build_window_pairs("q", [[7, 0]], {7: TEXT}, 1, 0, 0, 30, cache1)
    b = build_chunk_pairs("q", [[7, 0]], {7: TEXT}, 1, 30, cache2)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == [1] * len(b[1])


def test_select_top_with_first_scores_column():
    metas = [[1, 1, 0, "a"], [2, 2, 0, "b"], [3, 3, 0, "c"]]
    rows = select_top(metas, [1.0, 3.0, 2.0], top_chunks=2, first_scores=[0.5, 1.5, 2.5])
    assert rows == [[2, 2, 0, 3.0, "b", 1.5], [3, 3, 0, 2.0, "c", 2.5]]


def test_split_first_col_keeps_main_format():
    rows = [[2, 2, 0, 3.0, "b", 1.5], [3, 3, 0, 2.0, "c", 2.5]]
    main_rows, first_rows = split_first_col(rows)
    assert main_rows == [[2, 2, 0, 3.0, "b"], [3, 3, 0, 2.0, "c"]]
    assert first_rows == [[2, 2, 0, 1.5], [3, 3, 0, 2.5]]
