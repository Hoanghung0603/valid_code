import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from docwin_run import (  # noqa: E402
    build_docwin_pairs,
    pack_doc_scores,
    window_spans,
    write_zst,
)

TEXT = "Bệnh nhân sốt cao liên tục trong ba ngày thì đến khám. " * 40 + \
       "\n" + "Điều trị bằng truyền dịch và theo dõi tiểu cầu. " * 40


def test_window_spans_offsets_are_verbatim_word_aligned():
    text = "Đầu  dòng\nthứ hai   và ba.\nCuối cùng!"
    spans = window_spans(text, 3, 2)
    for s, e in spans:
        assert 0 <= s < e <= len(text)
        sub = text[s:e]
        assert sub == sub.strip() or sub[0] != " ", "cửa sổ bám biên từ"
        assert sub.split(), "cửa sổ không rỗng"
    assert [text[s:e] for s, e in spans][0] == "Đầu  dòng\nthứ"
    assert [text[s:e] for s, e in spans][-1].endswith("Cuối cùng!")


def test_window_spans_covers_all_words():
    words = [f"t{i}" for i in range(100)]
    text = " ".join(words)
    for w, s in [(30, 20), (30, 30), (50, 10), (100, 50), (99, 98), (2, 1)]:
        covered: set = set()
        for a, b in window_spans(text, w, s):
            for wd in text[a:b].split():
                covered.add(int(wd[1:]))
        assert covered == set(range(100)), f"phủ hết từ (w={w}, s={s})"


def test_window_spans_last_window_reaches_end_and_overlap():
    text = " ".join(f"t{i}" for i in range(100))
    spans = window_spans(text, 30, 20)
    wins = [text[s:e] for s, e in spans]
    assert wins[0].split() == [f"t{i}" for i in range(30)]
    assert wins[1].split() == [f"t{i}" for i in range(20, 50)]
    assert wins[-1].split()[-1] == "t99"
    assert all(len(w.split()) == 30 for w in wins[:-1])
    assert 1 <= len(wins[-1].split()) <= 30


def test_window_spans_short_text_single_window():
    for n in (1, 29, 30):
        text = " ".join(f"t{i}" for i in range(n))
        spans = window_spans(text, 30, 20)
        assert len(spans) == 1
        assert text[spans[0][0]:spans[0][1]].split() == text.split()
    assert len(window_spans("ngắn", 300, 250)) == 1


def test_window_spans_empty_text():
    assert window_spans("", 300, 250) == []
    assert window_spans("   \n\t ", 300, 250) == []


def test_window_spans_bad_params():
    with pytest.raises(ValueError):
        window_spans("a b c", 2, 3)
    with pytest.raises(ValueError):
        window_spans("a b c", 2, 0)
    with pytest.raises(ValueError):
        window_spans("a b c", 0, 1)


def test_window_spans_matches_chunk_rerank_window_texts():
    from chunk_rerank import window_texts
    for text in (TEXT, "một hai ba bốn năm sáu bảy tám", "x"):
        for w, s in [(300, 250), (3, 2), (5, 5)]:
            spans = window_spans(text, w, s)
            wins = [text[a:b] for a, b in spans]
            assert wins == window_texts(text, w, s) or (len(wins) == 1 and wins[0].strip()
                                                        == window_texts(text, w, s)[0].strip())
            if len(spans) > 1:
                assert wins == window_texts(text, w, s)


def test_build_docwin_pairs_groups_and_missing_docs():
    docs = {10: TEXT, 20: "x y z"}
    cand = [[10, 100], [99, 101], [20, 102], [30, 103]]
    pairs, groups = build_docwin_pairs("q?", cand, docs, top_docs=4,
                                       window_words=300, window_stride=250)
    assert [g[0] for g in groups] == [10, 20]
    assert sum(len(g[1]) for g in groups) == len(pairs)
    assert all(p[0] == "q?" for p in pairs)
    t20 = "x y z"
    assert groups[1][1] == [(0, len(t20))]
    assert pairs[-1] == ("q?", t20)
    # cửa sổ của doc đầu là nguyên văn text
    s0, e0 = groups[0][1][0]
    assert pairs[0] == ("q?", TEXT[s0:e0])
    assert TEXT[s0:e0] in TEXT


def test_build_docwin_pairs_respects_top_docs_and_skips_empty():
    docs = {1: "a b c", 2: "", 3: "d e"}
    cand = [[1, 0], [2, 1], [3, 2]]
    pairs, groups = build_docwin_pairs("q", cand, docs, top_docs=2,
                                       window_words=2, window_stride=1)
    assert [g[0] for g in groups] == [1]
    assert len(pairs) == 2
    pairs, groups = build_docwin_pairs("q", cand, docs, top_docs=3,
                                       window_words=2, window_stride=1)
    assert [g[0] for g in groups] == [1, 3]


def test_build_docwin_pairs_span_cache_reused():
    calls = {"n": 0}
    docs = {7: "một hai ba bốn năm sáu bảy tám chín mười"}
    cache = {}

    def spy(text, w, s):
        calls["n"] += 1
        return window_spans(text, w, s)

    real = window_spans
    import docwin_run
    docwin_run.window_spans = spy
    try:
        build_docwin_pairs("q", [[7, 0]], docs, 1, 3, 2, cache)
        build_docwin_pairs("q", [[7, 0]], docs, 1, 3, 2, cache)
    finally:
        docwin_run.window_spans = real
    assert calls["n"] == 1, "cache: chỉ cắt cửa sổ một lần"
    assert cache[7] == real(docs[7], 3, 2)


def test_pack_doc_scores_groups_and_rounds():
    groups = [(10, [(0, 5), (3, 8)]), (20, [(0, 2)])]
    out = pack_doc_scores(groups, [1.23456, 2.0, -3.14159])
    assert out == {10: [[0, 5, 1.2346], [3, 8, 2.0]], 20: [[0, 2, -3.1416]]}
    out3 = pack_doc_scores(groups, [1.23456, 2.0, -3.14159], ndigits=3)
    assert out3 == {10: [[0, 5, 1.235], [3, 8, 2.0]], 20: [[0, 2, -3.142]]}
    assert pack_doc_scores([], []) == {}


def test_write_zst_roundtrip(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    data = {"1": {10: [[0, 5, 3.1416]], 20: [[2, 9, -0.5]]}}
    p = write_zst(str(tmp_path / "out.json"), data)
    assert p.name == "out.json.zst"
    with open(p, "rb") as f:
        raw = zstandard.ZstdDecompressor().decompress(f.read(), max_output_size=1 << 20)
    # JSON đổi key int -> str khi round-trip
    assert json.loads(raw.decode("utf-8")) == {"1": {"10": [[0, 5, 3.1416]], "20": [[2, 9, -0.5]]}}


def test_write_zst_rounds_3_when_over_limit(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    data = {"1": {10: [[0, 5, 3.14159]]}}
    p = write_zst(str(tmp_path / "out.json"), data, limit=10)
    with open(p, "rb") as f:
        raw = zstandard.ZstdDecompressor().decompress(f.read(), max_output_size=1 << 20)
    assert json.loads(raw.decode("utf-8")) == {"1": {"10": [[0, 5, 3.142]]}}


def test_import_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    import docwin_run
    import importlib
    importlib.reload(docwin_run)
    assert docwin_run.PART_SIZE == 50
    assert callable(docwin_run.score_pairs)
