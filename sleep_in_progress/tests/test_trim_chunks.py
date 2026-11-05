import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from trim_chunks import (best_window, cut_chunk, entry_len, item_len, list_json_len,  # noqa: E402
                         locate_chunk, overlap_chars, trim_around, word_spans)

# tu dang "w000" = 4 ky tu + 1 khoang trang => buoc 5 ky tu/ tu
STEP = 5


def _words(n, prefix="w"):
    return [f"{prefix}{i:03d}" for i in range(n)]


def _chunk(n, prefix="w"):
    return " ".join(_words(n, prefix))


def _span(n, a, b, prefix="w"):
    return " ".join(_words(n, prefix)[a:b])


def _win(a, b):
    """Cua so tu tu a den tu b (bao ca), toa do ky tu trong chunk."""
    return [a * STEP, b * STEP + STEP - 1]


def test_word_spans():
    s = "  a bb   ccc "
    assert word_spans(s) == [(2, 3), (4, 6), (9, 12)]


def test_trim_can_giua():
    chunk = _chunk(100)
    out = trim_around(chunk, *_win(40, 69), 50)
    assert out == _span(100, 30, 80)
    assert len(out.split()) == 50
    assert chunk.find(out) >= 0


def test_trim_can_giua_chua_toan_bo_cua_so():
    chunk = _chunk(100)
    out = trim_around(chunk, *_win(40, 69), 50)
    assert out.find("w040") >= 0 and out.rfind("w069") < len(out)
    assert out.split()[0] == "w030" and out.split()[-1] == "w079"
    assert "w029" not in out and "w080" not in out


def test_trim_chan_bien_dau():
    chunk = _chunk(100)
    out = trim_around(chunk, *_win(0, 9), 50)
    assert out == _span(100, 0, 50)
    assert chunk.find(out) == 0
    assert len(out.split()) == 50


def test_trim_chan_bien_cuoi():
    chunk = _chunk(100)
    out = trim_around(chunk, *_win(90, 99), 50)
    assert out == _span(100, 50, 100)
    assert chunk.endswith(out.split()[-1])
    assert len(out.split()) == 50


def test_trim_nguyen_van():
    chunk = _chunk(80, "t")
    out = trim_around(chunk, *_win(30, 59), 45)
    i = chunk.find(out)
    assert i >= 0
    assert chunk[i:i + len(out)] == out
    assert set(out.split()) <= set(chunk.split())


def test_trim_chia_het_khoang_trang():
    chunk = _chunk(80, "t")
    out = trim_around(chunk, *_win(30, 59), 45)
    assert out.split()[0] == "t023" and out.split()[-1] == "t067"
    assert len(out.split()) == 45


def test_chunk_ngan_giu_nguyen():
    chunk = _chunk(50)
    assert trim_around(chunk, *_win(0, 9), 50) == chunk
    assert trim_around(chunk, *_win(12, 23), 45) != chunk
    assert trim_around(_chunk(10), 0, 30, 50) == _chunk(10)
    assert trim_around(_chunk(10), 0, 30, 0) == _chunk(10)


def test_trim_cua_so_rong_giu_nguyen():
    chunk = _chunk(80)
    assert trim_around(chunk, 500, 600, 50) == chunk


def test_trim_cua_so_nho_giua_chunk():
    chunk = _chunk(200)
    out = trim_around(chunk, *_win(99, 100), 10)
    assert len(out.split()) == 10
    assert out.split()[0] == "w095" and out.split()[-1] == "w104"


def test_overlap_chars():
    assert overlap_chars(0, 10, 5, 15) == 5
    assert overlap_chars(0, 10, 10, 20) == 0
    assert overlap_chars(0, 10, 0, 10) == 10


def test_best_window_chon_logit_cao():
    wins = [[0, 100, 1.0], [50, 150, 5.0], [200, 300, 9.0]]
    assert best_window(wins, 0, 200) == (50, 150, 5.0)
    assert best_window(wins, 0, 80) == (0, 100, 1.0)


def test_best_window_nguong_50_phan_tram():
    wins = [[0, 100, 3.0]]
    assert best_window(wins, 50, 150) == (0, 100, 3.0)
    assert best_window(wins, 51, 151) is None
    assert best_window([], 0, 10) is None


def test_cut_chunk_khong_cua_so_giu_nguyen():
    text = "dau " + _chunk(30) + " cuoi"
    chunk = _chunk(30)
    out, meta = cut_chunk(text, chunk, [], 20)
    assert out == chunk and meta["tim_duoc"] and not meta["co_cua_so"] and not meta["cat"]


def test_cut_chunk_khong_tim_thay_text():
    out, meta = cut_chunk("khac het", _chunk(30), [[0, 100, 1.0]], 20)
    assert out == _chunk(30) and not meta["tim_duoc"]


def test_cut_chunk_cat_quanh_cua_so():
    text = "xxx " + _chunk(100) + " yyy"
    chunk = _chunk(100)
    cs = text.find(chunk)
    wins = [[cs + _win(40, 69)[0], cs + _win(40, 69)[1], 1.0],
            [cs + _win(10, 19)[0], cs + _win(10, 19)[1], 0.5]]
    out, meta = cut_chunk(text, chunk, wins, 50)
    assert meta["co_cua_so"] and meta["cat"] and meta["logit"] == 1.0
    assert out == _span(100, 30, 80)
    assert text.find(out) >= 0 and chunk.find(out) >= 0


def test_cut_chunk_chunk_ngan_khong_cat():
    text = _chunk(20)
    out, meta = cut_chunk(text, text, [[0, 120, 1.0]], 450)
    assert out == text and meta["tim_duoc"] and not meta["cat"]


def test_cut_chunk_hai_vi_tri_trong_text():
    c = _chunk(60, "d")
    text = "mo " + c + " giua " + c + " het"
    wins = [[text.rfind(c) + _win(20, 39)[0], text.rfind(c) + _win(20, 39)[1], 2.0]]
    out, meta = cut_chunk(text, c, wins, 30)
    assert meta["vi_tri"] == text.rfind(c)
    assert out == _span(60, 15, 45, "d")


def test_entry_item_len_trung_json_dumps():
    e = {"doc_id": 12, "chunk_text": "tiếng việt có dấu"}
    assert entry_len(12, "tiếng việt có dấu") == len(json.dumps(e, ensure_ascii=False))
    items = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": [e]},
             {"id": 2, "relevant_docs": [3], "relevant_chunks": []}]
    got = list_json_len([item_len(1, [1, 2], entry_len(12, "tiếng việt có dấu"), 1),
                         item_len(2, [3], 0, 0)])
    assert got == len(json.dumps(items, ensure_ascii=False))


def test_locate_chunk():
    assert locate_chunk("aXbXc", "X") == [1, 3]
    assert locate_chunk("abc", "z") == []
