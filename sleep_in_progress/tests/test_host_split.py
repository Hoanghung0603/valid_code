import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from host_split import format_report, host_of, is_h, split_cache  # noqa: E402

KEYS = ["vinmec", "suckhoe", "benhvien", "thuoc", "hospital"]


def test_host_of():
    assert host_of("https://www.Vinmec.com/tre-em/bai-viet") == "vinmec.com"
    assert host_of("http://phunusuckhoe.giadinhonline.vn/a.html") == "phunusuckhoe.giadinhonline.vn"
    assert host_of("not a url") == ""
    assert host_of("") == ""


def test_is_h():
    assert is_h("vinmec.com", KEYS)
    assert is_h("suckhoedoisong.vn", KEYS)
    assert is_h("thaythuocvietnam.vn", KEYS)
    assert is_h("a-hospital.com", KEYS)
    assert not is_h("thanhnien.vn", KEYS)
    assert not is_h("", KEYS)


def test_split_cache_keeps_order_and_groups():
    cache = {
        "1": [[1, 0, 3.0], [2, 0, 2.0], [3, 0, 1.0]],
        "2": [[2, 0, 3.0]],
        "3": [[4, 0, 3.0]],
    }
    id2url = {1: "https://vinmec.com/a", 2: "https://thanhnien.vn/b", 3: "https://benhvienbachmai.gov.vn/c",
              4: "https://vnexpress.net/d"}
    h_cache, n_cache, stats = split_cache(cache, id2url, KEYS, top=30)
    assert [r[0] for r in h_cache["1"]] == [1, 3]
    assert [r[0] for r in n_cache["1"]] == [2]
    assert h_cache["2"] == [] and [r[0] for r in n_cache["2"]] == [2]
    assert [r[0] for r in n_cache["3"]] == [4]
    assert stats["câu có ≥1 H"] == 1
    assert stats["câu có ≥1 N"] == 3
    assert stats["câu có cả H và N"] == 1
    assert stats["TB doc H/câu"] == 0.6667
    assert stats["hosts"][0] == {"host": "thanhnien.vn", "nhóm": "N", "số lần": 2}


def test_split_cache_top_limit():
    cache = {"1": [[1, 0, 3.0], [2, 0, 2.0], [3, 0, 1.0]]}
    id2url = {1: "https://vinmec.com/a", 2: "https://thanhnien.vn/b", 3: "https://medlatec.vn/c"}
    h_cache, n_cache, _ = split_cache(cache, id2url, KEYS, top=2)
    assert [r[0] for r in h_cache["1"]] == [1]
    assert [r[0] for r in n_cache["1"]] == [2]


def test_missing_url_is_n():
    cache = {"1": [[9, 0, 3.0]]}
    h_cache, n_cache, stats = split_cache(cache, {}, KEYS, top=30)
    assert h_cache["1"] == [] and [r[0] for r in n_cache["1"]] == [9]
    assert stats["hosts"][0]["host"] == ""


def test_format_report_has_table_and_stats():
    cache = {"1": [[1, 0, 3.0], [2, 0, 2.0]]}
    id2url = {1: "https://vinmec.com/a", 2: "https://thanhnien.vn/b"}
    _, _, stats = split_cache(cache, id2url, KEYS, top=30)
    rep = format_report(stats)
    assert "số lần  nhóm  host" in rep
    assert "vinmec.com" in rep and "thanhnien.vn" in rep
    assert "TB doc H/câu" in rep
