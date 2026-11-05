import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sort_cache import sort_cache, sort_rows  # noqa: E402


def test_sort_rows_giam_dan():
    rows = [["a", 0, 1.0], ["b", 1, 3.0], ["c", 2, 2.0]]
    assert sort_rows(rows) == [["b", 1, 3.0], ["c", 2, 2.0], ["a", 0, 1.0]]


def test_sort_rows_none_xuong_cuoi():
    rows = [["a", 0, None], ["b", 1, 0.5], ["c", 2, None]]
    assert sort_rows(rows) == [["b", 1, 0.5], ["a", 0, None], ["c", 2, None]]


def test_sort_rows_dong_diem_giu_thu_tu_goc():
    rows = [["a", 0, 1.0], ["b", 1, 1.0], ["c", 2, 1.0], ["d", 3, 2.0]]
    assert sort_rows(rows) == [["d", 3, 2.0], ["a", 0, 1.0], ["b", 1, 1.0], ["c", 2, 1.0]]


def test_sort_rows_giu_nguyen_dong():
    rows = [["a", 0, -1.5], ["b", 1, None], ["c", 2, 0.0]]
    out = sort_rows(rows)
    assert all(r in rows for r in out)
    assert len(out) == len(rows)


def test_sort_cache_giu_thu_tu_cau():
    cache = {"2": [["a", 0, 1.0]], "1": [["b", 1, 0.0]]}
    out = sort_cache(cache)
    assert list(out.keys()) == ["2", "1"]
    assert out["2"] == [["a", 0, 1.0]]


def test_sort_cache_dinh_dang_giuyen(tmp_path):
    cache = {"1": [["a", 0, 1.0], ["b", 1, 2.0]], "2": [["c", 2, None]]}
    p = tmp_path / "sorted.json"
    p.write_text(json.dumps(sort_cache(cache)), encoding="utf-8")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == {"1": [["b", 1, 2.0], ["a", 0, 1.0]], "2": [["c", 2, None]]}
    assert all(len(r) == 3 for rows in data.values() for r in rows)
