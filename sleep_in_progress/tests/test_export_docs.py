import json
import os
import sys
from pathlib import Path

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bigindex import build
from bigpipeline import load_texts
from boilerplate import BoilerplateFilter
from export_docs import PartWriter, iter_all_texts, make_body
from shards import ShardWriter

DOCS = [
    {"id": 101, "status": 200, "host": "h.vn", "title": "Sốt xuất huyết",
     "text": "Sốt xuất huyết dengue do muỗi vằn truyền.\nQUẢNG CÁO THUỐC\nBệnh nhân sốt cao, đau đầu. " * 3},
    {"id": 102, "status": 200, "host": "h.vn", "title": "Đau dạ dày",
     "text": "Viêm loét dạ dày tá tràng gây đau thượng vị, ợ chua. " * 5},
    {"id": 103, "status": 200, "host": "h2.vn", "title": "Tiểu đường",
     "text": "Đái tháo đường tuýp 2 cần kiểm soát đường huyết, insulin và chế độ ăn. " * 5},
    {"id": 104, "status": 404, "host": "h.vn", "title": "Mất", "text": "Không thấy trang này. " * 20},
]


def _bp() -> BoilerplateFilter:
    bp = BoilerplateFilter(min_docs=1, max_share=0.5, max_fit_docs=3000)
    bp._lines = {"h.vn": {"QUẢNG CÁO THUỐC"}}
    return bp


def _index(tmp: Path):
    d = tmp / "shards"
    w = ShardWriter(d, "crawl", max_records=2)
    for r in DOCS:
        w.write(r)
    w.close()
    build(sorted(d.glob("*.jsonl.zst")), tmp / "idx", "vi", nbits=20, workers=1, log=lambda m: None)
    import bigindex
    return bigindex.BigIndex(tmp / "idx")


def test_make_body():
    assert make_body("T", "x  y", 100) == "T\nx  y"
    assert make_body("T", "abcdef", 4) == "T\nab"
    assert make_body("", "  nội dung  ", 100).strip() == "nội dung"


def test_make_body_no_title():
    assert make_body("T", "x  y", 100, no_title=True) == "x  y"
    assert make_body("T", "abcdef", 4, no_title=True) == "abcd"
    assert make_body("", "  nội dung  ", 100, no_title=True) == "  nội dung  "


def test_iter_all_matches_load_texts(tmp_path):
    idx = _index(tmp_path)
    bp = _bp()
    wanted = {idx.doc_id(i): i for i in range(idx.n)}
    got = {d: body for d, _, body in iter_all_texts(idx, wanted, bp, 3000, log=lambda m: None)}
    texts = load_texts(idx, wanted, bp, log=lambda m: None)
    ref = {d: make_body(r["title"], r["text"], 3000) for d, r in texts.items()}
    assert got == ref
    assert 104 not in got, "status 404 không được xuất"
    assert "QUẢNG CÁO THUỐC" not in got[101]
    assert "Bệnh nhân sốt cao" in got[101]
    assert set(got) == set(wanted)


def test_iter_all_carries_idx_and_missing(tmp_path):
    idx = _index(tmp_path)
    bp = _bp()
    logs = []
    wanted = {idx.doc_id(i): i for i in range(idx.n)}
    wanted[999999] = 0
    rows = {(d, i) for d, i, _ in iter_all_texts(idx, wanted, bp, 3000, log=logs.append)}
    assert all(i == wanted[d] for d, i in rows)
    assert 999999 not in {d for d, _ in rows}
    assert any("CẢNH BÁO" in m for m in logs)


def test_iter_all_no_title_matches_load_texts(tmp_path):
    idx = _index(tmp_path)
    bp = _bp()
    wanted = {idx.doc_id(i): i for i in range(idx.n)}
    got = {d: body for d, _, body in iter_all_texts(idx, wanted, bp, 3000, log=lambda m: None, no_title=True)}
    texts = load_texts(idx, wanted, bp, log=lambda m: None)
    ref = {d: r["text"][:3000] for d, r in texts.items()}
    assert got == ref
    assert 104 not in got
    assert not got[101].startswith("Sốt xuất huyết\n")


def test_part_writer_roundtrip(tmp_path):
    out = tmp_path / "all.jsonl.zst"
    w = PartWriter(str(out), part_mb=1, log=lambda m: None)
    recs = [{"id": k, "idx": k, "text": os.urandom(2000).hex()} for k in range(2500)]
    for r in recs:
        w.write(r)
    w.close()
    assert len(w.paths) >= 2, "nhiều bản ghi dài phải xoay nhiều part"
    got = []
    for p in w.paths:
        with open(p, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as r:
                for ln in r.read().decode("utf-8").splitlines():
                    got.append(json.loads(ln))
    assert got == recs


def test_part_writer_single_file(tmp_path):
    out = tmp_path / "one.jsonl.zst"
    w = PartWriter(str(out), part_mb=0, log=lambda m: None)
    w.write({"id": 1, "idx": 2, "text": "x"})
    w.close()
    assert w.paths == [out]
    with open(out, "rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as r:
            assert json.loads(r.read().decode("utf-8")) == {"id": 1, "idx": 2, "text": "x"}
