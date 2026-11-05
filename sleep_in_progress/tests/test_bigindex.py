import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bigindex import BigIndex, build
from bm25 import BM25, tokenize
from shards import ShardWriter

VI = [
    ("Sốt xuất huyết", "Sốt xuất huyết dengue do muỗi vằn truyền. Bệnh nhân sốt cao, đau đầu, xuất huyết dưới da. " * 4),
    ("Đau dạ dày", "Viêm loét dạ dày tá tràng gây đau thượng vị, ợ chua, buồn nôn sau khi ăn. " * 5),
    ("Tiểu đường", "Đái tháo đường tuýp 2 cần kiểm soát đường huyết, insulin và chế độ ăn hợp lý. " * 5),
    ("Cao huyết áp", "Tăng huyết áp làm tăng nguy cơ đột quỵ, cần đo huyết áp đều đặn mỗi ngày. " * 5),
    ("Viêm gan B", "Virus viêm gan B lây qua đường máu, mẹ truyền sang con và quan hệ tình dục. " * 5),
]
ZH = "五倍子适量，为细末，外用敷于患处，每日两次。" * 30


def _write(tmp: Path) -> list[Path]:
    d = tmp / "shards"
    w = ShardWriter(d, "crawl", max_records=3)
    for i, (t, x) in enumerate(VI):
        w.write({"id": 100 + i, "status": 200, "host": "h.vn", "title": t, "text": x})
    w.write({"id": 200, "status": 200, "host": "h.cn", "title": "", "text": ZH})
    w.write({"id": 300, "status": 404, "host": "h.vn", "title": "x", "text": VI[0][1]})
    w.close()
    return sorted(d.glob("*.jsonl.zst"))


@pytest.mark.parametrize("workers", [1])
def test_matches_bm25s_and_filters_lang(tmp_path, workers):
    shards = _write(tmp_path)
    assert len(shards) >= 2, "cần nhiều part để thử gộp"
    meta = build(shards, tmp_path / "idx", "vi", nbits=20, workers=workers, budget=50, log=lambda m: None)
    assert meta["docs"] == 5 and meta["unique_ids"] == 5  # bỏ doc ZH và status 404
    assert meta["ranges"] > 1, "budget nhỏ phải chia nhiều dải"

    idx = BigIndex(tmp_path / "idx")
    texts = [t + "\n" + x for t, x in VI]
    ref = BM25().build([tokenize(s, "vi") for s in texts])
    for q in ["muỗi vằn gây sốt xuất huyết", "đau thượng vị ợ chua", "đường huyết insulin", "virus lây qua đường máu"]:
        qt = sorted(set(tokenize(q, "vi")))
        got = [(idx.doc_id(i), s) for i, s in idx.search(qt, k=3)]
        want = [(100 + i, s) for i, s in ref.search(qt, k=3)]
        assert [g[0] for g in got] == [w[0] for w in want], q
        for (_, gs), (_, ws) in zip(got, want):
            assert gs == pytest.approx(ws, rel=1e-3)


def test_unknown_query_empty_and_parts_cleaned(tmp_path):
    shards = _write(tmp_path)
    build(shards, tmp_path / "idx", "vi", nbits=20, workers=1, log=lambda m: None)
    idx = BigIndex(tmp_path / "idx")
    assert idx.search(tokenize("zzzz qqqq", "vi")) == []
    assert idx.search([]) == []
    assert not (tmp_path / "idx" / "parts").exists()
    assert idx.shard_of(0).endswith(".jsonl.zst")
