import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bigindex import BigIndex, build
from bigpipeline import build_submission
from boilerplate import BoilerplateFilter
from evaluate import validate
from shards import ShardWriter

DOCS = {
    1: ("Sốt xuất huyết", "Sốt xuất huyết dengue do muỗi vằn truyền. Bệnh nhân sốt cao, đau đầu.\n" * 10),
    2: ("Đau dạ dày", "Viêm loét dạ dày tá tràng gây đau thượng vị, ợ chua, buồn nôn.\n" * 10),
    3: ("Tiểu đường", "Đái tháo đường tuýp 2 cần kiểm soát đường huyết và insulin.\n" * 10),
    4: ("Cao huyết áp", "Tăng huyết áp làm tăng nguy cơ đột quỵ, cần đo huyết áp đều đặn.\n" * 10),
}


def _index(tmp: Path):
    d = tmp / "shards"
    w = ShardWriter(d, "crawl", max_records=2)  # 2 shard, để thử đọc theo shard
    for i, (t, x) in DOCS.items():
        w.write({"id": i, "status": 200, "host": "h.vn", "title": t, "text": x})
    w.close()
    build(sorted(d.glob("*.jsonl.zst")), tmp / "idx", "vi", nbits=20, workers=1, log=lambda m: None)
    idx = BigIndex(tmp / "idx")
    return idx, BoilerplateFilter.from_json((tmp / "idx" / "boilerplate.json").read_text(encoding="utf-8"))


def test_submission_matches_docs_and_chunks_verbatim(tmp_path):
    idx, bp = _index(tmp_path)
    sub = build_submission(idx, bp, [(10, "muỗi vằn gây sốt xuất huyết dengue"), (11, "đau thượng vị ợ chua"), (12, "zzzz qqqq")],
                           chunk_units=20, log=lambda m: None)
    assert validate(sub, 3) == []
    assert sub[0]["relevant_docs"][0] == 1
    assert sub[1]["relevant_docs"][0] == 2
    assert sub[2] == {"id": 12, "relevant_docs": [], "relevant_chunks": []}
    for s in sub[:2]:
        assert s["relevant_chunks"]
        for c in s["relevant_chunks"]:
            assert c["doc_id"] in s["relevant_docs"]
            assert c["chunk_text"] in DOCS[c["doc_id"]][1]


def test_chunk_docs_limits_chunk_sources_but_not_docs(tmp_path):
    idx, bp = _index(tmp_path)
    q = [(1, "sốt xuất huyết đau dạ dày tiểu đường huyết áp")]
    full = build_submission(idx, bp, q, top_docs=4, rel_ratio=0.0, chunk_units=20, max_chunks=50, log=lambda m: None)[0]
    cut = build_submission(idx, bp, q, top_docs=4, rel_ratio=0.0, chunk_units=20, max_chunks=50, chunk_docs=1,
                           log=lambda m: None)[0]
    assert cut["relevant_docs"] == full["relevant_docs"] and len(cut["relevant_docs"]) > 1
    assert {c["doc_id"] for c in cut["relevant_chunks"]} == {cut["relevant_docs"][0]}
    assert len({c["doc_id"] for c in full["relevant_chunks"]}) > 1
    assert validate([cut], 1) == []


def test_rel_ratio_prunes(tmp_path):
    idx, bp = _index(tmp_path)
    sub = build_submission(idx, bp, [(1, "sốt xuất huyết")], rel_ratio=0.99, log=lambda m: None)
    assert sub[0]["relevant_docs"] == [1]
