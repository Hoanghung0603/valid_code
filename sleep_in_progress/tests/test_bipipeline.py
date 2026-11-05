import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bigindex import BigIndex, build
from bipipeline import build_submission_bi, interleave, load_zh_map
from boilerplate import BoilerplateFilter
from evaluate import validate
from shards import ShardWriter

VI = {
    1: ("Sốt xuất huyết", "Sốt xuất huyết dengue do muỗi vằn truyền. Bệnh nhân sốt cao, đau đầu.\n" * 10),
    2: ("Đau dạ dày", "Viêm loét dạ dày tá tràng gây đau thượng vị, ợ chua, buồn nôn.\n" * 10),
}
ZH = {
    11: ("登革热", "登革热由伊蚊传播，患者出现高热、头痛、皮下出血。\n" * 10),
    12: ("胃痛", "胃溃疡与十二指肠溃疡引起上腹痛、反酸、恶心。\n" * 10),
}


def _mk(tmp: Path, name: str, docs: dict, lang: str):
    d = tmp / name
    w = ShardWriter(d, "crawl", max_records=10)
    for i, (t, x) in docs.items():
        w.write({"id": i, "status": 200, "host": "h", "title": t, "text": x})
    w.close()
    build(sorted(d.glob("*.jsonl.zst")), tmp / f"idx_{name}", lang, nbits=20, workers=1, log=lambda m: None)
    p = tmp / f"idx_{name}"
    return BigIndex(p), BoilerplateFilter.from_json((p / "boilerplate.json").read_text(encoding="utf-8"))


def test_interleave_dedups_and_keeps_rank_order():
    assert interleave([1, 2, 3], [9, 2, 8, 7]) == [1, 9, 2, 3, 8, 7]
    assert interleave([], [5]) == [5]


def test_load_zh_map_skips_empty(tmp_path):
    p = tmp_path / "z.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"id": 1, "query": "a", "zh": ""}, {"id": 1, "query": "a", "zh": "头痛"}, {"id": 2, "query": "b", "zh": ""}]),
        encoding="utf-8")
    assert load_zh_map(p) == {1: "头痛"}


def test_cache_gives_same_submission_as_live_search(tmp_path):
    from bigpipeline import search_docs
    from bipipeline import from_cache
    vi, zh = _mk(tmp_path, "vi", VI, "vi"), _mk(tmp_path, "zh", ZH, "zh")
    q = [(1, "muỗi vằn sốt xuất huyết"), (2, "đau dạ dày")]
    found = search_docs(vi[0], q, 2, 0.0)
    cache = {str(qid): [[d, i, 1.0] for d, i in hits] for qid, hits in found.items()}
    assert from_cache(cache, 1) == {qid: hits[:1] for qid, hits in found.items()}
    kw = dict(n_vi=2, n_zh=0, chunk_docs_vi=1, chunk_docs_zh=0, units_vi=20, units_zh=20, chunks_vi=2, chunks_zh=0,
              log=lambda m: None)
    live = build_submission_bi(vi, zh, q, {}, **kw)
    cached = build_submission_bi(vi, zh, q, {}, cache_vi=cache, **kw)
    assert live == cached


def test_zero_count_drops_a_language(tmp_path):
    vi, zh = _mk(tmp_path, "vi", VI, "vi"), _mk(tmp_path, "zh", ZH, "zh")
    q = [(1, "muỗi vằn sốt xuất huyết")]
    only_zh = build_submission_bi(vi, zh, q, {1: "登革热 伊蚊 高热"}, n_vi=0, n_zh=2, chunk_docs_vi=1, chunk_docs_zh=1,
                                  units_vi=20, units_zh=20, chunks_vi=2, chunks_zh=2, log=lambda m: None)
    only_vi = build_submission_bi(vi, zh, q, {1: "登革热 伊蚊 高热"}, n_vi=2, n_zh=0, chunk_docs_vi=1, chunk_docs_zh=1,
                                  units_vi=20, units_zh=20, chunks_vi=2, chunks_zh=2, log=lambda m: None)
    assert validate(only_zh, 1) == [] and validate(only_vi, 1) == []
    assert all(d >= 10 for d in only_zh[0]["relevant_docs"]) and only_zh[0]["relevant_docs"]
    assert all(c["doc_id"] >= 10 for c in only_zh[0]["relevant_chunks"])
    assert all(d < 10 for d in only_vi[0]["relevant_docs"]) and only_vi[0]["relevant_docs"]
    assert all(c["doc_id"] < 10 for c in only_vi[0]["relevant_chunks"])


def test_bilingual_submission_mixes_languages_and_keeps_chunks_verbatim(tmp_path):
    vi, zh = _mk(tmp_path, "vi", VI, "vi"), _mk(tmp_path, "zh", ZH, "zh")
    sub = build_submission_bi(vi, zh, [(1, "muỗi vằn sốt xuất huyết"), (2, "đau dạ dày")], {1: "登革热 伊蚊 高热"},
                              n_vi=2, n_zh=2, chunk_docs_vi=1, chunk_docs_zh=1, units_vi=20, units_zh=20,
                              chunks_vi=2, chunks_zh=2, log=lambda m: None)
    assert validate(sub, 2) == []
    s1, s2 = sub
    assert 1 in s1["relevant_docs"] and 11 in s1["relevant_docs"]  # cả VI và ZH
    assert all(d < 10 for d in s2["relevant_docs"])  # câu 2 không có bản dịch -> chỉ doc VI
    texts = {**{k: v[1] for k, v in VI.items()}, **{k: v[1] for k, v in ZH.items()}}
    for s in sub:
        for c in s["relevant_chunks"]:
            assert c["doc_id"] in s["relevant_docs"]
            assert c["chunk_text"] in texts[c["doc_id"]]
    assert any(c["doc_id"] == 11 for c in s1["relevant_chunks"])
