import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from coverage21 import (  # noqa: E402
    classify_host,
    coverage_report,
    host_table,
    lang_block,
    missing_by_host,
    scan_records,
    split_corpus,
    sub_text_fraction,
    zh_emb_ids,
)


def test_classify_host():
    assert classify_host("pubmed.ncbi.nlm.nih.gov") == "en"
    assert classify_host("www.120ask.com") == "zh"
    assert classify_host("zysjonline.com") == "zh"
    assert classify_host("ask.39.net") == "zh"
    assert classify_host("baby.39.net") == "zh"
    assert classify_host("www.familydoctor.com.cn") == "zh"
    assert classify_host("www.msdmanuals.cn") == "zh"
    assert classify_host("test.pmphai.com") == "zh"
    assert classify_host("www.youlai.cn") == "zh"
    assert classify_host("suckhoedoisong.vn") == "vi"
    assert classify_host("www.vinmec.com") == "vi"
    assert classify_host("thanhnien.vn") == "vi"
    assert classify_host("") == "vi"


def test_split_corpus_totals():
    ids = [1, 2, 3, 4]
    urls = ["https://a.vn/x", "https://www.120ask.com/y", "https://pubmed.ncbi.nlm.nih.gov/4/", "https://b.39.net/z"]
    sp = split_corpus(ids, urls)
    assert set(sp) == {"vi", "zh", "en"}
    assert sp["vi"] == {1: "https://a.vn/x"}
    assert set(sp["zh"]) == {2, 4}
    assert set(sp["en"]) == {3}


def test_scan_records_text_sets_and_votes():
    recs = [
        {"id": 1, "host": "a.vn", "status": 200, "text": "đau đầu " * 40},       # ok
        {"id": 2, "host": "a.vn", "status": 200, "text": "ngắn"},                # chỉ text_any
        {"id": 3, "host": "b.cn", "status": 200, "text": "头痛头晕 " * 50},       # ok, vote zh
        {"id": 4, "host": "b.cn", "status": 500, "text": "x" * 300},             # không tính
        {"id": 5, "host": "a.vn", "status": 200, "text": ""},                    # không tính
        {"id": 6, "host": "a.vn", "status": 200, "text": "khỏe " * 40},          # ok
    ]
    s = scan_records(recs, min_chars=200)
    assert s["text_ok"] == {1, 3, 6}
    assert s["text_any"] == {1, 2, 3, 6}
    assert s["records"] == 6
    assert s["votes"]["a.vn"]["vi"] == 3
    assert s["votes"]["b.cn"]["zh"] == 1


def test_missing_by_host_and_table():
    id_urls = {1: "https://a.vn/1", 2: "https://a.vn/2", 3: "https://b.cn/3", 4: "https://b.cn/4"}
    have = {1, 3}
    m = missing_by_host(id_urls, have)
    assert m == {"b.cn": [4], "a.vn": [2]}
    t = host_table(id_urls, have, top=2)
    rows = {r["host"]: r for r in t["all"]}
    assert rows["a.vn"] == {"host": "a.vn", "total": 2, "has_text": 1, "missing": 1}
    assert t["top_missing"][0]["host"] in ("a.vn", "b.cn")


def test_lang_block_and_sub_fraction():
    id_urls = {1: "https://a.vn/1", 2: "https://a.vn/2", 3: "https://a.vn/3"}
    b = lang_block(id_urls, text_ok={1, 2}, text_any={1, 2, 3}, index_ids={1})
    assert b["total"] == 3 and b["has_text"] == 2 and b["missing"] == 1
    assert b["in_index"] == 1 and b["text_not_in_index"] == 1
    sub = [
        {"id": 1, "relevant_docs": [1, 2, 9], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [1], "relevant_chunks": []},
    ]
    s = sub_text_fraction(sub, {1, 2})
    assert s["n_queries"] == 2 and s["n_doc_mentions"] == 4
    assert s["frac_mentions_in_text"] == 0.75
    assert s["frac_unique_in_text"] == round(2 / 3, 6)


def test_coverage_report_and_readme_match():
    vi = {1: "https://a.vn/1", 2: "https://a.vn/2"}
    zh = {10: "https://b.cn/10"}
    en = {20: "https://pubmed.ncbi.nlm.nih.gov/20/"}
    new = {"vi": vi, "zh": zh, "en": en}
    rep = coverage_report(new, text_ok={1, 10}, text_any={1, 2, 10}, vi_index={1}, zh_index={10},
                          zh_emb={10}, votes={"a.vn": {"vi": 2}, "b.cn": {"zh": 1}},
                          sub=[{"id": 1, "relevant_docs": [1, 10], "relevant_chunks": []}])
    assert rep["new_corpus"]["by_lang"] == {"vi": 2, "zh": 1, "en": 1}
    assert rep["new_corpus"]["match_readme"] is False
    assert rep["zh"]["in_zh_embedded"] == 1
    assert rep["zh_intersection_embedded"] == 1
    assert rep["vi_text_not_in_index"]["n"] == 0
    assert rep["sub_sanity"]["frac_mentions_in_text"] == 1.0
    assert rep["host_lang_votes"]["agree_with_rule"] == 2


def test_vi_text_not_in_index_grouped():
    vi = {1: "https://a.vn/1", 2: "https://a.vn/2", 3: "https://c.vn/3"}
    new = {"vi": vi, "zh": {}, "en": {}}
    rep = coverage_report(new, text_ok={1, 2, 3}, text_any={1, 2, 3}, vi_index={1}, zh_index=None,
                          zh_emb=None, votes={}, sub=None)
    assert rep["vi_text_not_in_index"]["n"] == 2
    assert rep["vi_text_not_in_index"]["by_host_n"] == {"a.vn": 1, "c.vn": 1}


def test_zh_emb_ids_reads_parts(tmp_path):
    import zstandard
    p = tmp_path / "docs_zh_all.part000.jsonl.zst"
    lines = [
        json.dumps({"id": 7, "idx": 0, "text": "头痛"}, ensure_ascii=False),
        json.dumps({"id": 9, "idx": 1, "text": "x"}, ensure_ascii=False),
    ]
    with p.open("wb") as f:
        f.write(zstandard.ZstdCompressor().compress(("\n".join(lines) + "\n").encode("utf-8")))
    ids, n = zh_emb_ids(tmp_path)
    assert ids == {7, 9} and n == 2
