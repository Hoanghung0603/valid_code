import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rewrite_queries import (
    MODES,
    build_combined,
    len_ok,
    load_done,
    load_queries,
    merge_text,
    strip_think,
    word_count,
)


def write_jsonl(p: Path, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_strip_think():
    open_t, close_t = "<" + "thinking" + ">", "</" + "thinking>"
    assert strip_think("trước khi suy nghĩ thì sau") == "trước khi suy nghĩ thì sau"
    assert strip_think(f"trước {open_t}nội bộ{close_t} sau") == "trước  sau"
    assert strip_think(f"{open_t}\nnội bộ\nnhiều dòng{close_t}cuối") == "cuối"
    assert strip_think(None) == ""
    assert strip_think("") == ""
    assert strip_think("  x  ") == "x"


def test_word_count_and_len_ok():
    assert word_count("một hai  ba") == 3
    assert word_count("") == 0
    assert word_count("đau bụng vùng rốn âm ỉ") == 6
    assert len_ok("một hai", 1, 3)
    assert not len_ok("một", 2, 3)
    assert not len_ok("một hai ba bốn", 2, 3)


def test_load_queries_jsonl_dedupe_and_skip(tmp_path):
    p = tmp_path / "q.jsonl"
    write_jsonl(p, [
        {"id": 1, "query": "câu một"},
        {"id": 2, "query": "câu hai"},
        {"id": 1, "query": "trùng id, bỏ"},
        {"id": 3, "query": ""},
        {"id": 4, "query": "câu bốn"},
    ])
    assert load_queries(p) == [(1, "câu một"), (2, "câu hai"), (4, "câu bốn")]


def test_load_queries_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    p = tmp_path / "q.parquet"
    pd.DataFrame({"id": [7, 8], "query": ["a", "b"]}).to_parquet(p)
    assert load_queries(p) == [(7, "a"), (8, "b")]


def test_load_done_resume_only_nonempty(tmp_path):
    p = tmp_path / "hyde.jsonl"
    write_jsonl(p, [
        {"id": 1, "query": "q1", "hyde": "đoạn xong"},
        {"id": 2, "query": "q2", "hyde": ""},
        {"id": 3, "query": "q3", "hyde": "đoạn khác"},
    ])
    done = load_done(p, "hyde")
    assert sorted(done) == [1, 3]
    assert done[1]["hyde"] == "đoạn xong"
    assert load_done(tmp_path / "missing.jsonl", "hyde") == {}
    assert load_done(p, "reason") == {}


def test_load_done_skips_broken_and_blank_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({"id": 1, "query": "q", "reason": "xong"}, ensure_ascii=False) + "\n"
        + "\n"
        + json.dumps({"id": 2, "query": "q", "reason": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert sorted(load_done(p, "reason")) == [1]


def test_merge_and_build_combined():
    assert merge_text("câu hỏi gốc", "đoạn sinh") == "câu hỏi gốc\nđoạn sinh"
    assert merge_text("câu hỏi gốc", "") == "câu hỏi gốc"
    assert merge_text("  câu hỏi gốc  ", None) == "câu hỏi gốc"
    gen = {1: {"hyde": "đoạn một"}, 2: {"hyde": ""}}
    out = build_combined([(1, "gốc một"), (2, "gốc hai"), (3, "gốc ba")], gen, "hyde")
    assert out == [
        {"id": 1, "query": "gốc một\nđoạn một"},
        {"id": 2, "query": "gốc hai"},
        {"id": 3, "query": "gốc ba"},
    ]


def test_modes_config_ranges():
    assert MODES["hyde"]["field"] == "hyde"
    assert (MODES["hyde"]["min_words"], MODES["hyde"]["max_words"]) == (80, 150)
    assert MODES["reason"]["field"] == "reason"
    assert (MODES["reason"]["min_words"], MODES["reason"]["max_words"]) == (40, 90)
