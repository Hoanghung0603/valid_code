import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import validate  # noqa: E402
from pubmed_docs import doc_text, export_docs, is_empty_doc, iter_texts  # noqa: E402
from pubmed_logit_stats import compare, scores_at  # noqa: E402
from pubmed_sub import build_sub, check_sub, load_texts  # noqa: E402


def test_doc_text_verbatim():
    assert doc_text("Title A", "Abstract B") == "Title A\nAbstract B"
    assert doc_text("", "Only abstract") == "\nOnly abstract"


def test_is_empty_doc():
    assert is_empty_doc("", "  ")
    assert not is_empty_doc("t", "")
    assert not is_empty_doc("", "a")


def test_build_sub_chunks_match_docs():
    cache = {"2": [[30, 0, 1.0], [10, 1, 0.5], [20, 2, 0.1]],
             "1": [[10, 1, 2.0], [30, 0, 1.0], [40, 3, 0.2]]}
    texts = {10: ("T10", "A10"), 20: ("T20", ""), 30: ("T30", "A30"), 40: ("", "")}
    items = build_sub(cache, texts, top=3)
    assert [it["id"] for it in items] == [1, 2]  # id tăng dần
    assert items[0]["relevant_docs"] == [10, 30, 40]
    assert [c["chunk_text"] for c in items[0]["relevant_chunks"]] == [
        "T10\nA10", "T30\nA30", "\n"]
    assert [c["doc_id"] for c in items[0]["relevant_chunks"]] == [10, 30, 40]
    assert all(c["doc_id"] in items[0]["relevant_docs"] for c in items[0]["relevant_chunks"])
    assert check_sub(items, top=3, n_queries=2) == []
    assert validate(items, n_queries=None) == []


def test_build_sub_top_k_and_dedup():
    cache = {"1": [[1, 0, 1.0], [1, 0, 0.9], [2, 1, 0.8], [3, 2, 0.7]]}
    texts = {1: ("a", "b"), 2: ("c", "d"), 3: ("e", "f")}
    items = build_sub(cache, texts, top=3)
    assert items[0]["relevant_docs"] == [1, 2, 3]
    assert check_sub(items, top=3, n_queries=1) == []


def test_check_sub_detects_bad_chunk_doc():
    items = [{"id": 1, "relevant_docs": [1, 2],
              "relevant_chunks": [{"doc_id": 1, "chunk_text": "x", "chunk_order": 0},
                                  {"doc_id": 9, "chunk_text": "y", "chunk_order": 1}]}]
    errs = check_sub(items, top=2, n_queries=1)
    assert any("9" in e for e in errs)


def test_export_and_load_texts(tmp_path):
    src = tmp_path / "pubmed_text.jsonl"
    rows = [{"id": 5, "pmid": 50, "title": "T5", "abstract": "A5", "lang": "eng", "year": "2020", "mesh": []},
            {"id": 6, "pmid": 60, "title": "", "abstract": "", "lang": "eng", "year": "2021", "mesh": []},
            {"id": 7, "pmid": 70, "title": "T7", "abstract": "A7", "lang": "eng", "year": "2022", "mesh": []}]
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    outdir = tmp_path / "docs"
    st = export_docs([src], outdir, part_size=1)
    assert st["n_docs"] == 2 and st["skipped_empty"] == 1 and st["n_parts"] == 2
    recs = list(iter_texts(sorted(outdir.glob("docs_pm.part*.jsonl.zst"))))
    assert [r["id"] for r in recs] == [5, 7]
    assert recs[0]["idx"] == 0 and recs[1]["idx"] == 1
    assert recs[0]["text"] == "T5\nA5"
    texts = load_texts([str(src)])
    assert texts[5] == ("T5", "A5") and texts[6] == ("", "")


def test_scores_at_and_compare():
    pm = {"1": [[9, 0, 3.0], [8, 1, 2.0], [7, 2, 1.0]],
          "2": [[6, 3, -1.0], [5, 4, -2.0], [4, 5, -3.0]]}
    pool = {"1": [[3, 0, 1.5], [2, 1, 0.5], [1, 2, -0.5]],
            "2": [[3, 0, 0.2], [2, 1, 0.1], [1, 2, 0.0]]}
    assert scores_at(pm["1"], 1) == (3.0, 3.0)
    assert scores_at(pm["1"], 3) == (1.0, 2.0)
    assert scores_at(pm["2"][:2], 3)[0] is None
    res = compare(pm, pool)
    assert res["queries_both"] == 2
    assert res["wins"]["pm_rank1_gt_pool_rank1"] == 0.5  # câu 1 pm 3.0 > 1.5; câu 2 -1.0 < 0.2
    assert res["wins"]["pm_rank1_gt_pool_top1"] == 0.5
    assert res["pm"]["rank1"]["mean"] == 1.0
