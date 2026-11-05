import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from chunk_len_stats import (  # noqa: E402
    chunk_len, collect, estimate, histogram, length_stats, load_submission,
    percentile, read_docs_jsonl, syl_span, window_hit,
)


def test_chunk_len_normalized_ws():
    assert chunk_len("  Hello   WORLD ") == 2
    assert chunk_len("m\u1ed9t hai\nba") == 3
    assert chunk_len("   ") == 0


def test_percentile_nearest_rank():
    v = list(range(1, 11))
    assert percentile(v, 5) == 1
    assert percentile(v, 25) == 3
    assert percentile(v, 50) == 5
    assert percentile(v, 75) == 8
    assert percentile(v, 95) == 10
    assert percentile([], 50) == 0


def test_length_stats_ratios():
    lengths = [100, 200, 204, 300, 901, 1025, 1281]
    st = length_stats(lengths)
    assert st["count"] == 7
    assert st["p50"] == 300 and st["max"] == 1281
    assert st["ratio_gt_1280"] == round(1 / 7, 4)
    assert st["ratio_gt_1024"] == round(2 / 7, 4)
    assert st["ratio_gt_900"] == round(3 / 7, 4)
    assert st["ratio_lt_205"] == round(3 / 7, 4)
    assert length_stats([])["count"] == 0


def test_histogram_bin100():
    h = histogram([50, 150, 199, 200, 1280])
    assert h == {"0-99": 1, "100-199": 2, "200-299": 1, "1200-1299": 1}
    assert histogram([]) == {}


def test_syl_span():
    text = "aa bb cc dd"
    starts = [0, 3, 6, 9]
    assert syl_span(starts, 0, 11) == (0, 4)
    assert syl_span(starts, 3, 6) == (1, 2)
    assert syl_span(starts, 0, 1) == (0, 1)  # it nhat 1 am ti


def test_window_hit_boundary_04():
    # cua so 512 am ti, buoc 462; nguong 0.4
    assert window_hit(0, 205, 2000) == (True, True)   # 205/512 >= 0.4
    assert window_hit(0, 204, 2000) == (False, False)  # 204/512 < 0.4
    assert window_hit(0, 512, 2000) == (True, True)
    assert window_hit(0, 1280, 2000) == (True, True)   # 512/1280 = 0.4 dung bien
    assert window_hit(0, 1281, 2000) == (True, False)  # 512/1281 < 0.4
    assert window_hit(1000, 1100, 2000) == (False, False)
    assert window_hit(400, 1000, 2000) == (True, True)  # giao max 512 voi cua so [462,974)
    assert window_hit(0, 1, 2000) == (False, False)


def test_estimate_old_vs_new():
    text = " ".join(f"s{i}" for i in range(2000))
    def seg(a, b):
        return " ".join(f"s{i}" for i in range(a, b))
    chunks = [
        seg(0, 205),          # 205 am ti: cu + moi deu kha nang khop
        seg(0, 204),          # 204 am ti: truot ca hai
        seg(0, 1280),         # 512/1280 = 0.4 bien: cu + moi
        seg(0, 1281),         # 512/1281 < 0.4: chi cu
        "khong co trong text doc",  # khong tim thay offset
        "   ",                # chunk rong
    ]
    sub = [{"id": 1, "relevant_docs": [10],
            "relevant_chunks": [{"doc_id": 10, "chunk_text": t} for t in chunks]}]
    res = estimate(sub, {"10": text}, win=512, step=462, threshold=0.4)
    assert res["chunks_total"] == 6
    assert res["chunks_empty"] == 1
    assert res["chunks_not_found_in_text"] == 1
    assert res["chunks_doc_missing"] == 0
    assert res["chunks_found"] == 4
    assert res["maybe_match_old"] == 3
    assert res["maybe_match_new"] == 2
    assert res["maybe_match_old_ratio"] == 0.75
    assert res["maybe_match_new_ratio"] == 0.5
    assert res["maybe_match_new_ratio_all"] == round(2 / 6, 4)


def test_estimate_multi_text_variants():
    # cung doc co 2 bien the text (export khac nhau); chi bien the 2 chua chunk
    text2 = " ".join(f"s{i}" for i in range(2000))
    text1 = "khac hon hoan toan " * 50
    sub = [{"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "s3 s4"}]}]
    docs = {"10": [text1, text2]}
    res = estimate(sub, docs, win=512, step=462, threshold=0.4)
    assert res["chunks_found"] == 1
    assert res["maybe_match_old"] == 0 and res["maybe_match_new"] == 0  # 2/512 < 0.4


def test_estimate_doc_missing():
    sub = [{"id": 1, "relevant_docs": [9], "relevant_chunks": [{"doc_id": 9, "chunk_text": "x y z"}]}]
    res = estimate(sub, {})
    assert res["chunks_doc_missing"] == 1 and res["chunks_found"] == 0


def test_collect_grouping():
    sub = [{"id": 1, "relevant_docs": [10],
            "relevant_chunks": [{"doc_id": 10, "chunk_text": "a b"}, {"doc_id": 20, "chunk_text": "c d e"}]}]
    lengths, tokens_q, by_doc = collect(sub, "ws")
    assert lengths == [2, 3] and tokens_q == [5]
    assert set(by_doc) == {"10", "20"}
    # chi lay N cau dau khi sample_queries > 0
    sub2 = sub + [{"id": 2, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "f"}]}]
    lengths2, _, _ = collect(sub2, "ws", sample_queries=1)
    assert lengths2 == [2, 3]


def test_read_docs_jsonl_and_zip(tmp_path):
    p = tmp_path / "docs.jsonl"
    p.write_text(json.dumps({"id": 1, "text": "x"}) + "\n" + json.dumps({"id": 2, "text": "y"}) + "\n",
                 encoding="utf-8")
    docs = read_docs_jsonl([p], wanted={"1"})
    assert docs == {"1": ["x"]}
    assert read_docs_jsonl([p]) == {"1": ["x"], "2": ["y"]}

    sub = [{"id": 1, "relevant_docs": [1], "relevant_chunks": [{"doc_id": 1, "chunk_text": "x"}]}]
    zp = tmp_path / "sub.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("submission.json", json.dumps(sub))
    assert load_submission(str(zp)) == sub
    jp = tmp_path / "sub.json"
    jp.write_text(json.dumps(sub), encoding="utf-8")
    assert load_submission(str(jp)) == sub
