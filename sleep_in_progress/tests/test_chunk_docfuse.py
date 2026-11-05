import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunk_docfuse import chunk_docfuse, compare, rescore_q, zscores  # noqa: E402


def test_zscores_constant_is_zero():
    assert zscores([2.0, 2.0]) == [0.0, 0.0]
    z = zscores([1.0, 3.0])
    assert z == [-1.0, 1.0]


def test_rescore_doc_score_lifts_chunk_and_missing_doc_gets_min():
    # chunk logit gần nhau (A 1.0, C 0.95, B 0.9) + 1 chunk thấp D; doc 2 có điểm fused vượt trội
    rows = [[1, 1, 0, 1.0, "A"], [9, 3, 0, 0.95, "C"], [2, 2, 0, 0.9, "B"], [3, 4, 0, -5.0, "D"]]
    fused = [[2, 0, 1.0], [1, 0, 0.0], [3, 0, 0.0], [4, 0, 0.0]]
    res = rescore_q(rows, fused)
    assert [r[4] for _, r in res] == ["B", "A", "C", "D"]
    # doc 9 ngoài fused nhận zd = min (= zd của doc 1) nên C đứng ngay sau A
    s = {r[4]: sc for sc, r in res}
    zc = {"A": s["A"], "C": s["C"]}
    assert zc["A"] > zc["C"]


def test_tie_keeps_old_order_and_dedup_and_k_from_ref():
    rows = [[1, 1, 0, 1.0, "X"], [1, 1, 1, 1.0, "X"], [1, 1, 2, 1.0, "Y"], [1, 1, 3, 1.0, "Z"]]
    fused = {"5": [[1, 0, 0.1]]}
    ref = [{"id": 5, "relevant_docs": [], "relevant_chunks": [{"doc_id": 1, "chunk_text": "Q"}] * 2}]
    out = chunk_docfuse({"5": rows}, fused, ref)
    assert [c["chunk_text"] for c in out[0]["relevant_chunks"]] == ["X", "Y"]
    st = compare(out, ref, fused)
    assert st["chunk/câu mới"] == 2 and st["overlap chunk TB"] == 0
    assert st["hạng rrf doc nguồn mới"]["TB"] == 1


def test_wd_weight_changes_order_and_default_is_one():
    from chunk_docfuse import identical_diff, pick_q
    rows = [[1, 1, 0, 1.0, "X"], [2, 2, 0, 0.0, "Y"]]
    fused = [[2, 0, 1.0], [1, 0, 0.0]]
    order = lambda wd: [c["chunk_text"] for c in pick_q(rows, fused, 2, wd)]  # noqa: E731
    assert order(1.0) == ["X", "Y"]          # hòa s=0: giữ thứ tự cũ
    assert order(2.0) == ["Y", "X"]
    assert order(0.5) == ["X", "Y"]
    assert [c["chunk_text"] for c in pick_q(rows, fused, 2)] == order(1.0)
    assert [s for s, _ in rescore_q(rows, fused, 3.0)] == [2.0, -2.0]
    ref = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 9, "chunk_text": "q"}] * 2}]
    a = chunk_docfuse({"1": rows}, {"1": fused}, ref, wd=2.0)
    b = chunk_docfuse({"1": rows}, {"1": fused}, ref)
    assert a[0]["relevant_chunks"][0]["chunk_text"] == "Y"
    assert identical_diff(a, b) == 1 and identical_diff(b, b) == 0
