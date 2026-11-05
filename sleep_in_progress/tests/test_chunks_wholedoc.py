import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunks_wholedoc import build, cut, order_docs, pick_q, verify_verbatim  # noqa: E402

DOCS = {1: "a " * 3000, 2: "b " * 1500, 3: "c " * 5000, 4: "", 5: "a " * 3000}


def test_cut_is_verbatim_prefix_on_space():
    t = "xin chào bạn nhé"
    assert cut(t, 100) == t
    c = cut(t, 11)
    assert t.startswith(c) and c == "xin chào"


def test_order_rr_keeps_cands_order():
    assert order_docs([[3, 0], [1, 1], [2, 2]], None) == [3, 1, 2]


def test_order_dw_by_best_window_then_cands():
    wins = {"1": [[0, 5, -1.0], [5, 9, 2.0]], "2": [[0, 5, 0.5]]}
    # doc 3 không có cửa sổ -> xuống cuối
    assert order_docs([[3, 0], [2, 1], [1, 2]], wins) == [1, 2, 3]


def test_pick_budget_truncates_last_and_skips_empty_and_dup():
    out = pick_q([4, 1, 5, 2, 3], DOCS, budget=10000, doc_cap=100000, min_chars=2000)
    # 4 rỗng bỏ; 1 (6000); 5 trùng text với 1 bỏ; 2 (3000) -> 9000; 3 còn 1000 < 2000 -> dừng
    assert [c["doc_id"] for c in out] == [1, 2]
    out2 = pick_q([1, 3], DOCS, budget=10000, doc_cap=100000, min_chars=2000)
    assert [c["doc_id"] for c in out2] == [1, 3]
    assert len(out2[1]["chunk_text"]) <= 4000
    assert DOCS[3].startswith(out2[1]["chunk_text"])


def test_doc_cap():
    out = pick_q([3], DOCS, budget=100000, doc_cap=1000)
    assert len(out[0]["chunk_text"]) <= 1000


def test_build_verbatim():
    sub = build({"7": [[1, 0], [2, 1]]}, DOCS, budget=100000, doc_cap=100000)
    assert sub[0]["id"] == 7 and sub[0]["relevant_docs"] == []
    assert verify_verbatim(sub, DOCS) == 0
