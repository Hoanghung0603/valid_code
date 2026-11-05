import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunks_from_rerank import build, pick  # noqa: E402

ROWS = [[1, 1, 0, 3.0, "a"], [2, 2, 0, 1.0, "b"], [1, 1, 0, 0.9, "a"], [3, 5, 1, -0.5, "c"], [4, 7, 0, -2.0, "d"]]


def test_pick_fixed_k_dedups_text():
    assert [c["chunk_text"] for c in pick(ROWS, k=3)] == ["a", "b", "c"]


def test_pick_threshold_and_clamp():
    assert [c["chunk_text"] for c in pick(ROWS, thr=0.0)] == ["a", "b"]
    assert len(pick(ROWS, thr=0.0, min_k=3)) == 3
    assert len(pick(ROWS, thr=-5.0, max_k=2)) == 2


def test_spread_takes_best_chunk_of_each_doc_first():
    rows = [[1, 1, 0, 3.0, "a0"], [1, 1, 1, 2.5, "a1"], [2, 2, 0, 2.0, "b0"], [1, 1, 2, 1.5, "a2"], [3, 3, 0, -1.0, "c0"]]
    assert [c["chunk_text"] for c in pick(rows, k=4, spread=True)] == ["a0", "b0", "c0", "a1"]
    # thr: n = số chunk logit > 0 (= 4), nhưng chunk được giữ là theo thứ tự rải
    assert [c["chunk_text"] for c in pick(rows, thr=0.0, spread=True)] == ["a0", "b0", "c0", "a1"]
    assert [c["chunk_text"] for c in pick(rows, thr=0.0)] == ["a0", "a1", "b0", "a2"]


def test_build_format():
    out = build({"7": ROWS}, k=2)
    assert out == [{"id": 7, "relevant_docs": [],
                    "relevant_chunks": [{"doc_id": 1, "chunk_text": "a"}, {"doc_id": 2, "chunk_text": "b"}]}]
