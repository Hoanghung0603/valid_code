import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from full8b_sub import base_docs, full8b_sub, mean_inter  # noqa: E402


def test_base_docs_small_n_uses_ens():
    ens = [[1, 0, 3.0], [2, 0, 2.0], [3, 0, 1.0]]
    pool = [[9, 0, 5.0], [1, 0, 4.0]]
    assert base_docs(pool, ens, 2, top_ens=3) == [1, 2]


def test_base_docs_large_n_fills_from_pool_dedup_and_cuts():
    ens = [[1, 0, 3.0], [2, 0, 2.0], [3, 0, 1.0]]
    pool = [[2, 0, 9.0], [7, 0, 8.0], [1, 0, 7.0], [8, 0, 6.0], [9, 0, 5.0]]
    assert base_docs(pool, ens, 5, top_ens=2) == [1, 2, 7, 8, 9]
    assert base_docs(pool, ens, 4, top_ens=2) == [1, 2, 7, 8]


def test_full8b_sub_adaptive_n_and_chunk_docs():
    pool = {"1": [[5, 0, 0.0], [6, 0, -1.0], [7, 0, -3.0], [8, 0, None]],
            "2": [[5, 0, -9.0]]}
    ens = {"1": [[6, 0, 1.0], [9, 0, 0.5], [5, 0, 0.1]], "2": [[4, 0, 1.0], [3, 0, 0.0]]}
    chunks = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 9, "text": "a"},
                                                                  {"doc_id": 11, "text": "b"}]},
              {"id": 2, "relevant_docs": [], "relevant_chunks": [{"doc_id": 3, "text": "c"}]}]
    out, ns = full8b_sub(pool, ens, chunks, thr=-2.6, lo=1, hi=10, top_ens=100)
    assert ns == [2, 1]
    assert out[0]["relevant_docs"] == [6, 9, 11]
    assert out[1]["relevant_docs"] == [4, 3]
    assert out[0]["relevant_chunks"] is chunks[0]["relevant_chunks"]
    # n_q > top_ens: ens cắt ở top_ens rồi bù pool
    out, ns = full8b_sub(pool, ens, chunks, thr=-2.6, lo=3, hi=10, top_ens=1)
    assert ns == [3, 3]
    assert out[0]["relevant_docs"][:3] == [6, 5, 7]


def test_mean_inter():
    sub = [{"id": 1, "relevant_docs": [1, 2, 3]}, {"id": 2, "relevant_docs": [4]}]
    ref = [{"id": 1, "relevant_docs": [2, 3, 9]}, {"id": 2, "relevant_docs": [5]}]
    assert mean_inter(sub, ref) == 1.0
