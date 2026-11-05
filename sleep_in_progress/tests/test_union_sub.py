import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from combine_sub import combine_sub  # noqa: E402
from union_sub import adaptive_n, load_ranked, rerank_order, union_sub  # noqa: E402


def test_adaptive_n_threshold_and_clamp():
    hits = [[1, 0, 2.0], [2, 0, 0.5], [3, 0, -1.0], [4, 0, None], [5, 0, -3.0]]
    assert adaptive_n(hits, -2.0, 0, 10) == 3
    assert adaptive_n(hits, -2.0, 4, 10) == 4
    assert adaptive_n(hits, -2.0, 0, 2) == 2


def test_union_sub_adaptive_cut():
    cache = {"1": [[7, 0, 1.0], [8, 0, -0.5], [9, 0, -5.0]], "2": [[5, 0, -9.0], [6, 0, -9.5]]}
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": []},
                  {"id": 2, "relevant_docs": [], "relevant_chunks": []}]
    out = union_sub(cache, chunks_sub, thr=-1.0, min_n=1, max_n=5)
    assert out[0]["relevant_docs"] == [7, 8]
    assert out[1]["relevant_docs"] == [5]


def test_keeps_rrf_order():
    docs_sub = [{"id": 1, "relevant_docs": [3, 1, 4, 5], "relevant_chunks": []}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 5, "chunk_text": "a"}]}]
    out = union_sub(docs_sub, chunks_sub, docs_n=2, add_k=0)
    assert out[0]["relevant_docs"][:2] == [3, 1]


def test_no_duplicates():
    docs_sub = [{"id": 1, "relevant_docs": [3, 1, 4], "relevant_chunks": []}]
    chunks = [{"doc_id": 4, "chunk_text": "a"}, {"doc_id": 9, "chunk_text": "b"}, {"doc_id": 9, "chunk_text": "c"}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": chunks}]
    out = union_sub(docs_sub, chunks_sub, docs_n=3, add_k=0)
    docs = out[0]["relevant_docs"]
    assert docs == [3, 1, 4, 9]
    assert len(docs) == len({str(d) for d in docs})


def test_chunk_docs_included():
    docs_sub = [{"id": 1, "relevant_docs": [3, 1], "relevant_chunks": []}]
    chunks = [{"doc_id": 9, "chunk_text": "x"}, {"doc_id": 1, "chunk_text": "y"}, {"doc_id": 7, "chunk_text": "z"}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": chunks}]
    out = union_sub(docs_sub, chunks_sub, docs_n=2, add_k=0)
    assert out[0]["relevant_docs"] == [3, 1, 9, 7]
    assert out[0]["relevant_chunks"] == chunks
    assert out[0]["relevant_chunks"] is chunks


def test_k0_equals_v14_behavior():
    docs_sub = [{"id": 1, "relevant_docs": list(range(30)), "relevant_chunks": []}]
    chunks = [{"doc_id": 25, "chunk_text": "a"}, {"doc_id": 99, "chunk_text": "b"}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": chunks}]
    u = union_sub(docs_sub, chunks_sub, docs_n=20, add_k=0)
    truncated = [{"id": 1, "relevant_docs": list(range(20)), "relevant_chunks": []}]
    assert u == combine_sub(truncated, chunks_sub)


def test_rerank_order_logit_desc_ties_by_position():
    cand = [[10, 0, 0.5], [11, 1, 1.0], [12, 2, 0.5], [13, 3, -1.0], [14, 4, None]]
    assert rerank_order(cand) == [11, 10, 12, 13, 14]


def test_add_rerank_top_k_new_docs_in_rerank_order():
    docs_sub = [{"id": 1, "relevant_docs": [1, 2, 3, 4, 5], "relevant_chunks": []}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 3, "chunk_text": "a"}]}]
    rerank_map = {"1": [[5, 0, 0.1], [4, 1, 0.9], [7, 2, 0.5], [8, 3, 0.2]]}
    out = union_sub(docs_sub, chunks_sub, rerank_map, docs_n=3, add_k=2)
    assert out[0]["relevant_docs"] == [1, 2, 3, 4, 7]


def test_add_rerank_skips_existing_docs():
    docs_sub = [{"id": 1, "relevant_docs": [1, 2, 3, 4], "relevant_chunks": []}]
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": []}]
    rerank_map = {"1": [[4, 0, 0.9], [3, 1, 0.8], [9, 2, 0.7], [8, 3, 0.6]]}
    out = union_sub(docs_sub, chunks_sub, rerank_map, docs_n=4, add_k=3)
    assert out[0]["relevant_docs"] == [1, 2, 3, 4, 9]


def test_load_ranked_from_cache_format():
    cache = {"2": [[5, 50, 0.9], [6, 60, 0.1]], "1": [7, [8, 80, 0.5]]}
    out = load_ranked(cache)
    assert out == [(2, [5, 6]), (1, [7, 8])]


def test_load_ranked_from_submission_format():
    sub = [{"id": 3, "relevant_docs": [1, 2], "relevant_chunks": []}]
    assert load_ranked(sub) == [(3, [1, 2])]


def test_docs_from_cache_format():
    cache = {"1": [[3, 30, 0.9], [1, 10, 0.5], [4, 40, 0.1]]}
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 4, "chunk_text": "a"}]}]
    out = union_sub(cache, chunks_sub, docs_n=2, add_k=0)
    assert out[0]["id"] == 1
    assert out[0]["relevant_docs"] == [3, 1, 4]
    assert out[0]["relevant_chunks"] == chunks_sub[0]["relevant_chunks"]


def test_docs_from_cache_with_add_rerank():
    cache = {"1": [[1, 10, 0.5], [2, 20, 0.4]]}
    chunks_sub = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 9, "chunk_text": "x"}]}]
    rerank_map = {"1": [[2, 20, 0.2], [5, 50, 0.8], [9, 90, 0.6]]}
    out = union_sub(cache, chunks_sub, rerank_map, docs_n=2, add_k=2)
    assert out[0]["relevant_docs"] == [1, 2, 5, 9]
