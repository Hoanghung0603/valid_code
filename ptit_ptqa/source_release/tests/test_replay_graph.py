from src.retrieval.replay_graph import build_replay_graph, query_anchors


def test_query_anchor_fallback_uses_chunk_evidence():
    rows = [{"doc_id": 7, "url": "https://example.org/7", "chunk_ref": 11,
             "doc_rank": 2, "chunk_rank": 1, "chunk_text": "Răng chảy máu khi thức dậy."},
            {"doc_id": 8, "url": "https://example.org/8", "chunk_ref": 12,
             "doc_rank": 3, "chunk_rank": 2, "chunk_text": "Chảy máu chân răng cần khám."}]
    graph = build_replay_graph(2, "Răng chảy máu là sao?", rows)
    assert graph["mode"] == "replay_evidence"
    assert graph["edge_basis"] == "normalized_chunk_overlap"
    assert {edge["concept_id"] for doc in graph["documents"] for edge in doc["evidence"]}.issubset(
        {concept["id"] for concept in graph["concepts"]})
    assert graph["documents"][0]["evidence"][0]["chunk_ref"] == 11


def test_direct_replay_path_when_no_literal_overlap():
    rows = [{"doc_id": 7, "url": None, "chunk_ref": 11,
             "doc_rank": 1, "chunk_rank": 1, "chunk_text": "Unrelated snippet"}]
    graph = build_replay_graph(3, "Beta-HCG dưới 0.10?", rows)
    assert graph["edge_basis"] == "replay_rank"
    assert graph["concepts"][0]["id"] == "replay_path"
    assert graph["documents"][0]["evidence"][0]["source"] == "replay_rank"
    assert query_anchors("Em hỏi bác sĩ về răng")
