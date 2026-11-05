from scripts.build_exp144_docs_candidates import append_owner_floor, fuse_current_and_dense


def test_fuse_current_and_dense_preserves_consensus_and_admits_novel_docs() -> None:
    fused = fuse_current_and_dense([1, 2, 3, 6], [4, 1, 5, 2], top_k=4)
    assert fused[0] == 1
    assert 4 in fused
    assert len(fused) == len(set(fused)) == 4


def test_append_owner_floor_is_stable_and_unique() -> None:
    row = {"relevant_chunks": [{"doc_id": 3}, {"doc_id": 7}, {"doc_id": 7}, {"doc_id": 8}]}
    assert append_owner_floor([1, 3], row) == [1, 3, 7, 8]
