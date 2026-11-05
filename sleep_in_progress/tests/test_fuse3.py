import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from full8b_sub import full8b_sub  # noqa: E402
from fuse3 import fuse3, fuse_q  # noqa: E402


def test_fuse_q_rrf_scores_and_missing_sources():
    pool = [[1, 10, 5.0], [2, 20, 4.0], [3, 30, 3.0]]
    q8b = [[3, 30, 9.0], [1, 10, 8.0]]          # doc 2 không có trong 8B
    dense = [[3, 30, 0.9], [9, 90, 0.8]]          # doc 9 ngoài pool: bỏ
    out = fuse_q(pool, q8b, dense, (1, 1, 1), k=60)
    assert [r[0] for r in out] == [3, 1, 2]
    s3 = 1 / 63 + 1 / 61 + 1 / 61
    assert abs(out[0][2] - round(s3, 8)) < 1e-12
    assert abs(out[2][2] - round(1 / 62, 8)) < 1e-12
    assert out[0][1] == 30


def test_fuse_q_dense_weight_and_tie_and_idx_fallback():
    pool = [[1, None, 5.0], [2, 20, 4.0]]
    dense = [[2, 20, 0.9], [1, 11, 0.8]]
    # w dense 0, không có 8B: thứ tự bge
    assert [r[0] for r in fuse_q(pool, [], dense, (1, 0, 0))] == [1, 2]
    # đồng điểm (bge 1/61+dense 1/62 = bge 1/62+dense 1/61): giữ thứ tự pool
    assert [r[0] for r in fuse_q(pool, [], dense, (1, 0, 1))] == [1, 2]
    # dense trọng số 2: doc 2 lên đầu
    out = fuse_q(pool, [], dense, (1, 0, 2))
    assert [r[0] for r in out] == [2, 1]
    assert out[1][1] == 11


def test_fuse_q_bm25_source_optional():
    pool = [[1, 0, 5.0], [2, 0, 4.0]]
    bm = [[2, 0, 30.0], [1, 0, 20.0]]
    # 3 trọng số hoặc w_bm25 = 0: giống hệt không có BM25
    base = fuse_q(pool, [], [], (1, 0, 0))
    assert fuse_q(pool, [], [], (1, 0, 0), bm25_hits=bm) == base
    assert fuse_q(pool, [], [], (1, 0, 0, 0), bm25_hits=bm) == base
    # BM25 trọng số 2: doc 2 lên đầu (1/62 + 2/61 > 1/61 + 2/62)
    out = fuse_q(pool, [], [], (1, 0, 0, 2), bm25_hits=bm)
    assert [r[0] for r in out] == [2, 1]
    f = fuse3({"1": pool}, {}, {}, (1, 0, 0, 2), bm25={1: bm})
    assert [r[0] for r in f["1"]] == [2, 1]


def test_fuse_q_extra_sources():
    pool = [[1, 0, 5.0], [2, 0, 4.0], [3, 0, 3.0]]
    base = fuse_q(pool, [], [], (1, 0, 0))
    assert fuse_q(pool, [], [], (1, 0, 0), extra=[([[3, 0, 1.0]], 0.0)]) == base
    out = fuse_q(pool, [], [], (1, 0, 0), extra=[([[3, 0, 1.0]], 5), ([[3, 0, 1.0], [2, 0, 1.0]], 1)])
    # doc 3: 1/63 + 5/61 + 1/61; doc 2: 1/62 + 1/62; doc 1: 1/61
    assert [r[0] for r in out] == [3, 2, 1]
    f = fuse3({"1": pool}, {}, {}, (1, 0, 0), extras=[({1: [[2, 0, 1.0]]}, 3)])
    assert [r[0] for r in f["1"]][0] == 2


def test_fuse3_keys_and_full_mode_via_full8b_sub():
    pool = {"1": [[1, 0, 0.0], [2, 0, -1.0], [3, 0, -9.0]]}
    fused = fuse3(pool, {"1": [[3, 0, 1.0]]}, {1: [[3, 0, 0.9], [2, 0, 0.5]]}, (1, 1, 1))
    assert [r[0] for r in fused["1"]] == [3, 2, 1]
    chunks = [{"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 7, "text": "x"}]}]
    out, ns = full8b_sub(pool, fused, chunks, thr=-2.6, lo=1, hi=10, top_ens=10**9)
    assert ns == [2]
    assert out[0]["relevant_docs"] == [3, 2, 7]
