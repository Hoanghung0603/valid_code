import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zhnew_dense import (  # noqa: E402
    fallback_queries,
    load_emb_blocks,
    load_queries,
    load_zh_ids,
    rrf_fuse,
)


def test_load_queries_prefers_nonempty_zh(tmp_path):
    p = tmp_path / "zh.jsonl"
    recs = [
        {"id": 1, "query": "vi 1", "zh": ""},
        {"id": 1, "query": "vi 1", "zh": "中文一"},
        {"id": 2, "query": "vi 2", "zh": ""},
        {"id": 2, "query": "vi 2", "zh": ""},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
    m = load_queries(str(p), field="zh")
    assert m == {1: "中文一"}
    m2 = load_queries(str(p), field="query")
    assert m2 == {1: "vi 1", 2: "vi 2"}


def test_fallback_queries():
    vi = {1: "a", 2: "b", 3: "c"}
    zh = {1: "甲", 3: ""}
    merged, fell = fallback_queries(vi, zh)
    assert merged == {1: "甲", 2: "b", 3: "c"}
    assert fell == [2, 3]


def test_rrf_fuse_scores_and_order():
    a = [[10, -1, 0.9], [11, -1, 0.8], [12, -1, 0.7]]
    b = [[12, -1, 0.9], [10, -1, 0.5], [13, -1, 0.4]]
    rows = rrf_fuse(a, b, k=60)
    sc = {r[0]: r[2] for r in rows}
    assert abs(sc[10] - (1 / 61 + 1 / 62)) < 1e-6
    assert abs(sc[12] - (1 / 63 + 1 / 61)) < 1e-6
    assert abs(sc[11] - 1 / 62) < 1e-6
    assert abs(sc[13] - 1 / 63) < 1e-6
    assert [r[0] for r in rows] == [10, 12, 11, 13]
    assert all(r[1] == -1 for r in rows)


def test_rrf_fuse_stable_on_tie():
    a = [[1, -1, 1.0], [2, -1, 1.0]]
    b = [[2, -1, 1.0], [1, -1, 1.0]]
    rows = rrf_fuse(a, b, k=60)
    assert rows[0][0] == 1  # đồng điểm, doc của nhánh VI đứng trước
    assert len(rows) == 2


def test_load_emb_blocks_filters_allowed(tmp_path):
    np.savez(tmp_path / "blk_00000.npz", ids=np.array([5, 6, 7], np.int64),
             emb=np.zeros((3, 4), np.float16))
    np.savez(tmp_path / "blk_00001.npz", ids=np.array([7, 8], np.int64),
             emb=np.ones((2, 4), np.float16))
    ids, emb, n_total = load_emb_blocks(str(tmp_path), {6, 7, 8})
    assert n_total == 5
    assert sorted(int(x) for x in ids) == [6, 7, 7, 8]
    assert emb.shape == (4, 4)
    ids2, emb2, _ = load_emb_blocks(str(tmp_path), {99})
    assert len(ids2) == 0 and emb2.shape[0] == 0


def test_load_zh_ids_variants(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text("[3, 1, 2]", encoding="utf-8")
    assert load_zh_ids(str(p)) == [1, 2, 3]
    np.savez(tmp_path / "ids.npz", ids=np.array([9, 8], np.int64))
    assert load_zh_ids(str(tmp_path / "ids.npz")) == [8, 9]
    p.write_text('{"ids": [5, 4]}', encoding="utf-8")
    assert load_zh_ids(str(p)) == [4, 5]
