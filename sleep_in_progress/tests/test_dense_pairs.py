import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dense_pairs  # noqa: E402
from dense_pairs import (  # noqa: E402
    build_hits,
    doc_row_map,
    map_pairs,
    pair_cosines,
    score_query_set,
)


def test_doc_row_map():
    ids = np.array([10, 20, 30], dtype=np.int64)
    assert doc_row_map(ids) == {10: 0, 20: 1, 30: 2}


def test_pair_cosines_dot_of_unit_vectors():
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=np.float16)
    q = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float16)
    sc = pair_cosines(embs, q, [0, 1, 2])
    assert sc.shape == (3,)
    assert float(sc[0]) == pytest.approx(0.707, abs=1e-2)
    assert float(sc[1]) == pytest.approx(0.707, abs=1e-2)
    assert float(sc[2]) == pytest.approx(1.0, abs=1e-2)
    assert pair_cosines(embs, q, []).shape == (0,)


def test_build_hits_rounds_and_keeps_order():
    sc = np.array([0.123456, -0.5], dtype=np.float32)
    out = build_hits([[7, 1], [8, 2]], sc)
    assert out == [[7, 1, 0.1235], [8, 2, -0.5]]


def test_map_pairs_missing_embedding_counted():
    pairs = {"1": [[5, 0], [6, 1]], "2": [[9, 2]]}
    mapped, missing = map_pairs(pairs, {5: 0, 9: 1})
    assert mapped == {"1": [(5, 0, 0)], "2": [(9, 2, 1)]}
    assert missing == 1


def test_score_query_set_matches_rows():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    qe = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    mapped = {"7": [(10, 0, 0), (11, 1, 1)], "8": [(11, 1, 1)]}
    out = score_query_set(embs, qe, mapped, [7, 8])
    assert out["7"] == [[10, 0, 1.0], [11, 1, 0.0]]
    assert out["8"] == [[11, 1, 1.0]]


def _write_blk(dirp: Path):
    ids = np.array([10, 11, 12], dtype=np.int64)
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]], dtype=np.float16)
    np.savez(dirp / "blk_00000.npz", ids=ids, emb=emb)


def test_main_two_query_sets(tmp_path, monkeypatch):
    d = tmp_path
    (d / "emb").mkdir()
    _write_blk(d / "emb")
    (d / "pairs.json").write_text(json.dumps({"1": [[10, 0], [11, 1]], "2": [[12, 2], [99, 3]]}),
                                  encoding="utf-8")
    (d / "q.json").write_text(json.dumps({"id": 1, "query": "a"}) + "\n" +
                              json.dumps({"id": 2, "query": "b"}) + "\n", encoding="utf-8")
    (d / "qh.json").write_text(json.dumps({"id": 1, "query": "ha"}) + "\n" +
                               json.dumps({"id": 2, "query": "hb"}) + "\n", encoding="utf-8")
    fake = {
        ("a", 256): np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        ("ha", 512): np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)],
                               [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=np.float16),
    }
    calls = []

    def fake_embed(model, texts, max_len, batch):
        calls.append((tuple(texts), max_len))
        return fake[(texts[0], max_len)]

    monkeypatch.setattr(dense_pairs, "embed_queries", fake_embed)
    old = sys.argv
    sys.argv = ["dense_pairs.py", "--emb", str(d / "emb"), "--pairs", str(d / "pairs.json"),
                "--queries", str(d / "q.json"), "--query-len", "256",
                "--queries2", str(d / "qh.json"), "--query-len2", "512",
                "--out", str(d / "out.json"), "--out2", str(d / "out2.json"),
                "--report", str(d / "rep.json")]
    try:
        dense_pairs.main()
    finally:
        sys.argv = old
    out = json.loads((d / "out.json").read_text(encoding="utf-8"))
    out2 = json.loads((d / "out2.json").read_text(encoding="utf-8"))
    assert out["1"] == [[10, 0, 1.0], [11, 1, 0.0]]
    assert out["2"][0][:2] == [12, 2]
    assert out["2"][0][2] == pytest.approx(0.707, abs=1e-3)
    assert [h[2] for h in out2["1"]] == pytest.approx([0.707, 0.707], abs=1e-3)
    assert out2["2"][0][2] == pytest.approx(0.0, abs=1e-3)
    assert len(calls) == 2
    rep = json.loads((d / "rep.json").read_text(encoding="utf-8"))
    assert rep["cap"] == 3
    assert rep["doc_thieu_emb"] == 1
    assert [b["query_len"] for b in rep["bo_truy_van"]] == [256, 512]


def test_main_requires_out2_with_queries2(tmp_path):
    d = tmp_path
    (d / "emb").mkdir()
    _write_blk(d / "emb")
    (d / "pairs.json").write_text(json.dumps({"1": [[10, 0]]}), encoding="utf-8")
    (d / "q.json").write_text(json.dumps({"id": 1, "query": "a"}) + "\n", encoding="utf-8")
    old = sys.argv
    sys.argv = ["dense_pairs.py", "--emb", str(d / "emb"), "--pairs", str(d / "pairs.json"),
                "--queries", str(d / "q.json"), "--out", str(d / "o.json"),
                "--queries2", str(d / "q.json")]
    try:
        with pytest.raises(SystemExit) as e:
            dense_pairs.main()
    finally:
        sys.argv = old
    assert e.value.code == 2
