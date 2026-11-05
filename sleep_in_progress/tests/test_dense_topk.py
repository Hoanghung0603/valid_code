import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dense_topk import format_hits, load_embeddings, load_idx_map, topk_scan


def write_blk(dir_path: Path, name: str, ids, emb):
    np.savez(dir_path / name, ids=np.asarray(ids, dtype=np.int64), emb=np.asarray(emb, dtype=np.float16))


def test_load_embeddings_concat_dirs_and_dup_error(tmp_path):
    a, b = tmp_path / "emb", tmp_path / "emb_all"
    a.mkdir()
    b.mkdir()
    write_blk(a, "blk_00000.npz", [1, 2], [[1.0, 0.0], [0.0, 1.0]])
    write_blk(b, "blk_00001.npz", [3], [[1.0, 1.0]])
    ids, embs = load_embeddings([a, b])
    assert list(ids) == [1, 2, 3]
    assert embs.shape == (3, 2)
    assert embs.dtype == np.float16
    write_blk(b, "blk_00002.npz", [2], [[0.5, 0.5]])
    with pytest.raises(SystemExit):
        load_embeddings([a, b])
    with pytest.raises(SystemExit):
        load_embeddings([tmp_path / "missing"])


def test_load_idx_map_from_docs_and_json(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    p = tmp_path / "d.jsonl.zst"
    rows = [{"id": 11, "idx": 0, "text": "aa"}, {"id": 22, "idx": 5, "text": "bb"}]
    with open(p, "wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw) as w:
            for r in rows:
                w.write((json.dumps(r) + "\n").encode("utf-8"))
    (tmp_path / "idx.json").write_text(json.dumps({"33": 9}), encoding="utf-8")
    m = load_idx_map([p], [tmp_path / "idx.json"])
    assert m == {11: 0, 22: 5, 33: 9}
    assert load_idx_map() == {}
    assert load_idx_map(idx_from=[tmp_path / "idx.json"]) == {33: 9}


def test_read_jsonl_zst_roundtrip(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    from dense_topk import read_jsonl_zst
    p = tmp_path / "d.jsonl.zst"
    rows = [{"id": 1, "idx": 2, "text": "x"}, {"id": 3, "idx": 4, "text": "y"}]
    with open(p, "wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw) as w:
            for r in rows:
                w.write((json.dumps(r) + "\n").encode("utf-8"))
    assert list(read_jsonl_zst(p)) == rows


def test_topk_scan_matches_full_scan_across_chunks():
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]], dtype=np.float16)
    qe = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    best_s, best_i = topk_scan(embs, qe, top=2, chunk=3)
    assert list(best_i[0]) == [0, 2]
    assert list(best_i[1]) == [1, 2]
    assert best_s[0][0] == pytest.approx(1.0, abs=1e-3)
    assert best_s[1][0] == pytest.approx(1.0, abs=1e-3)
    assert best_s[0][1] == pytest.approx(0.6, abs=1e-2)


def test_topk_scan_top_larger_than_corpus_and_progress():
    embs = np.array([[0.5, 0.5]], dtype=np.float16)
    qe = np.array([[1.0, 0.0]], dtype=np.float16)
    seen = []
    best_s, best_i = topk_scan(embs, qe, top=10, chunk=1, progress=lambda s, n: seen.append((s, n)))
    assert best_s.shape == (1, 1) and list(best_i[0]) == [0]
    assert seen == [(1, 1)]


def test_topk_scan_uses_score_fn():
    embs = np.zeros((4, 2), dtype=np.float16)
    qe = np.zeros((1, 2), dtype=np.float16)

    def fn(e, q):
        sc = np.arange(e.shape[0], dtype=np.float32).reshape(-1, 1) + q.shape[0]
        return np.repeat(sc, q.shape[0], axis=1)

    best_s, best_i = topk_scan(embs, qe, top=2, chunk=3, score_fn=fn)
    assert list(best_i[0]) == [2, 1]
    assert list(best_s[0]) == [3.0, 2.0]


def test_format_hits_sorted_rounded_with_idx():
    ids = np.array([10, 20, 30], dtype=np.int64)
    best_s = np.array([[-0.5, 1.25, 0.12345679]], dtype=np.float32)
    best_i = np.array([[0, 1, 2]], dtype=np.int64)
    out = format_hits(best_s, best_i, ids, {10: 7, 20: 8, 30: 9}, ["5"])
    assert out == {"5": [[20, 8, 1.25], [30, 9, 0.1235], [10, 7, -0.5]]}


def test_format_hits_skips_empty_slots_and_errors_on_missing_idx():
    ids = np.array([10, 20], dtype=np.int64)
    best_s = np.array([[1.0, -np.inf]], dtype=np.float32)
    best_i = np.array([[0, -1]], dtype=np.int64)
    out = format_hits(best_s, best_i, ids, {10: 1}, ["1"])
    assert out == {"1": [[10, 1, 1.0]]}
    with pytest.raises(SystemExit):
        format_hits(best_s, best_i, ids, {}, ["1"])


def test_main_cpu_with_query_emb_and_idx_from(tmp_path, monkeypatch, capsys):
    from dense_topk import main
    a = tmp_path / "emb"
    a.mkdir()
    write_blk(a, "blk_00000.npz", [1, 2, 3], [[1.0, 0.0], [0.0, 1.0], [0.75, 0.75]])
    (tmp_path / "q.jsonl").write_text('{"id": 100, "query": "a"}\n{"id": 200, "query": "b"}\n',
                                      encoding="utf-8")
    (tmp_path / "idx.json").write_text(json.dumps({"1": 11, "2": 22, "3": 33}), encoding="utf-8")
    np.savez(tmp_path / "qe.npz", emb=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16))
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", [
        "dense_topk.py", "--emb", str(a), "--queries", str(tmp_path / "q.jsonl"),
        "--idx-from", str(tmp_path / "idx.json"), "--query-emb", str(tmp_path / "qe.npz"),
        "--out", str(out), "--top", "2", "--chunk", "2", "--device", "cpu",
    ])
    main()
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res == {"100": [[1, 11, 1.0], [3, 33, 0.75]], "200": [[2, 22, 1.0], [3, 33, 0.75]]}
    assert "2 câu x 2" in capsys.readouterr().out
