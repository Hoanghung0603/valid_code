import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rerank_run import (  # noqa: E402
    build_pairs,
    merge_parts,
    merge_scores,
    part_file,
    plan_parts,
    split_parts,
)


def test_split_parts_sizes():
    assert split_parts(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
    assert split_parts([], 3) == []
    assert split_parts([1, 2, 3], 3) == [[1, 2, 3]]
    assert split_parts([1, 2, 3, 4], 3) == [[1, 2, 3], [4]]


def test_plan_parts_marks_existing_files_as_done(tmp_path):
    qids = [f"q{i}" for i in range(5)]
    plan = plan_parts(qids, 2, tmp_path)
    assert [p[0] for p in plan] == [0, 1, 2]
    assert [p[1] for p in plan] == [["q0", "q1"], ["q2", "q3"], ["q4"]]
    assert [p[2] for p in plan] == [False, False, False]
    part_file(tmp_path, 1).write_text(json.dumps({"q2": [], "q3": []}), encoding="utf-8")
    plan = plan_parts(qids, 2, tmp_path)
    assert [p[2] for p in plan] == [False, True, False]
    assert plan_parts([], 2, tmp_path) == []


def test_build_pairs_skips_missing_docs_and_limits_top():
    docs = {1: "a", 3: "c"}
    cand = [[1, 10], [2, 20], [3, 30], [4, 40]]
    pairs, pos = build_pairs("qq", cand, docs, top=3)
    assert pairs == [("qq", "a"), ("qq", "c")]
    assert pos == [0, 2]
    pairs, pos = build_pairs("qq", cand, docs, top=1)
    assert pairs == [("qq", "a")] and pos == [0]
    assert build_pairs("qq", cand, {}, top=2) == ([], [])


def test_merge_scores_keeps_bm25_order_and_rounds():
    cand = [[1, 10], [2, 20], [3, 30]]
    res = merge_scores(cand, 3, [0, 2], [1.23456, -0.5])
    assert res == [[1, 10, 1.2346], [2, 20, None], [3, 30, -0.5]]
    assert merge_scores(cand, 2, [0, 1], [2.0, 1.0]) == [[1, 10, 2.0], [2, 20, 1.0]]
    assert merge_scores(cand, 3, [], []) == [[1, 10, None], [2, 20, None], [3, 30, None]]


def test_merge_parts_in_part_and_query_order(tmp_path):
    qids = ["q0", "q1", "q2", "q3"]
    plan = plan_parts(qids, 2, tmp_path)
    part_file(tmp_path, 0).write_text(json.dumps({"q1": 1, "q0": 0}), encoding="utf-8")
    part_file(tmp_path, 1).write_text(json.dumps({"q3": 3}), encoding="utf-8")
    out = merge_parts(plan, tmp_path)
    assert list(out) == ["q0", "q1", "q3"]
    assert out["q3"] == 3


def test_pipeline_build_score_merge_roundtrip(tmp_path):
    docs = {10: "d10", 20: "d20", 30: "d30"}
    cands = {"qA": [[10, 1], [20, 2], [99, 3]], "qB": [[30, 4], [10, 5]]}
    qids = ["qA", "qB"]
    plan = plan_parts(qids, 1, tmp_path)
    scored = {}
    for pi, chunk, done in plan:
        part = {}
        for q in chunk:
            pairs, pos = build_pairs(q, cands[q], docs, top=3)
            fake = [float(len(t)) for _, t in pairs]  # điểm giả: độ dài text
            part[q] = merge_scores(cands[q], 3, pos, fake)
        part_file(tmp_path, pi).write_text(json.dumps(part), encoding="utf-8")
        scored.update(part)
    out = merge_parts(plan, tmp_path)
    assert out == scored
    assert out["qA"] == [[10, 1, 3.0], [20, 2, 3.0], [99, 3, None]]
    assert out["qB"] == [[30, 4, 3.0], [10, 5, 3.0]]


def test_import_without_torch_or_zstandard(monkeypatch):
    for name in ("torch", "zstandard"):
        monkeypatch.setitem(sys.modules, name, None)
    import rerank_run
    importlib.reload(rerank_run)
    assert not hasattr(rerank_run, "torch")
    assert callable(rerank_run.score_pairs)
    assert rerank_run.PART_SIZE == 50


def test_read_docs_roundtrip(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    from rerank_run import load_docs, read_docs
    p = tmp_path / "d.jsonl.zst"
    rows = [{"id": 1, "idx": 7, "text": "aa"}, {"id": 2, "idx": 8, "text": "bbbb"}]
    with open(p, "wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw) as w:
            for r in rows:
                w.write((json.dumps(r) + "\n").encode("utf-8"))
    assert list(read_docs(str(p))) == rows
    assert load_docs(str(p)) == {1: "aa", 2: "bbbb"}
    assert load_docs(str(p), doc_chars=1) == {1: "a", 2: "b"}


def test_load_docs_multi_file_and_wanted_filter(tmp_path):
    zstandard = pytest.importorskip("zstandard")
    from rerank_run import load_docs

    def write_zst(name, rows):
        p = tmp_path / name
        with open(p, "wb") as raw:
            with zstandard.ZstdCompressor().stream_writer(raw) as w:
                for r in rows:
                    w.write((json.dumps(r) + "\n").encode("utf-8"))
        return p

    p0 = write_zst("a.jsonl.zst", [{"id": 1, "idx": 0, "text": "t1"}, {"id": 2, "idx": 1, "text": "t2"}])
    p1 = write_zst("b.jsonl.zst", [{"id": 3, "idx": 2, "text": "t3"}, {"id": 2, "idx": 1, "text": "t2"}])
    assert load_docs([str(p0), str(p1)]) == {1: "t1", 2: "t2", 3: "t3"}
    assert load_docs([str(p0), str(p1)], doc_chars=1) == {1: "t", 2: "t", 3: "t"}
    assert load_docs([str(p0), str(p1)], wanted={2, 3}) == {2: "t2", 3: "t3"}
    assert load_docs(str(p0), wanted={3}) == {}
    assert load_docs([str(p0), str(p1)], doc_chars=1, wanted={1}) == {1: "t"}
