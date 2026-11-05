"""Tests for src.evaluate, src.select, src.submit, src.devset and run.py (mocked modules)."""
from __future__ import annotations

import json
import math
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import select as sel  # noqa: E402
from src import submit as sub  # noqa: E402
from src import devset  # noqa: E402

QUERY_PARQUET = ROOT / "data" / "ViBioMIR" / "query.parquet"


def entry(qid, docs, chunks=()):
    return {"id": qid, "relevant_docs": list(docs),
            "relevant_chunks": [{"doc_id": d, "chunk_text": t} for d, t in chunks]}


def f2(p, r):
    return 5 * p * r / (4 * p + r) if p + r else 0.0


# ------------------------------------------------------------------ evaluate
def test_prf2_formula():
    assert ev.prf2(0, 0, 3) == (0.0, 0.0, 0.0)
    p, r, f = ev.prf2(1, 1, 4)
    assert (p, r) == (1.0, 0.25) and f == pytest.approx(1.25 / 4.25)


def test_doc_level_hand_computed():
    gold = [entry(1, ["a", "b", "c"]), entry(2, ["a", "b", "c", "d"]), entry(3, ["z"])]
    pred = [entry(1, ["x", "a", "b", "a"]),  # duplicate 'a' ignored
            entry(2, ["a"]),
            entry(3, [])]
    res = ev.evaluate(pred, gold, per_query=True)
    q = {r["id"]: r["doc"] for r in res["per_query"]}
    assert q[1]["P"] == pytest.approx(2 / 3) and q[1]["R"] == pytest.approx(2 / 3)
    assert q[1]["F2"] == pytest.approx(2 / 3) and q[1]["MRR@5"] == pytest.approx(0.5)
    assert q[2]["F2"] == pytest.approx(0.294117647) and q[2]["MRR@5"] == 1.0
    assert q[3]["F2"] == 0.0 and q[3]["MRR@5"] == 0.0
    assert res["doc"]["F2"] == pytest.approx((2 / 3 + 1.25 / 4.25) / 3)
    assert res["doc"]["P"] == pytest.approx((2 / 3 + 1) / 3)
    assert res["doc"]["MRR@5"] == pytest.approx(0.5)
    assert res["doc"]["n"] == 3


def test_mrr_only_top5():
    gold = [entry(1, ["g"])]
    pred = [entry(1, ["1", "2", "3", "4", "5", "g"])]
    assert ev.evaluate(pred, gold)["doc"]["MRR@5"] == 0.0
    pred = [entry(1, ["1", "2", "3", "4", "g"])]
    assert ev.evaluate(pred, gold)["doc"]["MRR@5"] == pytest.approx(0.2)


def test_missing_and_extra_pred():
    gold = [entry(1, ["a"]), entry(2, ["b"])]
    pred = [entry(1, ["a"]), entry(99, ["q"])]
    res = ev.evaluate(pred, gold)
    assert res["n_missing_pred"] == 1 and res["n_extra_pred"] == 1
    assert res["doc"]["F2"] == pytest.approx(0.5)


def test_chunk_exact_normalisation_and_doc_id():
    gold = [entry(1, ["d1", "d2"], [("d1", "Sỏi thận gây tắc nghẽn."), ("d2", "Uống nhiều nước.")])]
    pred = [entry(1, ["d1", "d2"], [("d1", "  Sỏi  thận gây\ntắc nghẽn. "),  # whitespace -> match
                                    ("d1", "Uống nhiều nước."),                # wrong doc -> no
                                    ("d2", "uống nhiều nước.")])]              # case -> no (exact)
    r = ev.evaluate(pred, gold, chunk_match="exact")["chunk"]
    assert r["P"] == pytest.approx(1 / 3) and r["R"] == pytest.approx(0.5)
    assert r["F2"] == pytest.approx(f2(1 / 3, 0.5)) and r["MRR@5"] == 1.0
    # NFC: decomposed Vietnamese text equals composed text
    import unicodedata
    nfd = unicodedata.normalize("NFD", "Sỏi thận gây tắc nghẽn.")
    assert ev.chunk_matches({"doc_id": "d1", "chunk_text": nfd},
                            {"doc_id": "d1", "chunk_text": "Sỏi thận gây tắc nghẽn."})


def test_chunk_contain_and_one_to_one():
    g = "Sỏi thận có thể gây tắc nghẽn đường tiết niệu."
    gold = [entry(1, ["d1"], [("d1", g)])]
    # two predictions both containing the single gold chunk -> only 1 hit
    pred = [entry(1, ["d1"], [("d1", "Mở đầu. " + g), ("d1", g + " Kết luận.")])]
    r = ev.evaluate(pred, gold, chunk_match="contain")["chunk"]
    assert r["P"] == 0.5 and r["R"] == 1.0
    # pred shorter than gold (gold contains pred) also matches
    pred = [entry(1, ["d1"], [("d1", "gây tắc nghẽn")])]
    assert ev.evaluate(pred, gold, chunk_match="contain")["chunk"]["F2"] == 1.0
    assert ev.evaluate(pred, gold, chunk_match="exact")["chunk"]["F2"] == 0.0


def test_maximum_matching_beats_greedy():
    # p0 matches g0 and g1, p1 matches only g0: greedy (p0->g0) gives 1, optimum is 2
    gold = [entry(1, ["d"], [("d", "alpha beta"), ("d", "alpha beta gamma delta")])]
    pred = [entry(1, ["d"], [("d", "alpha"), ("d", "alpha beta")])]
    assert ev.max_bipartite_matching([[0, 1], [0]], 2) == 2
    r = ev.evaluate(pred, gold, chunk_match="contain")["chunk"]
    assert r["P"] == 1.0 and r["R"] == 1.0


def test_chunk_overlap_mode_vi_and_zh():
    a = {"doc_id": "d", "chunk_text": "bệnh nhân bị sỏi thận cần uống nhiều nước"}
    b = {"doc_id": "d", "chunk_text": "bệnh nhân bị sỏi thận nên đi khám"}
    # tokens a=9, b=8, common=5 -> F1 = 2*5/(9+8) = 0.588
    assert ev.token_overlap(a["chunk_text"], b["chunk_text"]) == pytest.approx(10 / 17)
    assert ev.chunk_matches(a, b, "overlap", 0.5)
    assert not ev.chunk_matches(a, b, "overlap", 0.6)
    # Chinese: character tokens. 肾结石患者 vs 肾结石治疗 -> 3/5 common -> F1 0.6
    assert ev.tokens("肾结石患者") == ("肾", "结", "石", "患", "者")
    assert ev.token_overlap("肾结石患者", "肾结石治疗") == pytest.approx(0.6)
    assert ev.tokens("BGE-M3 模型") == ("bge", "m3", "模", "型")
    assert ev.token_overlap("x y", "x y z w", "recall") == 0.5
    assert ev.token_overlap("x y", "x y z w", "jaccard") == 0.5


def test_empty_gold_policies():
    gold = [entry(1, ["a"], []), entry(2, [], [])]
    pred_empty = [entry(1, ["a"]), entry(2, [])]
    pred_noise = [entry(1, ["a"]), entry(2, ["x"])]
    r = ev.evaluate(pred_empty, gold, empty_gold="skip")
    assert r["doc"]["n"] == 1 and r["doc"]["F2"] == 1.0 and r["chunk"]["n"] == 0
    assert r["chunk"]["F2"] == 0.0
    r = ev.evaluate(pred_empty, gold, empty_gold="strict")
    assert r["doc"]["n"] == 2 and r["doc"]["F2"] == 1.0 and r["chunk"]["F2"] == 1.0
    r = ev.evaluate(pred_noise, gold, empty_gold="strict")
    assert r["doc"]["F2"] == 0.5


def test_evaluate_cli_and_zip(tmp_path, capsys):
    gold = [entry(1, ["a"], [("a", "xin chào")])]
    sub.write_zip(gold, tmp_path / "p.zip")
    (tmp_path / "g.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    assert ev.main(["--pred", str(tmp_path / "p.zip"), "--gold", str(tmp_path / "g.json"),
                    "--chunk-match", "all", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"exact", "contain", "overlap"}
    assert all(v["final"] == 1.0 for v in out.values())


# -------------------------------------------------------------------- select
def cand(cid, doc, score, text=None, lang="vi", start=None, end=None):
    c = {"chunk_id": cid, "doc_id": doc, "lang": lang, "source": "corpus",
         "text": text if text is not None else f"text of {cid}", "score": score, "stage": "rerank"}
    if start is not None:
        c["start"], c["end"] = start, end
    return c


def test_select_empty():
    assert sel.select([], {}) == ([], [])
    assert sel.select([cand("a#0", "a", 1.0, text="  ")], {}) == ([], [])


def test_select_relative_and_abs_thresholds():
    cs = [cand("a#0", "a", 0.9), cand("b#0", "b", 0.8), cand("c#0", "c", 0.5), cand("d#0", "d", 0.1)]
    docs, chunks = sel.select(cs, {"alpha": 0.8, "max_k": 10})  # thr = 0.72
    assert [c["doc_id"] for c in chunks] == ["a", "b"]
    assert docs == ["a", "b"]
    docs, chunks = sel.select(cs, {"alpha": None, "tau_abs": 0.4})
    assert [c["doc_id"] for c in chunks] == ["a", "b", "c"]
    docs, chunks = sel.select(cs, {"alpha": None, "tau_abs": 0.4, "max_k": 2})
    assert len(chunks) == 2
    # nothing passes tau -> min_k still returns top1 (never empty)
    docs, chunks = sel.select(cs, {"tau_abs": 5.0, "min_k": 1})
    assert [c["doc_id"] for c in chunks] == ["a"] and docs == ["a"]
    docs, chunks = sel.select(cs, {"tau_abs": 5.0, "min_k": 3})
    assert len(chunks) == 3
    # min_k=0 is clamped to 1 (never empty with candidates)
    assert len(sel.select(cs, {"tau_abs": 5.0, "min_k": 0})[1]) == 1


def test_select_negative_scores_relative():
    cs = [cand("a#0", "a", -1.0), cand("b#0", "b", -1.1), cand("c#0", "c", -3.0)]
    # thr = -1 - 0.2*1 = -1.2
    _, chunks = sel.select(cs, {"alpha": 0.8})
    assert [c["doc_id"] for c in chunks] == ["a", "b"]
    # sigmoid normalisation keeps order
    _, chunks = sel.select(cs, {"alpha": 0.5, "score_norm": "sigmoid"})
    assert chunks[0]["doc_id"] == "a"


def test_select_dedupe_offsets_and_containment():
    cs = [cand("a#0", "a", 0.9, "abc def", start=0, end=7),
          cand("a#1", "a", 0.85, "def ghi", start=4, end=11),   # overlaps a#0 -> dropped
          cand("a#2", "a", 0.84, "xyz", start=20, end=23),      # disjoint -> kept
          cand("b#0", "b", 0.83, "Uống nhiều nước mỗi ngày"),
          cand("b#1", "b", 0.82, "nhiều nước"),                  # contained, no offsets -> dropped
          cand("c#0", "c", 0.81, "nhiều nước")]                  # other doc -> kept
    _, chunks = sel.select(cs, {"alpha": 0.0, "max_k": 10})
    assert [c["chunk_text"] for c in chunks] == ["abc def", "xyz", "Uống nhiều nước mỗi ngày",
                                                 "nhiều nước"]
    _, chunks = sel.select(cs, {"alpha": 0.0, "dedupe": False})
    assert len(chunks) == 6


def test_select_docs_superset_and_doc_cutoff():
    cs = [cand("a#0", "a", 0.9), cand("b#0", "b", 0.5), cand("c#0", "c", 0.88),
          cand("d#0", "d", 0.2)]
    # chunk max_k=1 but docs alpha looser -> more docs than chunk docs
    docs, chunks = sel.select(cs, {"alpha": 0.9, "max_k": 1, "doc_alpha": 0.5, "doc_max_k": 5})
    assert [c["doc_id"] for c in chunks] == ["a"]
    assert docs == ["a", "c", "b"]
    # doc cutoff tighter than chunks -> chunk docs are still included
    docs, chunks = sel.select(cs, {"alpha": 0.5, "max_k": 5, "doc_max_k": 1})
    assert set(c["doc_id"] for c in chunks) <= set(docs)
    assert docs == ["a", "c", "b"]


def test_select_doc_agg_and_lang_weights():
    cs = [cand("a#0", "a", 0.9, "t1"), cand("b#0", "b", 0.85, "u1"), cand("b#1", "b", 0.84, "u2"),
          cand("b#2", "b", 0.83, "u3")]
    docs, _ = sel.select(cs, {"alpha": 0.9, "doc_agg": "max"})
    assert docs[0] == "a"
    docs, _ = sel.select(cs, {"alpha": 0.9, "doc_agg": "max_bonus", "doc_bonus": 0.05})
    assert docs[0] == "b"  # 0.85 + 2*0.05 > 0.9
    docs, _ = sel.select(cs, {"alpha": 0.9, "doc_agg": "max_bonus", "doc_bonus": 0.05,
                              "doc_bonus_cap": 0.04})
    assert docs[0] == "a"
    docs, _ = sel.select(cs, {"alpha": 0.9, "doc_agg": "sum_top2"})
    assert docs[0] == "b"
    cs2 = [cand("a#0", "a", 0.9, lang="vi"), cand("e#0", "e", 0.8, lang="en")]
    _, chunks = sel.select(cs2, {"alpha": 0.95, "lang_bias": {"en": 0.2}})
    assert chunks[0]["doc_id"] == "e"
    _, chunks = sel.select(cs2, {"alpha": 0.95, "max_chunks_per_doc": 1, "lang_weights": {"vi": 0.5}})
    assert chunks[0]["doc_id"] == "e"


def test_select_output_schema():
    docs, chunks = sel.select([cand("a#0", "a", 1.0, "x")], None)
    assert docs == ["a"] and chunks == [{"doc_id": "a", "chunk_text": "x"}]


def test_tune_grid_search(tmp_path):
    # q1 has 2 relevant chunks with high scores; q2 has 1 -> best alpha must keep both for q1
    dev = {1: [cand("a#0", "a", 0.9), cand("b#0", "b", 0.7), cand("c#0", "c", 0.2)],
           2: [cand("d#0", "d", 0.95), cand("e#0", "e", 0.3)]}
    gold = [entry(1, ["a", "b"], [("a", "text of a#0"), ("b", "text of b#0")]),
            entry(2, ["d"], [("d", "text of d#0")])]
    best, score, results = sel.tune(dev, gold, {"alpha": [0.1, 0.7, 0.95], "max_k": [1, 5]})
    assert best["alpha"] == 0.7 and best["max_k"] == 5
    assert score == pytest.approx(1.0)
    assert len(results) == 6 and results[0]["score"] >= results[-1]["score"]
    # CLI over a run.py-style cache
    cache = tmp_path / "rerank.jsonl"
    cache.write_text("\n".join(json.dumps({"id": k, "in": "x", "data": v}) for k, v in dev.items()))
    (tmp_path / "gold.json").write_text(json.dumps(gold))
    assert sel.main(["tune", "--cands", str(cache), "--gold", str(tmp_path / "gold.json"),
                     "--grid", '{"alpha":[0.7,0.95]}', "--config", str(tmp_path / "none.yaml"),
                     "--out", str(tmp_path / "res.json")]) == 0
    assert json.loads((tmp_path / "res.json").read_text())["best"]["alpha"] == 0.7


# -------------------------------------------------------------------- submit
def test_validate_ok_and_errors():
    good = [entry(1, ["a"], [("a", "x")]), entry(2, [])]
    assert sub.validate(good, [1, 2]) == []
    bad = [{"id": "1", "relevant_docs": "a", "relevant_chunks": []},
           {"id": 2, "relevant_docs": [3]},
           {"id": 2, "relevant_docs": ["a"], "relevant_chunks": [{"doc_id": 5, "chunk_text": ""},
                                                                 {"doc_id": "b", "chunk_text": "y"}]}]
    errs = sub.validate(bad, [1, 2, 3])
    joined = "\n".join(errs)
    for frag in ("id must be an integer", "relevant_docs must be a list", "missing field relevant_chunks",
                 "non-empty string", "duplicate id", "doc_id must be a non-empty string",
                 "chunk_text must be a non-empty string", "not in relevant_docs", "missing"):
        assert frag in joined, frag
    with pytest.raises(ValueError):
        sub.validate(bad, [1], raise_on_error=True)
    assert sub.validate({"id": 1}, None)  # not a list
    assert sub.validate([{"id": True, "relevant_docs": [], "relevant_chunks": []}], None)


def test_validate_verbatim_lookup():
    pred = [entry(1, ["a"], [("a", "nguyên văn")]), entry(2, ["b"], [("b", "viết lại")]),
            entry(3, ["zz"], [("zz", "x")])]
    texts = {"a": "Đây là đoạn nguyên văn.", "b": "Đây là bản gốc."}
    errs = sub.validate(pred, [1, 2, 3], doc_lookup=texts)
    assert len(errs) == 2
    assert any("verbatim" in e and "id=2" in e for e in errs)
    assert any("not found" in e for e in errs)
    assert sub.validate(pred[:1], [1], doc_lookup=texts.get) == []


def test_complete_fills_and_coerces():
    pred = [{"id": "2", "relevant_docs": [123, "123"], "relevant_chunks": [
        {"doc_id": 9, "chunk_text": "t"}, {"doc_id": 9, "chunk_text": "t"}, {"doc_id": 1, "chunk_text": ""}]}]
    out = sub.complete(pred, [3, 1, 2])
    assert [e["id"] for e in out] == [1, 2, 3]
    assert out[0] == entry(1, [])
    assert out[1]["relevant_docs"] == ["123", "9"]
    assert out[1]["relevant_chunks"] == [{"doc_id": "9", "chunk_text": "t"}]
    assert sub.validate(out, [1, 2, 3]) == []


def test_write_zip_structure(tmp_path):
    pred = [entry(1, ["123"], [("123", "Sỏi thận 肾结石")])]
    p = sub.write_zip(pred, tmp_path / "out" / "sub.zip")
    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
        assert names == ["submission.json"]
        raw = zf.read(names[0])
    assert "Sỏi thận 肾结石".encode("utf-8") in raw  # ensure_ascii=False
    assert b"\\u" not in raw
    assert sub.read_zip(p) == pred
    # nested json_name is flattened to root
    p2 = sub.write_zip(pred, tmp_path / "s2.zip", json_name="dir/x")
    assert zipfile.ZipFile(p2).namelist() == ["x.json"]
    # read_zip rejects bad structure
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("a/sub.json", "[]")
    with pytest.raises(ValueError):
        sub.read_zip(bad)
    assert sub.main(["check", str(bad), "--queries", str(tmp_path / "nope.parquet")]) == 1


def test_submit_cli_check(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text('{"id": 1, "query": "a"}\n{"id": 2, "query": "b"}\n')
    sub.write_zip([entry(1, ["a"]), entry(2, [])], tmp_path / "ok.zip")
    assert sub.main(["check", str(tmp_path / "ok.zip"), "--queries", str(q)]) == 0
    sub.write_zip([entry(1, ["a"])], tmp_path / "miss.zip")
    assert sub.main(["check", str(tmp_path / "miss.zip"), "--queries", str(q)]) == 1


def test_load_doc_texts(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"doc_id": ["1", "2"], "text": ["t1", "t2"], "lang": ["vi", "zh"]}),
                   tmp_path / "d.parquet")
    assert sub.load_doc_texts(["2", "3"], tmp_path) == {"2": "t2"}


# -------------------------------------------------------------------- devset
def _write_docs(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    docs = {"doc_id": [str(i) for i in range(20)],
            "lang": ["vi", "zh", "en", "vi"] * 5,
            "source": ["corpus"] * 20,
            "title": ["t"] * 20,
            "text": [f"Tiêu đề {i}\n\n" + ("Đoạn văn y khoa số %d nói về sỏi thận và cách điều trị. " % i) * 3
                     + "\n\nngắn" for i in range(20)]}
    pq.write_table(pa.table(docs), tmp_path / "docs.parquet")
    return docs


def test_devset_sample_generate_build(tmp_path):
    docs = _write_docs(tmp_path)
    chunks = devset.sample_chunks(tmp_path / "docs.parquet", n=5, min_chars=50)
    assert len(chunks) == 5 and len({c["doc_id"] for c in chunks}) == 5
    for c in chunks:  # offsets exact w.r.t. Doc.text
        assert docs["text"][int(c["doc_id"])][c["start"]:c["end"]] == c["text"]
    q = devset.sample_chunks(tmp_path / "docs.parquet", per_lang={"zh": 2, "en": 1}, min_chars=50)
    assert sorted(c["lang"] for c in q) == ["en", "zh", "zh"]

    prompts = tmp_path / "p.jsonl"
    assert devset.write_prompts(chunks, prompts) == 5
    rec = json.loads(prompts.read_text(encoding="utf-8").splitlines()[0])
    assert rec["chunk_text"] in rec["prompt"] and "tiếng Việt" in rec["prompt"]

    calls = []

    def fake_generate(ps, max_new_tokens=0, json=False):
        calls.append(len(ps))
        outs = ['{"question": "Bị sỏi thận thì nên làm gì?"}', "not json at all {",
                {"question": "Câu hỏi dạng dict?"}, '```json\n{"question": "Có fence?"}\n```',
                "Câu hỏi thô không có json"]
        return outs[:len(ps)]

    gen = tmp_path / "g.jsonl"
    n = devset.generate_questions(prompts, gen, batch_size=5, generate_fn=fake_generate)
    assert n == 4  # one unparsable
    # resumable: second call does nothing
    assert devset.generate_questions(prompts, gen, generate_fn=fake_generate) == 0
    assert calls == [5]
    nq = devset.build_synthetic(gen, tmp_path / "q.jsonl", tmp_path / "gold.json")
    gold = json.loads((tmp_path / "gold.json").read_text(encoding="utf-8"))
    assert nq == 4 and sub.validate(gold, [g["id"] for g in gold]) == []
    assert gold[0]["relevant_chunks"][0]["chunk_text"] == chunks[0]["text"]
    # gold evaluates perfectly against itself
    assert ev.evaluate(gold, gold)["final"] == 1.0


def test_devset_sample_chunk_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"chunk_id": [f"{i}#0" for i in range(10)], "doc_id": [str(i) for i in range(10)],
                             "lang": ["vi"] * 10, "source": ["corpus"] * 10, "start": [0] * 10,
                             "end": [100] * 10, "text": ["x" * 100] * 10}), tmp_path / "c.parquet")
    out = devset.sample_chunks(tmp_path, n=3)
    assert len(out) == 3 and all(c["chunk_id"].endswith("#0") for c in out)


def test_pooling_roundtrip():
    r1 = [entry(1, ["a", "b"], [("a", "x"), ("b", "y")])]
    r2 = [entry(1, ["b", "c"], [("b", " y "), ("c", "z")])]
    p = devset.pool({"s1": r1, "s2": r2}, queries={1: "q"})
    assert p[0]["query"] == "q"
    assert [d["doc_id"] for d in p[0]["docs"]] == ["b", "a", "c"]
    assert p[0]["docs"][0]["systems"] == ["s1", "s2"]
    assert len(p[0]["chunks"]) == 3 and p[0]["chunks"][0]["doc_id"] == "b"
    p[0]["docs"][1]["label"] = 1     # a
    p[0]["chunks"][2]["label"] = 1   # c/z
    gold = devset.pool_to_gold(p)
    assert gold == [entry(1, ["a", "c"], [("c", "z")])]


# ------------------------------------------------------------- run.py dry run
def _install_fakes(monkeypatch, retrieve_extra=True, fail_ids=(), calls=None):
    calls = calls if calls is not None else {}

    def bump(k):
        calls[k] = calls.get(k, 0) + 1

    q = types.ModuleType("src.query")

    def build_plans(queries, cfg=None, llm=None, use_cache=True):
        bump("plan")
        return [{"id": x["id"], "query_vi": x["query"], "vi": [x["query"]], "en": ["en " + str(x["id"])],
                 "zh": [], "pubmed": [f"term{x['id']}"], "keywords": {}} for x in queries]
    q.build_plans = build_plans

    pm = types.ModuleType("src.pubmed")

    def candidates_for_plan(plan):
        bump("pubmed")
        pid = str(30000000 + plan["id"])
        return [{"doc_id": pid, "source": "pubmed", "lang": "en", "url": "", "title": "T",
                 "text": f"T\n\nAbstract for {plan['id']}.", "fetched_at": ""}]
    pm.candidates_for_plan = candidates_for_plan

    rt = types.ModuleType("src.retrieve")

    def _local(plan):
        qid = plan["id"]
        if qid in fail_ids:
            raise RuntimeError("boom")
        return [{"chunk_id": f"{qid}00#0", "doc_id": f"{qid}00", "lang": "vi", "source": "corpus",
                 "text": f"Đoạn {qid} thứ nhất", "score": 0.5, "stage": "rrf"},
                {"chunk_id": f"{qid}01#0", "doc_id": f"{qid}01", "lang": "zh", "source": "corpus",
                 "text": f"段落 {qid}", "score": 0.4, "stage": "rrf"}]

    if retrieve_extra:
        def retrieve(plan, k, extra_docs=None):
            bump("retrieve")
            out = _local(plan)
            for d in extra_docs or []:
                out.append({"chunk_id": d["doc_id"] + "#0", "doc_id": d["doc_id"], "lang": "en",
                            "source": "pubmed", "text": d["text"], "score": 0.3, "stage": "dense"})
            return out
    else:
        def retrieve(plan, k):
            bump("retrieve")
            return _local(plan)
    rt.retrieve = retrieve

    rr = types.ModuleType("src.rerank")

    def rerank(query_vi, candidates):
        bump("rerank")
        out = []
        for i, c in enumerate(candidates):
            c = dict(c, stage="rerank", score=[0.95, 0.9, 0.1, 0.05][min(i, 3)])
            out.append(c)
        return out
    rr.rerank = rerank

    ch = types.ModuleType("src.chunk")
    ch.chunk_doc = lambda doc, cfg: [{"chunk_id": doc["doc_id"] + "#0", "doc_id": doc["doc_id"],
                                      "start": 0, "end": len(doc["text"]), "text": doc["text"]}]
    for name, mod in (("src.query", q), ("src.pubmed", pm), ("src.retrieve", rt),
                      ("src.rerank", rr), ("src.chunk", ch)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _cfg(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("select:\n  alpha: 0.8\n  max_k: 5\nrun:\n  log_every: 0\n", encoding="utf-8")
    return p


@pytest.fixture
def run_mod():
    import importlib
    return importlib.import_module("run")


@pytest.mark.skipif(not QUERY_PARQUET.exists(), reason="query.parquet missing")
def test_run_dry_full_1200(tmp_path, monkeypatch, run_mod):
    calls = _install_fakes(monkeypatch, fail_ids={5})
    out = tmp_path / "sub.zip"
    argv = ["--queries", str(QUERY_PARQUET), "--config", str(_cfg(tmp_path)), "--out", str(out),
            "--cache-dir", str(tmp_path / "cache")]
    assert run_mod.main(argv) == 0
    pred = sub.read_zip(out)
    qids = sub.load_query_ids(QUERY_PARQUET)
    assert len(pred) == 1200 == len(qids)
    assert sub.validate(pred, qids) == []
    by = {e["id"]: e for e in pred}
    first = qids[0]
    assert by[first]["relevant_docs"] == [f"{first}00", f"{first}01"]
    assert len(by[first]["relevant_chunks"]) == 2
    if 5 in by:  # failed retrieve -> still an entry, empty lists
        assert by[5] == entry(5, [])
    assert calls["plan"] == 1 and calls["rerank"] == 1200 - (5 in by)
    for st in ("plan", "pubmed", "retrieve", "rerank", "select"):
        assert (tmp_path / "cache" / f"{st}.jsonl").exists()
    assert json.loads((tmp_path / "cache" / "timing.json").read_text())["n_queries"] == 1200

    # rerun only select+submit with a different cutoff: no upstream calls, uses cache
    before = dict(calls)
    cfg2 = tmp_path / "cfg2.yaml"
    cfg2.write_text("select:\n  alpha: 0.0\n  max_k: 5\n", encoding="utf-8")
    out2 = tmp_path / "sub2.zip"
    assert run_mod.main(argv[:2] + ["--config", str(cfg2), "--out", str(out2), "--cache-dir",
                                    str(tmp_path / "cache"), "--stages", "select,submit"]) == 0
    assert calls == before
    pred2 = {e["id"]: e for e in sub.read_zip(out2)}
    assert len(pred2[first]["relevant_chunks"]) == 3  # pubmed chunk now included (alpha=0)
    assert f"{30000000 + first}" in pred2[first]["relevant_docs"]

    # full rerun resumes from cache: failed query retried, others cached
    assert run_mod.main(argv) == 0
    assert calls["rerank"] == before["rerank"]


def test_run_limit_no_extra_docs_and_missing_module(tmp_path, monkeypatch, run_mod):
    calls = _install_fakes(monkeypatch, retrieve_extra=False)
    q = tmp_path / "q.jsonl"
    q.write_text("".join(json.dumps({"id": i, "query": f"câu {i}"}, ensure_ascii=False) + "\n"
                         for i in range(1, 11)), encoding="utf-8")
    out = tmp_path / "s.zip"
    base = ["--queries", str(q), "--config", str(_cfg(tmp_path)), "--cache-dir", str(tmp_path / "c")]
    assert run_mod.main(base + ["--out", str(out), "--limit", "3"]) == 0
    pred = sub.read_zip(out)
    assert [e["id"] for e in pred] == [1, 2, 3]
    # pubmed chunks were appended by run.py's fallback chunking -> reranked
    rer = [json.loads(l) for l in (tmp_path / "c" / "rerank.jsonl").read_text().splitlines()]
    assert any(c["source"] == "pubmed" for c in rer[0]["data"])

    # rerank module missing -> clear error (rc 2) unless --allow-missing
    monkeypatch.setitem(sys.modules, "src.rerank", None)  # import -> ImportError
    assert run_mod.main(base + ["--out", str(out), "--force"]) == 2
    assert run_mod.main(base + ["--out", str(out), "--allow-missing", "--stages",
                                "rerank,select,submit"]) == 0
    pred = sub.read_zip(out)
    assert len(pred) == 10 and sub.validate(pred, range(1, 11)) == []
    # queries 4..10 have no retrieve cache (stage not listed) -> empty entries, still valid
    assert pred[-1] == entry(10, [])
    assert pred[0]["relevant_docs"]


def test_call_flex(run_mod):
    def f(a, b=1):
        return (a, b)

    def g(a, **kw):
        return (a, kw)
    assert run_mod.call_flex(f, 1, b=2, c=3) == (1, 2)
    assert run_mod.call_flex(g, 1, c=3) == (1, {"c": 3})
