"""Offline tests of the FPT API clients: batching, ordering, caching and the submission guard (no network)."""

import pytest

from medret.rerank import fpt_api
from medret.retrieve import translate


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setattr(fpt_api, "load_env", lambda: {"FPT_API_BASE": "http://x", "FPT_API_KEY": "k"})
    monkeypatch.setattr(translate, "load_env", lambda: {"FPT_API_BASE": "http://x", "FPT_API_KEY": "k"})


def test_rerank_batches_per_query_keeps_order_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_post(self, query, docs):
        calls.append((query, len(docs)))
        return [len(d) / 100 for d in docs]

    monkeypatch.setattr(fpt_api.FPTReranker, "_post", fake_post)
    r = fpt_api.FPTReranker(cache_path=tmp_path / "c.sqlite", batch_docs=2, threads=2)
    pairs = [("q1", "a"), ("q2", "bb"), ("q1", "ccc"), ("q1", "dddd")]
    assert r.score_pairs(pairs) == pytest.approx([0.01, 0.02, 0.03, 0.04])
    assert sorted(calls) == [("q1", 1), ("q1", 2), ("q2", 1)]  # one query per request, <= batch_docs docs
    calls.clear()
    assert r.score_pairs(pairs) == pytest.approx([0.01, 0.02, 0.03, 0.04]) and not calls  # all from cache


def test_translate_cache_and_guard(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(translate.Translator, "_chat", lambda self, t: seen.append(t) or f"zh:{t}")
    tr = translate.Translator("Qwen3.6-27B", cache_path=tmp_path / "t.sqlite", threads=2)
    assert tr.translate(["a", "b"]) == ["zh:a", "zh:b"]
    assert tr.translate(["b", "c"]) == ["zh:b", "zh:c"] and seen.count("b") == 1
    with pytest.raises(RuntimeError):
        translate.assert_submittable(["Qwen3.6-27B"])
    translate.assert_submittable(["Qwen2.5-VL-7B-Instruct"])
