"""Unit tests for src/query.py and src/llm.py JSON parsing (no GPU needed).

Run: .venv/bin/python -m pytest tests/test_query.py -q
"""
import json

import pytest

from src.llm import _common_prefix_len, parse_json
from src import query as Q

GOOD = {"terms": [["sỏi thận", "kidney stones", "肾结石"], ["tắc nghẽn đường tiết niệu", "urinary tract obstruction", "尿路梗阻"]],
        "vi": ["Xử trí tắc nghẽn đường tiết niệu do sỏi thận"],
        "en": ["Management of urinary tract obstruction caused by kidney stones"],
        "zh": ["肾结石引起尿路梗阻的处理"],
        "pubmed": ["kidney calculi urinary obstruction treatment", '"nephrolithiasis"[MeSH Terms] AND obstruction']}


def test_parse_json_plain_and_fenced():
    s = json.dumps(GOOD, ensure_ascii=False)
    assert parse_json(s) == GOOD
    assert parse_json("```json\n" + s + "\n```") == GOOD
    assert parse_json("Here you go: " + s + " hope it helps {x}") == GOOD


def test_parse_json_trailing_comma_and_truncated():
    assert parse_json('{"a": [1, 2,], "b": "x",}') == {"a": [1, 2], "b": "x"}
    t = parse_json('{"vi": ["một", "hai"], "en": ["one", "tw')
    assert t["vi"] == ["một", "hai"] and t["en"][0] == "one"
    assert parse_json("no json here") is None
    assert parse_json("") is None


def test_normalize_plan_schema():
    plan, probs = Q.normalize_plan(GOOD, 7, "Cần làm gì đối với tắc nghẽn đường tiết niệu do sỏi thận?")
    assert set(plan) == {"id", "query_vi", "vi", "en", "zh", "pubmed", "keywords"}
    assert set(plan["keywords"]) == {"vi", "en", "zh"}
    assert plan["id"] == 7 and plan["keywords"]["en"] == ["kidney stones", "urinary tract obstruction"]
    # field tags / quotes / AND removed
    assert plan["pubmed"][1] == "nephrolithiasis obstruction"
    assert not probs


def test_normalize_plan_bad_input_falls_back():
    plan, probs = Q.normalize_plan(None, 3, "Đau đầu là bệnh gì?")
    assert plan["vi"] == ["Đau đầu là bệnh gì?"] and plan["en"] == [] and "empty:en" in probs
    # Chinese in en / non-Chinese in zh are filtered
    plan, _ = Q.normalize_plan({"en": ["头痛"], "zh": ["headache"], "vi": "đau đầu"}, 3, "x")
    assert plan["en"] == [] and plan["zh"] == [] and plan["vi"] == ["đau đầu"]


def test_leak_guard_drops_fewshot_terms():
    raw = dict(GOOD, terms=GOOD["terms"] + [["bong gân cổ chân", "ankle sprain", "踝关节扭伤"]],
               vi=GOOD["vi"] + ["Bong gân bao lâu khỏi?"], en=GOOD["en"] + ["Ankle sprain recovery"],
               zh=GOOD["zh"] + ["踝关节扭伤恢复"], pubmed=GOOD["pubmed"] + ["ankle sprain radiography"])
    plan, probs = Q.normalize_plan(raw, 1, "Sỏi thận gây tắc nghẽn đường tiết niệu?")
    assert "ankle sprain" not in plan["keywords"]["en"] and "leak" in probs
    assert len(plan["en"]) == len(plan["vi"]) == len(plan["zh"]) == 1
    assert not any("ankle" in p for p in plan["pubmed"])
    plan, _ = Q.normalize_plan(raw, 1, "Sỏi thận và bong gân cổ chân?")
    assert "ankle sprain" in plan["keywords"]["en"] and len(plan["en"]) == 2


def test_common_prefix_len():
    assert _common_prefix_len([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert _common_prefix_len([[5], [6]]) == 0


class FakeLLM:
    def __init__(self, bad_ids=()):
        self.calls = 0
        self.bad_ids = set(bad_ids)
        self.stats = {}

    def generate(self, msgs, max_new_tokens=0, prefill="", **kw):
        self.calls += 1
        outs = []
        for m in msgs:
            q = m[-1]["content"]
            if any(f"#{i}" in q for i in self.bad_ids):
                outs.append(prefill + "garbage")
            else:
                outs.append(json.dumps(GOOD, ensure_ascii=False))
        return outs


def test_build_plans_cache_and_fallback(tmp_path):
    cfg = {"cache": str(tmp_path / "q.sqlite"), "batch_size": 2}
    qs = [{"id": 1, "query": "Sỏi thận #1"}, {"id": 2, "query": "Tắc nghẽn #2"}, {"id": 3, "query": "x #3"}]
    llm = FakeLLM(bad_ids=[3])
    plans = Q.build_plans(qs, cfg, llm=llm)
    assert [p["id"] for p in plans] == [1, 2, 3]
    assert plans[0]["en"] == GOOD["en"]
    assert plans[2]["en"] == [] and plans[2]["vi"] == ["x #3"]  # fallback after retry
    n = llm.calls
    plans2 = Q.build_plans(qs, cfg, llm=llm)  # all cached now
    assert llm.calls == n and plans2 == plans
    # changed query text -> cache miss
    Q.build_plans([{"id": 1, "query": "Sỏi thận khác #1"}], cfg, llm=llm)
    assert llm.calls == n + 1


def test_build_messages_contains_query():
    m = Q.build_messages("Beta-HCG dưới 0.10 U/L là sao?")
    assert [x["role"] for x in m] == ["system", "user"] and m[-1]["content"].startswith("Beta-HCG")
    m = Q.build_messages("Beta-HCG?", fewshot=True)
    assert m[0]["role"] == "system" and json.loads(m[2]["content"]) == Q.EX_A


def test_parse_plan_json_salvages_malformed_terms():
    t = ('{"terms": [["thủ dâm", "self-stimulation (penile"), "自慰"]], "vi": ["x", "y"], '
         '"en": ["X"], "zh": ["中文"], "pubmed": ["k k"]}')
    d = Q.parse_plan_json(t)
    assert d["vi"] == ["x", "y"] and d["en"] == ["X"] and d["zh"] == ["中文"] and d["pubmed"] == ["k k"]
    assert Q.parse_plan_json(json.dumps(GOOD, ensure_ascii=False)) == GOOD
