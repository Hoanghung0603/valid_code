"""Tests for src/pubmed.py.

Offline tests use httpx.MockTransport. Live tests hit NCBI / Europe PMC and run only with PUBMED_LIVE=1:
    PUBMED_LIVE=1 .venv/bin/python -m pytest tests/test_pubmed.py -q -s
"""
import json
import os
import threading
import time
from urllib.parse import parse_qs

import httpx
import pytest

from src import pubmed as pm

XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedArticle><MedlineCitation Status="MEDLINE"><PMID Version="1">111</PMID>
<Article><Journal><Title>Test Journal</Title><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Kidney <i>stones</i>: treatment.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Stones are   common.</AbstractText>
<AbstractText Label="CONCLUSIONS">Drain   obstructed kidneys<sup>1</sup>.</AbstractText></Abstract>
<Language>eng</Language>
<PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType><PublicationType UI="D016454">Review</PublicationType></PublicationTypeList>
</Article>
<MeshHeadingList><MeshHeading><DescriptorName UI="D007669">Kidney Calculi</DescriptorName></MeshHeading></MeshHeadingList>
</MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">222</PMID>
<Article><Journal><JournalIssue><PubDate><MedlineDate>1998 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
<ArticleTitle>No abstract here.</ArticleTitle></Article></MedlineCitation></PubmedArticle>
<PubmedBookArticle><BookDocument><PMID Version="1">333</PMID><ArticleTitle>Nephrolithiasis</ArticleTitle>
<Book><BookTitle>StatPearls</BookTitle><PubDate><Year>2024</Year></PubDate></Book>
<Abstract><AbstractText>Plain unlabeled abstract.</AbstractText></Abstract></BookDocument></PubmedBookArticle>
</PubmedArticleSet>"""


def test_parse_xml():
    docs = {d["doc_id"]: d for d in pm.parse_pubmed_xml(XML)}
    d = docs["111"]
    assert d["title"] == "Kidney stones: treatment."
    assert d["text"] == ("Kidney stones: treatment.\n\nBACKGROUND: Stones are common.\n\n"
                         "CONCLUSIONS: Drain obstructed kidneys1.")
    assert d["year"] == 2019 and d["has_abstract"] and d["source"] == "pubmed" and d["lang"] == "en"
    assert d["pub_types"] == ["Journal Article", "Review"] and d["mesh_terms"] == ["Kidney Calculi"]
    assert d["url"] == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert docs["222"]["has_abstract"] is False and docs["222"]["year"] == 1998
    assert docs["222"]["text"] == "No abstract here."
    assert docs["333"]["text"] == "Nephrolithiasis\n\nPlain unlabeled abstract." and docs["333"]["year"] == 2024


def test_clean_and_relax():
    assert pm.clean_terms("beta-hCG below 0.10 U/L interpretation") == ["beta-hCG", "below", "interpretation"]
    assert pm.clean_terms("beta-hCG below 0.10 U/L interpretation", drop_generic=True) == ["beta-hCG"]
    assert pm.clean_terms("COVID-19 vaccine HbA1c 7.5%") == ["COVID-19", "vaccine", "HbA1c"]
    v = pm.relaxations("SCC tumor marker 1.52 elevated", 8)
    assert v[0] == ("SCC tumor marker elevated", False)
    assert v[-1][1] is True and " OR " in v[-1][0]
    assert sum(1 for _, o in v if o) == 1 and len(v) <= 8
    q = '("Kidney Calculi"[MeSH] OR nephrolithiasis) AND "ureteral obstruction"[tiab] AND therapy[sh]'
    assert pm.split_clauses(q) == ['("Kidney Calculi"[MeSH] OR nephrolithiasis)', '"ureteral obstruction"[tiab]',
                                   "therapy[sh]"]
    bv = [x for x, _ in pm.relaxations(q, 8)]
    assert '("Kidney Calculi"[MeSH] OR nephrolithiasis) "ureteral obstruction"[tiab]' in bv
    # query.py style: keyword string with an OR group, implicit AND
    assert pm.split_clauses("kidney calculi (obstruction OR hydronephrosis) treatment") == [
        "kidney", "calculi", "(obstruction OR hydronephrosis)", "treatment"]
    assert pm.split_clauses("hcg OR gonadotropin negative") == ["(hcg OR gonadotropin)", "negative"]
    assert pm.is_boolean(q) and not pm.is_boolean("kidney stone treatment")


def test_rrf():
    fused = pm.rrf([(["a", "b", "c"], 1.0), (["c", "a"], 1.0), (["d"], 0.1)], k=60)
    assert [p for p, _ in fused][:2] == ["a", "c"] and fused[-1][0] == "d"


def test_rate_limiter_spacing_threadsafe():
    rl = pm.RateLimiter(20)  # 50 ms spacing
    stamps, lock = [], threading.Lock()

    def go():
        rl.acquire()
        with lock:
            stamps.append(time.monotonic())

    ts = [threading.Thread(target=go) for _ in range(10)]
    t0 = time.monotonic()
    [t.start() for t in ts]
    [t.join() for t in ts]
    stamps.sort()
    assert stamps[-1] - t0 >= 0.4
    assert min(b - a for a, b in zip(stamps, stamps[1:])) > 0.03


@pytest.fixture
def mock_client(tmp_path):
    calls = {"esearch": [], "efetch": [], "epmc": [], "n429": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        form = parse_qs(req.content.decode()) if req.method == "POST" else {}
        if "esearch" in url:
            term = form["term"][0]
            calls["esearch"].append(term)
            if term == "flaky" and calls["n429"] == 0:
                calls["n429"] += 1
                return httpx.Response(429, headers={"Retry-After": "0"})
            if term.startswith("kidney stone urinary obstruction treatment extra"):
                ids, cnt = [], 0  # too specific -> relaxation
            elif " OR " in term:
                ids, cnt = ["900"], 5000
            else:
                ids, cnt = ["111", "222", "333"], 3 if "relax" in term else 50
            return httpx.Response(200, json={"esearchresult": {"count": str(cnt), "idlist": ids}})
        if "efetch" in url:
            calls["efetch"].append(form["id"][0])
            return httpx.Response(200, content=XML.encode())
        if "elink" in url:
            calls.setdefault("elink", []).append(form.get("id"))
            return httpx.Response(200, json={"linksets": [{"ids": [i], "linksetdbs": [
                {"linkname": "pubmed_pubmed", "links": [{"id": i, "score": 9}, {"id": "333", "score": 5}]}]}
                for i in form.get("id", [])]})
        if "europepmc" in url:
            calls["epmc"].append(req.url.params["query"])
            return httpx.Response(200, json={"hitCount": 100, "resultList": {"result": [
                {"id": "333", "source": "MED", "pmid": "333"}, {"id": "444", "source": "MED", "pmid": "444"}]}})
        return httpx.Response(404)

    cfg = pm.load_config()
    cfg.update(cache_db=str(tmp_path / "pm.sqlite"), max_retries=2, ncbi_rps=1000, epmc_rps=1000)
    c = pm.PubMedClient(cfg)
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c, calls


def test_search_fetch_cache(mock_client):
    c, calls = mock_client
    ranked = c.search(["kidney stone treatment"], retmax=10)
    assert ranked[0] in {"111", "333"} and set(ranked) == {"111", "222", "333", "444"}
    n = len(calls["esearch"]) + len(calls["epmc"])
    c.search(["kidney stone treatment"], retmax=10)  # cached
    assert len(calls["esearch"]) + len(calls["epmc"]) == n
    docs = c.fetch(["111", "222", "333", "555"])
    assert [d["doc_id"] for d in docs] == ["111", "333"]  # 222 has no abstract, 555 missing
    assert len(c.fetch(["222"], keep_no_abstract=True)) == 1
    c.fetch(["111", "222", "333", "555"])  # all cached (555 remembered as missing)
    assert len(calls["efetch"]) == 1


def test_relaxation_and_retry(mock_client):
    c, calls = mock_client
    out = c.esearch_relaxed("kidney stone urinary obstruction treatment extra", 10)
    assert out and len(calls["esearch"]) > 1  # fell back to relaxed variants
    assert c.esearch("flaky", 10)[0] == ["111", "222", "333"] and calls["n429"] == 1


def test_candidates_for_plan(mock_client):
    c, calls = mock_client
    plan = {"id": 1, "query_vi": "x", "en": ["kidney stone treatment"],
            "pubmed": ['"Kidney Calculi"[MeSH] AND therapy'], "vi": [], "zh": []}
    docs = c.candidates_for_plan(plan)
    assert [d["doc_id"] for d in docs][:2] and all(d["has_abstract"] for d in docs)
    assert docs[0]["pm_rank"] == 0 and docs[0]["pm_score"] > 0
    assert len(calls["epmc"]) == 1 and "SRC:MED" in calls["epmc"][0]  # only plan["en"] goes to EPMC
    rel = c.elink(["111", "222"], per_hit=5)
    assert rel == {"111": ["333"], "222": ["333"]}  # self-links removed
    n = len(calls.get("elink", []))
    c.elink(["111"])
    assert len(calls.get("elink", [])) == n  # cached


def test_cli_resume(tmp_path, mock_client, monkeypatch):
    c, calls = mock_client
    monkeypatch.setattr(pm, "get_client", lambda cfg=None: c)
    plans = tmp_path / "plans.jsonl"
    plans.write_text("\n".join(json.dumps({"id": i, "en": [f"kidney stone treatment {i}"], "pubmed": []})
                               for i in range(3)))
    out, cands = tmp_path / "pm.parquet", tmp_path / "c.jsonl"
    pm.main(["--plans", str(plans), "--out", str(out), "--cands-out", str(cands)])
    import pandas as pd

    df = pd.read_parquet(out)
    assert set(df.doc_id) == {"111", "333"} and {"pub_types", "mesh_terms", "year"} <= set(df.columns)
    assert all(t.startswith(ti) for t, ti in zip(df.text, df.title))
    assert len(cands.read_text().splitlines()) == 3
    n = len(calls["esearch"])
    pm.main(["--plans", str(plans), "--out", str(out), "--cands-out", str(cands)])
    assert len(calls["esearch"]) == n  # resumed from cache


# ---------------------------------------------------------------- live
live = pytest.mark.skipif(os.environ.get("PUBMED_LIVE") != "1", reason="set PUBMED_LIVE=1 for network tests")

LIVE_CASES = [  # (query_vi, english rewrites, expected words in top titles)
    ("Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?",
     ["kidney stone urinary obstruction treatment"], ["stone", "calculi", "ureter", "renal", "kidney", "obstruct"]),
    ("Cách phòng tránh nguy cơ lây nhiễm bệnh lao đa kháng là gì?",
     ["multidrug-resistant tuberculosis transmission prevention"], ["tuberculosis", "tb"]),
    ("Kết quả của Beta-HCG là dưới 0.10 U/L thì là như thế nào ạ?",
     ["beta-hCG below 0.10 U/L interpretation"], ["hcg", "gonadotropin", "pregnan"]),
    ("Chỉ số SCC của em sau khi xét nghiệm là 1.52 ...",
     ["SCC tumor marker 1.52 elevated"], ["scc", "squamous"]),
]


@live
@pytest.mark.parametrize("query_vi,en,words", LIVE_CASES)
def test_live_relevance(query_vi, en, words):
    c = pm.get_client()
    t = time.time()
    docs = c.candidates_for_plan({"query_vi": query_vi, "en": en, "pubmed": []}, max_candidates=50)
    dt = time.time() - t
    assert len(docs) >= 20, "relaxation should keep candidate lists non-trivial"
    top = docs[:10]
    rel = sum(any(w in d["text"].lower() for w in words) for d in top)
    print(f"\n{en[0]!r}: {len(docs)} docs in {dt:.1f}s, {rel}/10 top on-topic")
    for d in top[:5]:
        print("   ", d["doc_id"], d["title"][:90])
    assert rel >= 7
