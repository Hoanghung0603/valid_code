"""End-to-end test of chunk -> index -> retrieve -> rerank on the handwritten fixture
(tests/fixtures/docs_{vi,zh,en}.jsonl: 27 medical topics x 4 docs x 3 languages).

vi + zh docs are indexed on disk (2 doc files -> 2 manifest entries; vi chunk-level dense,
zh doc-level dense = 2-level path); en docs play the role of PubMed candidates and are passed
as `extra_docs` (chunked + embedded at query time).

Run:  .venv/bin/python -m pytest -q tests/test_retrieval.py      (GPU tests skip without CUDA)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunk import chunk_doc  # noqa: E402
from src import index as I  # noqa: E402

FIX = ROOT / "tests/fixtures"


def load_fixture(lang):
    with open(FIX / f"docs_{lang}.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


DOCS = {l: load_fixture(l) for l in ("vi", "zh", "en")}
TOPIC = {d["doc_id"]: d["topic"] for l in DOCS for d in DOCS[l]}

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    HAS_CUDA = False

# Hand-written QueryPlans for real ViBioMIR queries (query_vi is read from query.parquet).
PLANS = {
    5: {"topic": "mdr_tb_prevention", "vi": ["phòng ngừa lây nhiễm lao đa kháng thuốc"],
        "en": ["prevention of multidrug-resistant tuberculosis transmission"], "zh": ["耐多药结核病 如何预防传染"],
        "keywords": {"vi": ["lao đa kháng"], "en": ["MDR-TB", "infection control"], "zh": ["耐多药结核"]}},
    6: {"topic": "thyroid_nodule", "vi": ["điều trị nhân tuyến giáp"],
        "en": ["thyroid nodule management fine needle aspiration surgery"], "zh": ["甲状腺结节 是否需要治疗 手术"],
        "keywords": {"vi": ["nhân tuyến giáp"], "en": ["thyroid nodule"], "zh": ["甲状腺结节"]}},
    14: {"topic": "coarse_liver_parenchyma", "vi": ["nhu mô gan thô nghĩa là gì"],
         "en": ["coarse liver echotexture on ultrasound meaning"], "zh": ["肝实质回声增粗 是什么意思"],
         "keywords": {"vi": ["nhu mô gan thô"], "en": ["coarse liver parenchyma"], "zh": ["肝实质回声增粗"]}},
    622: {"topic": "hypertension_cure", "vi": ["tăng huyết áp có chữa khỏi hoàn toàn không"],
          "en": ["can hypertension be cured permanently"], "zh": ["高血压 能根治吗"],
          "keywords": {"vi": ["cao huyết áp"], "en": ["hypertension"], "zh": ["高血压"]}},
    830: {"topic": "bruxism", "vi": ["nghiến răng khi ngủ cách chữa"],
          "en": ["sleep bruxism treatment"], "zh": ["睡觉磨牙 怎么办"],
          "keywords": {"vi": ["nghiến răng"], "en": ["bruxism"], "zh": ["磨牙"]}},
    1195: {"topic": "oral_contraceptive_start", "vi": ["bắt đầu uống viên thuốc tránh thai đầu tiên khi nào"],
           "en": ["when to start combined oral contraceptive pill first pill"], "zh": ["口服避孕药 第一片 什么时候开始吃"],
           "keywords": {"vi": ["thuốc tránh thai"], "en": ["oral contraceptive"], "zh": ["避孕药"]}},
}


def query_text(qid):
    t = pq.read_table(ROOT / "data/ViBioMIR/query.parquet").to_pylist()
    return {r["id"]: r["query"] for r in t}[qid]


def make_plan(qid, rewrites=True):
    p = PLANS[qid]
    plan = {"id": qid, "query_vi": query_text(qid), "vi": [], "en": [], "zh": [], "pubmed": [],
            "keywords": {"vi": [], "en": [], "zh": []}}
    if rewrites:
        plan.update({k: p[k] for k in ("vi", "en", "zh", "keywords")})
    return plan


# ----------------------------------------------------------------------------- chunking
@pytest.mark.parametrize("mode", ["packed", "sentence", "paragraph", ["packed", "sentence"]])
def test_chunk_offsets_exact(mode):
    cfg = {"chunk": {"mode": mode, "target": {"vi": 300, "zh": 120, "en": 400},
                     "max": {"vi": 500, "zh": 200, "en": 600}, "min": {"vi": 60, "zh": 30, "en": 80}}}
    n = 0
    for lang, docs in DOCS.items():
        for d in docs:
            ch = chunk_doc(d, cfg)
            assert ch, d["doc_id"]
            for i, c in enumerate(ch):
                assert d["text"][c["start"]:c["end"]] == c["text"]
                assert c["chunk_id"] == f"{d['doc_id']}#{i}"
                assert c["text"] == c["text"].strip() and c["text"]
                assert len(c["text"]) <= cfg["chunk"]["max"][lang] + 1
            n += len(ch)
    assert n > 324


def test_chunk_edge_cases():
    d = {"doc_id": "1", "lang": "zh", "source": "corpus",
         "text": "  头痛怎么办？\n\n\n" + "长句子没有标点" * 60 + "\n\n结束。  "}
    for c in chunk_doc(d, {"target": {"zh": 50}, "max": {"zh": 80}, "min": {"zh": 10}}):
        assert d["text"][c["start"]:c["end"]] == c["text"]
        assert len(c["text"]) <= 80
    pm = {"doc_id": "38912345", "lang": "en", "source": "pubmed",
          "text": "Kidney stones.\n\nBACKGROUND: Stones obstruct the ureter. The level was 0.10 U/L. "
                  "METHODS: Cohort. RESULTS: Stenting helped (95% CI 1.2-2.0). CONCLUSIONS: Decompress early."}
    ch = chunk_doc(pm, {"mode": "sentence"})
    assert ch[0]["text"] == "Kidney stones."  # title chunk
    assert any(c["text"].startswith("METHODS:") for c in ch)
    assert not any(c["text"].endswith("0.") for c in ch)  # decimals not split
    assert chunk_doc({"doc_id": "x", "lang": "vi", "text": ""}, {}) == []


# ------------------------------------------------------------------------------ BM25
def test_tokenizers():
    assert "sỏi_thận" in I.tokenize("Sỏi thận gây tắc nghẽn", "vi")
    zh = I.tokenize("肾结石引起尿路梗阻怎么办", "zh")
    assert "肾结石" in zh
    en = I.tokenize("Kidney stones causing obstruction", "en")
    assert "kidney" in en and "stone" in en


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Write fixture docs as 2 parquet files and build chunks + BM25 (+ dense if CUDA)."""
    tmp = tmp_path_factory.mktemp("ix")
    ddir = tmp / "docs"
    ddir.mkdir()
    cols = ["doc_id", "source", "lang", "url", "title", "text", "fetched_at"]
    # 4 doc files (vi, zh, vi, zh halves) -> 4 manifest entries, >= 2 BM25 shards per language
    parts = [("corpus-00000", DOCS["vi"][::2]), ("corpus-00001", DOCS["zh"][::2]),
             ("corpus-00002", DOCS["vi"][1::2]), ("corpus-00003", DOCS["zh"][1::2])]
    for name, docs in parts:
        rows = [{**{k: d.get(k, "") for k in cols}, "fetched_at": "2026-10-01T00:00:00"} for d in docs]
        pq.write_table(pa.Table.from_pylist(rows), ddir / f"{name}.parquet")
    cfg = I.load_config()
    cfg["index"] = {"dir": str(tmp / "indexes"), "chunks_dir": str(tmp / "chunks"),
                    "docs_glob": str(ddir / "*.parquet"), "workers": 2,
                    "bm25": {"shard_max_chunks": 100},  # several shards per language
                    "dense": {"unit": {"vi": "chunk", "zh": "doc", "en": "chunk"}, "batch_tokens": 4096,
                              "cache_path": str(tmp / "embed.sqlite"),
                              "ivf": {"min_vectors": 10 ** 9}}}
    cfg.setdefault("retrieve", {})
    cfg["rerank"] = dict(cfg.get("rerank") or {}, cache_path=str(tmp / "rerank.sqlite"))
    man = I.build_chunks(cfg)
    assert len(man.entries) == 4
    # incremental: second call is a no-op
    man2 = I.build_chunks(cfg)
    assert len(man2.entries) == 4
    I.build_bm25(cfg, langs=["vi", "zh"])
    if HAS_CUDA:
        I.build_dense(cfg, langs=["vi", "zh"])
    return cfg, tmp


def test_chunk_store_roundtrip(built):
    cfg, _ = built
    st = I.ChunkStore(cfg)
    docs = {d["doc_id"]: d for l in ("vi", "zh") for d in DOCS[l]}
    n = st.man.n_chunks
    for g in np.linspace(0, n - 1, 25).astype(int):
        c = st.get([g])[0]
        assert docs[c["doc_id"]]["text"][c["start"]:c["end"]] == c["text"]
    dg = st.doc_gid("920005")
    first, cnt = st.doc_chunk_range(dg)
    rows = st.get(range(first, first + cnt))
    assert all(r["doc_id"] == "920005" for r in rows) and cnt >= 1


def test_bm25_global_idf_matches_single_shard(built, tmp_path):
    cfg, tmp = built
    st = I._read_json(Path(cfg["index"]["dir"]) / "bm25/zh/state.json", {})
    assert len([x for x in st["shards"] if x["dir"]]) >= 2  # multi-shard
    cfg1 = json.loads(json.dumps(cfg))
    cfg1["index"]["dir"] = str(tmp_path / "ix1")
    cfg1["index"]["chunks_dir"] = str(tmp_path / "ch1")
    cfg1["index"]["bm25"] = {"shard_max_chunks": 10 ** 7}
    I.build_chunks(cfg1, verbose=False)
    I.build_bm25(cfg1, langs=["zh"], verbose=False)
    a = I.load_bm25(cfg, "zh").search(["高血压 能根治吗"], "zh", k=20)
    b = I.load_bm25(cfg1, "zh").search(["高血压 能根治吗"], "zh", k=20)
    # IDF is global; only avgdl is per shard (chunk lengths are capped, so it barely differs)
    assert np.allclose([s for _, s in a], [s for _, s in b], rtol=0.03)
    assert [g for g, _ in a][:5] == [g for g, _ in b][:5]


def test_bm25_topic_hits(built):
    cfg, _ = built
    st = I.ChunkStore(cfg)
    for qid, p in PLANS.items():
        for lang in ("vi", "zh"):
            texts = ([query_text(qid)] if lang == "vi" else []) + p[lang] + p["keywords"][lang]
            hits = I.load_bm25(cfg, lang).search(texts, lang, k=5)
            tops = [TOPIC[st.get([g])[0]["doc_id"]] for g, _ in hits]
            assert p["topic"] in tops[:3], (qid, lang, tops)


def test_dense_ivf_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    cfg = {"index": {"dir": str(tmp_path), "dense": {"unit": {"zh": "chunk"},
                     "ivf": {"min_vectors": 1, "factory": "IVF4,Flat", "train_size": 1000,
                             "nprobe": 4, "keep_fp16": False}}}}
    d = tmp_path / "dense/chunk_zh"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    np.save(d / "e00000.f16.npy", X[:300].astype(np.float16))
    np.save(d / "e00000.ids.npy", np.arange(300, dtype=np.int64))
    np.save(d / "e00001.f16.npy", X[300:].astype(np.float16))
    np.save(d / "e00001.ids.npy", np.arange(1000, 1300, dtype=np.int64))
    I._write_json(d / "state.json", {"files": {"0": 300, "1": 300}, "ivf_added": []})
    bf = I.load_dense(cfg, "zh").search(X[[5, 400]], k=3)
    I.build_ivf(cfg, "zh", verbose=False)
    assert not (d / "e00000.f16.npy").exists()
    di = I.load_dense(cfg, "zh")
    assert di.ivf is not None and len(di) == 600 and not di.parts
    iv = di.search(X[[5, 400]], k=3)
    assert {g for g, _ in iv[:2]} == {5, 1100} == {g for g, _ in bf[:2]}


# ------------------------------------------------------------------- retrieval (GPU)
gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")


@gpu
def test_retrieve_cross_lingual(built):
    cfg, _ = built
    from src.retrieve import Retriever
    R = Retriever(cfg)
    en_by_topic = {}
    for d in DOCS["en"]:
        en_by_topic.setdefault(d["topic"], []).append(d)
    hits_lang = {"vi": 0, "zh": 0, "en": 0}
    for qid, p in PLANS.items():
        # extra docs = the topic's 4 abstracts + 16 distractor abstracts (PubMed-like candidate pool)
        extra = en_by_topic[p["topic"]] + [d for d in DOCS["en"] if d["topic"] != p["topic"]][qid % 50: qid % 50 + 16]
        res, dbg = R.retrieve(make_plan(qid), k=50, extra_docs=extra, return_debug=True)
        assert res and all(c["stage"] == "rrf" for c in res)
        for c in res:
            assert set(c) >= {"chunk_id", "doc_id", "lang", "source", "text", "score", "stage"}
        for lang in hits_lang:
            top = [TOPIC[c["doc_id"]] for c in res if c["lang"] == lang][:5]
            hits_lang[lang] += p["topic"] in top
        # pure cross-lingual dense: no rewrites, Vietnamese query only -> zh doc-level dense list
        res2, dbg2 = R.retrieve(make_plan(qid, rewrites=False), k=200, return_debug=True)
        zh_dense = [c for c in res2 if "dense_doc:zh" in c["ranks"]]
        zh_dense.sort(key=lambda c: c["ranks"]["dense_doc:zh"])
        top_zh = [TOPIC[c["doc_id"]] for c in zh_dense[:10]]
        assert p["topic"] in top_zh, (qid, top_zh)
    print("topic hit@5 per language:", hits_lang, "timings:", dbg["timings"])
    assert hits_lang["vi"] >= 5 and hits_lang["zh"] >= 5 and hits_lang["en"] >= 5


@gpu
def test_rerank_calibrated(built):
    cfg, _ = built
    from src.retrieve import Retriever
    from src.rerank import rerank
    R = Retriever(cfg)
    ok = 0
    for qid, p in PLANS.items():
        plan = make_plan(qid)
        cands = R.retrieve(plan, k=30, extra_docs=[d for d in DOCS["en"] if d["topic"] == p["topic"]])
        rr = rerank(plan["query_vi"], cands, top_n=10, cfg=cfg)
        assert len(rr) == min(10, len(cands))
        assert all(0.0 <= c["score"] <= 1.0 and c["stage"] == "rerank" for c in rr)
        assert all(rr[i]["score"] >= rr[i + 1]["score"] for i in range(len(rr) - 1))
        ok += TOPIC[rr[0]["doc_id"]] == p["topic"]
        # cache hit gives identical scores
        rr2 = rerank(plan["query_vi"], cands, top_n=10, cfg=cfg)
        assert [c["chunk_id"] for c in rr2] == [c["chunk_id"] for c in rr]
    assert ok >= len(PLANS) - 1
    from src.embed import release_encoder
    from src.rerank import release_reranker
    release_reranker()
    release_encoder()
