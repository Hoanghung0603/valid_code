"""Hybrid multilingual retrieval: BM25 per language + dense (BGE-M3) + RRF -> Candidates.

retrieve(plan, k=200, extra_docs=None, cfg=None) -> list[Candidate]

Lists fused with weighted RRF (score = sum_l w_l / (rrf_k + rank_l)):
  bm25:<lang>        on-disk BM25 of corpus chunks with the matching-language texts of the plan
                     (vi: query_vi + vi rewrites + keywords.vi; zh: zh rewrites + keywords.zh; en: en ...)
  dense:<lang>       chunk-level dense (if index.dense.unit[lang] == chunk); query vectors =
                     query_vi + all vi/en/zh rewrites, score = max cosine over query vectors
  dense_doc:<lang>   doc-level dense (unit == doc): top docs -> their chunks (plus chunks of docs hit
                     by BM25) are embedded on the fly (cached) and ranked by max cosine
  extra:bm25 / extra:dense
                     on-the-fly docs (e.g. PubMed candidates of this query) chunked + indexed in memory
Candidate = {chunk_id, doc_id, lang, source, text, score, stage="rrf", start, end, ranks{list: rank}}
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from src.chunk import chunk_doc
from src.index import (LANGS, ChunkStore, _unit, build_bm25_memory, icfg, load_bm25, load_config, load_dense)

DEFAULT_RCFG = {
    "k_bm25": 200,           # per list
    "k_dense": 200,
    "rrf_k": 60,
    "max_rewrites": 4,       # per language, for BM25 and dense query texts
    "keyword_weight": 1.0,   # BM25 weight of keyword strings relative to full rewrites
    "dense_doc_top": 30,     # doc-level dense: docs to expand into chunks
    "expand_bm25_docs": 10,  # also expand the top BM25 docs of doc-level-dense languages
    "expand_max_chunks_per_doc": 8,
    "expand_mode": "embed",  # embed: dense-score expanded chunks on the fly (cached) | inherit: doc rank order
    "weights": {"bm25": 1.0, "dense": 1.0, "dense_doc": 1.0, "extra_bm25": 1.0, "extra_dense": 1.0},
    "langs": ["vi", "zh", "en"],
    "extra_max_chunks": 3000,
}


def rcfg(cfg: dict | None) -> dict:
    r = dict(DEFAULT_RCFG)
    r.update((cfg or {}).get("retrieve") or {})
    w = dict(DEFAULT_RCFG["weights"])
    w.update(r.get("weights") or {})
    r["weights"] = w
    return r


def _uniq(xs):
    out, seen = [], set()
    for x in xs:
        if isinstance(x, str):
            x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def plan_texts(plan: dict, lang: str, m: int) -> list[str]:
    base = list(plan.get(lang) or [])[:m]
    if lang == "vi" and plan.get("query_vi"):
        base = [plan["query_vi"]] + base
    return _uniq(base)


def plan_keywords(plan: dict, lang: str) -> list[str]:
    kw = (plan.get("keywords") or {}).get(lang) or []
    return _uniq(kw)


def dense_query_texts(plan: dict, m: int) -> list[str]:
    q = [plan.get("query_vi") or ""]
    for lang in ("vi", "en", "zh"):
        q += list(plan.get(lang) or [])[:m]
    return _uniq(q)


class Retriever:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self.ix = icfg(self.cfg)
        self.r = rcfg(self.cfg)
        self.store = ChunkStore(self.cfg)
        self.bm25 = {l: load_bm25(self.cfg, l) for l in LANGS}
        self.dense = {l: load_dense(self.cfg, l) for l in LANGS}
        self._enc = None

    @property
    def enc(self):
        if self._enc is None:
            from src.embed import get_encoder
            self._enc = get_encoder(self.cfg)
        return self._enc

    # --------------------------------------------------------------------------------
    def _bm25_lists(self, plan, timings):
        lists = {}
        m = self.r["max_rewrites"]
        for lang in self.r["langs"]:
            idx = self.bm25.get(lang)
            if idx is None:
                continue
            texts = plan_texts(plan, lang, m)
            kws = plan_keywords(plan, lang)
            if not texts and not kws:
                continue
            t0 = time.time()
            weights = [1.0] * len(texts) + [float(self.r["keyword_weight"])] * len(kws)
            hits = idx.search(texts + kws, lang, k=int(self.r["k_bm25"]), weights=weights)
            lists[f"bm25:{lang}"] = [(int(g), s) for g, s in hits if self.store.is_active_chunk(g)]
            timings[f"bm25:{lang}"] = time.time() - t0
        return lists

    def _expand_docs(self, doc_gids, max_per_doc):
        gids = []
        for dg in doc_gids:
            first, n = self.store.doc_chunk_range(int(dg))
            gids.extend(range(first, first + min(n, max_per_doc)))
        return gids

    def _dense_lists(self, plan, Q, bm25_lists, timings):
        lists = {}
        for lang in self.r["langs"]:
            di = self.dense.get(lang)
            if di is None:
                continue
            t0 = time.time()
            unit = _unit(self.ix, lang)
            hits = di.search(Q, k=int(self.r["k_dense"]) if unit == "chunk" else int(self.r["dense_doc_top"]))
            if unit == "chunk":
                lists[f"dense:{lang}"] = [(int(g), s) for g, s in hits if self.store.is_active_chunk(g)]
            else:
                doc_gids = [int(g) for g, _ in hits]
                # docs hit by BM25 for this language also get their chunks dense-scored
                seen = set(doc_gids)
                nb = 0
                for g, _ in bm25_lists.get(f"bm25:{lang}", []):
                    if nb >= int(self.r["expand_bm25_docs"]):
                        break
                    c = self.store.get([g])[0]
                    dg = self.store.doc_gid(c["doc_id"], c["source"])
                    if dg is not None and dg not in seen:
                        seen.add(dg)
                        doc_gids.append(dg)
                        nb += 1
                cg = self._expand_docs(doc_gids, int(self.r["expand_max_chunks_per_doc"]))
                if self.r["expand_mode"] == "inherit":
                    lists[f"dense_doc:{lang}"] = [(int(g), 0.0) for g in cg]
                    timings[f"dense:{lang}"] = time.time() - t0
                    continue
                rows = self.store.get(cg)
                if rows:
                    V = self.enc.encode_cached([r["text"] for r in rows]).astype(np.float32)
                    sc = (V @ Q.T).max(axis=1)
                    o = np.argsort(-sc)
                    lists[f"dense_doc:{lang}"] = [(int(cg[i]), float(sc[i])) for i in o]
            timings[f"dense:{lang}"] = time.time() - t0
        return lists

    def _extra_lists(self, plan, Q, extra_docs, timings):
        """On-the-fly docs (PubMed): chunk -> in-memory BM25 (per lang) + dense (cached)."""
        t0 = time.time()
        chunks = []
        for d in extra_docs:
            chunks.extend(chunk_doc(d, self.cfg))
        chunks = chunks[: int(self.r["extra_max_chunks"])]
        if not chunks:
            return {}, []
        lists = {}
        m = self.r["max_rewrites"]
        bm = []
        for lang in sorted({c["lang"] for c in chunks}):
            idx_rows = [i for i, c in enumerate(chunks) if c["lang"] == lang]
            texts = plan_texts(plan, lang, m) + plan_keywords(plan, lang)
            if not texts and lang != "vi":
                texts = plan_texts(plan, "en", m) + plan_keywords(plan, "en")
            if not texts:
                continue
            mi = build_bm25_memory([chunks[i]["text"] for i in idx_rows], lang, self.ix["bm25"])
            for g, s in mi.search(texts, lang, k=len(idx_rows)):
                bm.append((idx_rows[g], s))
        bm.sort(key=lambda x: -x[1])
        lists["extra:bm25"] = [(-1 - i, s) for i, s in bm]  # negative ids = extra chunks
        V = self.enc.encode_cached([c["text"] for c in chunks]).astype(np.float32)
        sc = (V @ Q.T).max(axis=1)
        o = np.argsort(-sc)
        lists["extra:dense"] = [(-1 - int(i), float(sc[i])) for i in o]
        timings["extra"] = time.time() - t0
        return lists, chunks

    # --------------------------------------------------------------------------------
    def retrieve(self, plan: dict, k: int = 200, extra_docs: list[dict] | None = None,
                 return_debug: bool = False):
        timings: dict[str, float] = {}
        lists = self._bm25_lists(plan, timings)
        need_dense = any(self.dense.get(l) is not None for l in self.r["langs"]) or extra_docs
        Q = None
        if need_dense:
            t0 = time.time()
            Q = self.enc.encode(dense_query_texts(plan, self.r["max_rewrites"]), is_query=True).astype(np.float32)
            timings["qenc"] = time.time() - t0
            lists.update(self._dense_lists(plan, Q, lists, timings))
        extra_chunks = []
        if extra_docs:
            el, extra_chunks = self._extra_lists(plan, Q, extra_docs, timings)
            lists.update(el)
        # weighted RRF over item ids (gid >= 0 corpus chunk, < 0 extra chunk); key by chunk_id
        t0 = time.time()
        rrf_k = float(self.r["rrf_k"])
        W = self.r["weights"]
        fused = defaultdict(float)
        ranks = defaultdict(dict)
        info = {}
        need_rows = set()
        def _w(name):
            wkey = "extra_" + name.split(":")[1] if name.startswith("extra:") else name.split(":")[0]
            return float(W.get(name, W.get(wkey, 1.0)))

        for name, hits in lists.items():
            w = _w(name)
            for rank, (iid, _s) in enumerate(hits, start=1):
                if iid < 0:
                    c = extra_chunks[-1 - iid]
                    key = c["chunk_id"]
                    info.setdefault(key, c)
                else:
                    key = ("g", iid)
                    need_rows.add(iid)
                if name not in ranks[key]:
                    ranks[key][name] = rank
                    fused[key] += w / (rrf_k + rank)
        # resolve corpus gids -> chunk rows; merge with extra chunks of the same chunk_id
        top = sorted(fused.items(), key=lambda x: -x[1])
        # only fetch rows we may output (a bit more than k to absorb chunk_id collisions)
        keep = [key for key, _ in top[: int(k * 1.5) + 10]]
        rows = {r["gid"]: r for r in self.store.get(sorted({key[1] for key in keep if isinstance(key, tuple)}))}
        out: dict[str, dict] = {}
        for key in keep:
            if isinstance(key, tuple):
                c = rows[key[1]]
            else:
                c = info[key]
            cid = c["chunk_id"]
            if cid in out:  # same chunk from corpus index and extra docs
                o = out[cid]
                for n_, r_ in ranks[key].items():
                    if n_ not in o["ranks"]:
                        o["ranks"][n_] = r_
                        o["score"] += _w(n_) / (rrf_k + r_)
                continue
            out[cid] = {"chunk_id": cid, "doc_id": c["doc_id"], "lang": c["lang"], "source": c["source"],
                        "text": c["text"], "score": float(fused[key]), "stage": "rrf",
                        "start": int(c["start"]), "end": int(c["end"]), "ranks": dict(ranks[key])}
        res = sorted(out.values(), key=lambda x: -x["score"])[:k]
        timings["fuse"] = time.time() - t0
        if return_debug:
            return res, {"lists": {n: len(h) for n, h in lists.items()}, "timings": timings}
        return res


_RETRIEVER: Retriever | None = None


def get_retriever(cfg: dict | None = None, reload: bool = False) -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None or reload or (cfg is not None and cfg is not _RETRIEVER.cfg):
        _RETRIEVER = Retriever(cfg)
    return _RETRIEVER


def retrieve(plan: dict, k: int = 200, extra_docs: list[dict] | None = None, cfg: dict | None = None) -> list[dict]:
    """Interface function (docs/INTERFACES.md). `extra_docs`: Doc dicts fetched for this query
    (e.g. src.pubmed.candidates_for_plan(plan)) that are chunked/embedded in memory."""
    return get_retriever(cfg).retrieve(plan, k=k, extra_docs=extra_docs)
