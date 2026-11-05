"""Retrieval bench: does a channel bring gold docs into the top-100 / top-200? (doc level, Vietnamese docs)

  python scripts/analysis/retrieval_bench.py channels   # build per-channel doc rankings -> data/dev/bench/*.parquet
  python scripts/analysis/retrieval_bench.py eval       # recall@100/@200 of channels and fusions
Channels (top-300 docs each, exact cosine over all Vietnamese docs for the dense doc-level ones):
  chunk        current first stage: doc-level RRF of the bm25_vi / dense_vi chunk ranks (pool-vi-m1000)
  rerank       current final order (rerank + 0.1 RRF), for reference
  head_q / mean_q        original query vs doc head (first chunk) / doc-mean vector
  sub_head / sub_mean    each LLM sub-question (qstruct-v2) vs head / mean, RRF over sub-questions
  title_head / title_mean  each generated title vs head / mean, RRF over titles
  title_<model>  original query vs the doc title embedded alone by a probe model (python ... titles --models ...)
  card_q / card_ent  original query / LLM entities + synonyms + keywords vs the topic-card BM25 index
  (python scripts/analysis/retrieval_bench.py cards, after scripts/analysis/extract_metadata.py)
Metrics: proxy recall (gold_proxy.py: expected gold mass, LOO-checked against the LB) and, on the 30
Claude-judged queries, recall of grade-2 docs (unjudged docs count as misses: `judged` shows the coverage).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from medret.config import get_config
from medret.index.meta import DB_FILE, MetaStore

TOP = 300
BENCH = "bench"


def _doc_matrices(cfg, meta):
    """Vietnamese doc ids with their head vectors and doc-mean vectors: float16 .npy files on disk, opened as
    memmaps (written once), so the bench holds only one block of rows in RAM at a time (the laptop has ~3-4 GB
    free with the IDE and browser open)."""
    dense_dir = cfg.paths.index_dir / "dense" / cfg.index.dense_run
    hpath, mpath, ipath = dense_dir / "doc_head_vi.f16.npy", dense_dir / "doc_mean_vi.f16.npy", dense_dir / "doc_vi_ids.npy"
    if not hpath.exists():
        if not mpath.exists():  # older doc_vectors.py output
            m = np.load(dense_dir / "doc_mean_vi.npz")
            np.save(ipath, m["doc_id"])
            mm = np.lib.format.open_memmap(mpath, mode="w+", dtype=np.float16, shape=m["vec"].shape)
            mm[:] = m["vec"]
            mm.flush()
            del mm, m
        docs = np.load(ipath)
        heads = meta.con.execute("SELECT doc_id, input_id FROM chunks WHERE chunk_id % 10000 = 0 "
                                 "AND input_id IS NOT NULL").df().set_index("doc_id")["input_id"]
        hid = pd.Series(docs).map(heads)
        vec = np.load(dense_dir / "vectors.f16.npy", mmap_mode="r")
        H = np.lib.format.open_memmap(hpath, mode="w+", dtype=np.float16, shape=(len(docs), vec.shape[1]))
        ok = np.flatnonzero(hid.notna().to_numpy())
        ids = hid.to_numpy()[ok].astype(np.int64)
        order = np.argsort(ids)
        for s in range(0, len(ids), 100_000):
            sel = order[s: s + 100_000]
            H[ok[sel]] = vec[ids[sel]]
        H.flush()
        del H
    return np.load(ipath), np.load(hpath, mmap_mode="r"), np.load(mpath, mmap_mode="r")


def _topk(V: np.ndarray, D: np.ndarray, k: int = TOP, doc_block: int = 50_000, q_block: int = 256
          ) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product; D (float16) is converted to float32 one block of rows at a time, so the
    extra memory is ~doc_block x dim x 4 bytes plus q_block x doc_block scores (~0.5 GB)."""
    idx = np.empty((len(V), k), dtype=np.int64)
    sc = np.empty((len(V), k), dtype=np.float32)
    for qs in range(0, len(V), q_block):
        Vq = V[qs: qs + q_block].astype(np.float32)
        best_i = np.empty((len(Vq), 0), dtype=np.int64)
        best_s = np.empty((len(Vq), 0), dtype=np.float32)
        for ds in range(0, len(D), doc_block):
            sims = Vq @ D[ds: ds + doc_block].astype(np.float32).T
            kk = min(k, sims.shape[1] - 1)
            top = np.argpartition(-sims, kk, axis=1)[:, :kk]
            best_i = np.concatenate([best_i, top + ds], axis=1)
            best_s = np.concatenate([best_s, np.take_along_axis(sims, top, axis=1)], axis=1)
            keep = np.argpartition(-best_s, min(k, best_s.shape[1] - 1), axis=1)[:, :k]
            best_i, best_s = np.take_along_axis(best_i, keep, axis=1), np.take_along_axis(best_s, keep, axis=1)
        o = np.argsort(-best_s, axis=1)
        idx[qs: qs + q_block] = np.take_along_axis(best_i, o, axis=1)
        sc[qs: qs + q_block] = np.take_along_axis(best_s, o, axis=1)
    return idx, sc


def _rrf(frames: list[pd.DataFrame], k: int = 60) -> pd.DataFrame:
    """frames: query_id, doc_id, rank -> fused query_id, doc_id, rank (top TOP)."""
    f = pd.concat(frames)
    f = f.assign(s=1 / (k + f["rank"] + 1)).groupby(["query_id", "doc_id"])["s"].sum().reset_index()
    f = f.sort_values(["query_id", "s"], ascending=[True, False])
    f["rank"] = f.groupby("query_id").cumcount()
    return f[f["rank"] < TOP][["query_id", "doc_id", "rank"]]


def build_channels() -> None:
    from medret.retrieve.query_encoder import QueryEncoder
    from medret.select.selector import SelectConfig, choose_docs

    cfg = get_config()
    out = cfg.paths.dev_dir / BENCH
    out.mkdir(parents=True, exist_ok=True)
    meta = MetaStore(cfg.paths.index_dir / DB_FILE)
    src = cfg.paths.runs_dir / "pool-vi-m1000"
    # current pipeline: first stage (chunk RRF) and final order (rerank)
    c = pd.read_parquet(src / "candidates.parquet", columns=["query_id", "chunk_id", "rrf"])
    c["doc_id"] = c["chunk_id"] // 10_000
    vi = set(meta.con.execute("SELECT doc_id FROM documents WHERE lang = 'vi'").df()["doc_id"])
    c = c[c["doc_id"].isin(vi)].groupby(["query_id", "doc_id"])["rrf"].max().reset_index()
    c = c.sort_values(["query_id", "rrf"], ascending=[True, False])
    c["rank"] = c.groupby("query_id").cumcount()
    c[c["rank"] < TOP][["query_id", "doc_id", "rank"]].to_parquet(out / "chunk.parquet", index=False)
    rr = pd.read_parquet(src / "reranked.parquet", columns=["query_id", "doc_id", "lang", "score", "rrf"])
    rows = [(q, d, r) for q, g in rr.groupby("query_id")
            for r, d in enumerate(choose_docs(g, SelectConfig(k_max=TOP, tau_doc=-1.0, max_zh_docs=0)))]
    pd.DataFrame(rows, columns=["query_id", "doc_id", "rank"]).to_parquet(out / "rerank.parquet", index=False)
    import gc

    del c, rr, rows, vi
    gc.collect()
    # texts to encode: original queries, sub-questions, titles
    qs = pd.read_parquet(cfg.paths.runs_dir / "qstruct-v2" / "qstruct.parquet")
    texts = []  # (query_id, kind, text)
    for r in qs.itertuples():
        texts.append((r.query_id, "q", r.query))
        s = json.loads(r.qstruct)
        for sq in s["sub_questions"]:
            texts.append((r.query_id, "sub", sq["question"]))
            texts += [(r.query_id, "title", t) for t in sq["titles"]]
    t = pd.DataFrame(texts, columns=["query_id", "kind", "text"])
    enc = QueryEncoder(cfg.embed.model, cache_path=cfg.paths.data_dir / "cache" / "qvec.sqlite", dtype="float32",
                       threads=4)  # laptop budget: a quarter of the cores (bf16 on this CPU is emulated: far slower)
    V = enc.encode(t["text"].tolist()).astype(np.float32)
    print(f"encoded {len(t):,} texts: {t['kind'].value_counts().to_dict()}", flush=True)
    del enc  # the CPU model holds ~2.3 GB: release it before the doc matrices are scanned
    gc.collect()
    docs, H, M = _doc_matrices(cfg, meta)
    for target, D in (("head", H), ("mean", M)):
        idx, _ = _topk(V, D)
        long = pd.DataFrame({"query_id": np.repeat(t["query_id"].to_numpy(), TOP),
                             "kind": np.repeat(t["kind"].to_numpy(), TOP),
                             "item": np.repeat(np.arange(len(t)), TOP),
                             "doc_id": docs[idx.ravel()], "rank": np.tile(np.arange(TOP), len(t))})
        for kind, name in (("q", f"{target}_q"), ("sub", f"sub_{target}"), ("title", f"title_{target}")):
            part = long[long["kind"].eq(kind)]
            res = part[["query_id", "doc_id", "rank"]] if kind == "q" else _rrf([g for _, g in part.groupby("item")])
            res.to_parquet(out / f"{name}.parquet", index=False)
            print(f"channel {name}: {len(res):,} rows", flush=True)


def build_card_channels() -> None:
    """Topic cards (metadata) -> BM25 index -> channels card_q (original query) and card_ent (LLM entities,
    synonyms, keywords). Pruning as the chunk BM25: terms in > 10% of cards and stopwords are dropped."""
    from medret.index.cards import CARD_FILE, build_card_bm25, build_cards, card_terms
    from medret.index.lexical import BM25Index
    from medret.retrieve.pipeline import STOP_VI

    cfg = get_config()
    out = cfg.paths.dev_dir / BENCH
    meta = MetaStore(cfg.paths.index_dir / DB_FILE)
    idx_dir = cfg.paths.index_dir / "bm25" / "card_vi"
    if not (idx_dir / "medret_bm25.json").exists():
        docs = meta.con.execute("SELECT doc_id, title FROM documents WHERE lang = 'vi'").df()
        md = pd.read_parquet(cfg.paths.parsed_dir / "metadata" / "vi.parquet")
        cards = build_cards(docs, md, cfg.paths.index_dir / CARD_FILE)
        build_card_bm25(cards, idx_dir)
    bm = BM25Index(idx_dir)
    qs = pd.read_parquet(cfg.paths.runs_dir / "qstruct-v2" / "qstruct.parquet")
    rows_q, rows_e = [], []
    for r in qs.itertuples():
        s = json.loads(r.qstruct)
        ent = " ".join([e["name"] + " " + " ".join(e.get("synonyms", [])) for e in s["entities"]] + [s["keywords"]])
        for text, rows in ((r.query, rows_q), (ent, rows_e)):
            terms = bm.prune(card_terms(text), 0.1, STOP_VI) or card_terms(text)
            hits = bm.search_terms(terms, TOP)
            rows += [(r.query_id, cid // 10_000, k) for k, (cid, _) in enumerate(hits)]
    for name, rows in (("card_q", rows_q), ("card_ent", rows_e)):
        pd.DataFrame(rows, columns=["query_id", "doc_id", "rank"]).to_parquet(out / f"{name}.parquet", index=False)
        print(f"channel {name}: {len(rows):,} rows")


def build_title_channels(models: list[str]) -> None:
    """Original query vs doc title (probe embeddings, data/embeddings/probe-<tag>/all.f16.npy; built by
    probe_eval.py) -> channel title_<tag>: exact top-300 over the Vietnamese docs."""
    import glob

    cfg = get_config()
    out = cfg.paths.dev_dir / BENCH
    fmts = {"bgem3-title": 0, "e5-large": 1, "e5-large-instruct": 2, "qwen3-emb-0.6b": 3, "qwen3-emb-4b": 3,
            "e5i-head": 2, "aitvn-head": 0}
    qids = pd.read_parquet(cfg.paths.runs_dir / "pool-vi-m1000" / "queries.parquet")["id"].to_numpy()
    for tag in models:
        d = cfg.paths.data_dir / "embeddings" / f"probe-{tag}"
        ids = np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "shard-*.ids.npy")))])
        if not (d / "all.f16.npy").exists():  # shards concatenated in the same sorted order as the ids
            files = sorted(glob.glob(str(d / "shard-*.vec.npy")))
            first = np.load(files[0], mmap_mode="r")
            allv = np.lib.format.open_memmap(d / "all.f16.npy", mode="w+", dtype=np.float16,
                                             shape=(len(ids), first.shape[1]))
            pos = 0
            for f in files:
                v = np.load(f, mmap_mode="r")
                allv[pos: pos + len(v)] = v
                pos += len(v)
            allv.flush()
            del allv
        vec = np.load(d / "all.f16.npy", mmap_mode="r")
        doc_rows = np.flatnonzero(ids >= 0)  # query rows have negative ids
        assert (np.diff(doc_rows) == 1).all(), "doc rows are expected to be contiguous"
        D = vec[doc_rows[0]: doc_rows[-1] + 1]
        row = pd.Series(np.arange(len(ids)), index=ids)
        Q = np.stack([vec[row[-(fmts[tag] * 10_000 + int(q))]].astype(np.float32) for q in qids])
        idx, _ = _topk(Q, D)
        docs = ids[doc_rows[0]: doc_rows[-1] + 1]
        res = pd.DataFrame({"query_id": np.repeat(qids, TOP), "doc_id": docs[idx.ravel()],
                            "rank": np.tile(np.arange(TOP), len(qids))})
        res.to_parquet(out / f"title_{tag}.parquet", index=False)
        print(f"channel title_{tag}: {len(res):,} rows", flush=True)


def evaluate_channels() -> None:
    cfg = get_config()
    out = cfg.paths.dev_dir / BENCH
    u = pd.read_parquet(cfg.paths.dev_dir / "proxy_pairs_scored.parquet")
    u = u[u["lang"].eq("vi")]
    g = u.set_index(["query_id", "doc_id"])["g"]
    tot = u.groupby("query_id")["g"].sum()  # fitted c = 1: the universe holds the expected gold
    j = pd.read_parquet(cfg.paths.dev_dir / "claude_v1_judged.parquet")
    lab = j.set_index(["query_id", "doc_id"])["grade"]
    pos = j[j["grade"] == 2].groupby("query_id").size()
    ch = {p.stem: pd.read_parquet(p) for p in sorted(out.glob("*.parquet"))}
    fusions = {
        "chunk+head_q": ["chunk", "head_q"],
        "chunk+head_q+mean_q": ["chunk", "head_q", "mean_q"],
        "chunk+sub_head+sub_mean": ["chunk", "sub_head", "sub_mean"],
        "chunk+title_head+title_mean": ["chunk", "title_head", "title_mean"],
        "chunk+head_q+mean_q+sub+title": ["chunk", "head_q", "mean_q", "sub_head", "sub_mean", "title_head",
                                          "title_mean"],
    }
    for tc in [c for c in ch if c.startswith("title_") and c not in ("title_head", "title_mean")]:
        fusions[f"chunk+head_q+mean_q+{tc}"] = ["chunk", "head_q", "mean_q", tc]
    if "card_q" in ch:
        fusions["chunk+card_q+card_ent"] = ["chunk", "card_q", "card_ent"]
        fusions["all+cards"] = fusions["chunk+head_q+mean_q+sub+title"] + ["card_q", "card_ent"]
    for name, parts in fusions.items():
        if all(p in ch for p in parts):
            ch[name] = _rrf([ch[p] for p in parts])
    print(f"{'channel':34s} proxyR@100 proxyR@200 | claudeR@100 claudeR@200 judged@200")
    for name, df in ch.items():
        row = []
        for k in (100, 200):
            top = df[df["rank"] < k]
            key = pd.MultiIndex.from_frame(top[["query_id", "doc_id"]])
            hit = pd.Series(g.reindex(key).fillna(0).to_numpy(), index=top["query_id"]).groupby(level=0).sum()
            row.append((hit / tot.reindex(hit.index)).mean())
        for k in (100, 200):
            top = df[(df["rank"] < k) & df["query_id"].isin(pos.index)]
            key = pd.MultiIndex.from_frame(top[["query_id", "doc_id"]])
            gr = pd.Series(lab.reindex(key).to_numpy(), index=top["query_id"])
            row.append(((gr == 2).groupby(level=0).sum() / pos).mean())
            if k == 200:
                row.append(gr.notna().mean())
        print(f"{name:34s} {row[0]:9.3f} {row[1]:10.3f} | {row[2]:10.3f} {row[3]:11.3f} {row[4]:9.2f}")


def main() -> None:
    from medret.system import lower_priority

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["channels", "cards", "titles", "eval"])
    ap.add_argument("--models", nargs="*", default=["e5-large-instruct", "bgem3-title"])
    a = ap.parse_args()
    lower_priority()
    if a.cmd == "titles":
        build_title_channels(a.models)
    else:
        {"channels": build_channels, "cards": build_card_channels, "eval": evaluate_channels}[a.cmd]()


if __name__ == "__main__":
    main()
