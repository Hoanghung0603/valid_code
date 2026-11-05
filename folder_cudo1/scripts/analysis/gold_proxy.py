"""Gold proxy: P(doc is gold | ranking features), fitted to the leaderboard P / R of doc sets we submitted.

  python scripts/analysis/gold_proxy.py features   # -> data/dev/proxy_pairs.parquet (query, doc, features)
  python scripts/analysis/gold_proxy.py fit        # fit + leave-one-set-out check + rank the strategies
Why: the LLM-judged dev set measures relevance, but the gold is a narrower notion. The LB-to-dev precision ratio
runs from 0.43 (LLM-judged docs) to 0.83 (top docs by query-vs-head cosine), so relevance alone cannot rank
strategies. Every submitted doc set gives an aggregate label (its LB precision, often its recall too); a small
logistic model over per-doc features is fitted to those aggregates ("learning from label proportions").
Universe per query: docs of the deep run's candidates (bm25_vi / dense_vi routes, rerank scores) + head top-300.
Recall: R_q = sum_top g / (c * sum_universe g), c >= 1 for gold outside the universe (fitted).
fit --mq-features adds the ranks of the multi-query channels (LLM sub-questions and generated titles vs doc head /
doc mean, retrieval_bench.py) as features: their fitted weight says whether query variants carry gold signal.
The 0.3241 submission used the 6-feature fit on the first 13 sets: data/dev/proxy_v1_20261019/.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from medret.chunk.run import CHUNK_ID_STRIDE
from medret.config import get_config
from medret.index.meta import DB_FILE, MetaStore
from medret.submit import read_submission

RUN = "pool-vi-m1000"
# doc sets with leaderboard numbers: name -> (run, k, LB P, LB R or None, note)
LB_SETS = {
    "rerank@40": ("sub-vi200-deep-full40-v1", 40, 0.2826, None, "chunk P of whole-doc chunks = doc P@40"),
    "rerank@65": ("sub-vi200-deep-full65-lzma-v1", 65, 0.2438, None, "chunk P, top-65 whole docs"),
    "rerank@200": ("sub-vi200-deep-full40-v1", 200, 0.1494, 0.4788, "doc P / R"),
    "head@40": ("sub-vi100-head-full40-v1", 40, 0.3027, None, "chunk P"),
    "head@100": ("sub-vi100-head-full40-v1", 100, 0.2205, 0.3518, "doc P / R"),
    "dense@40": ("probe-densevi-k40-v1", 40, 0.2972, 0.2210, "doc P / R (7% zh docs count as misses)"),
    "bm25@40": ("probe-bm25vi-k40-v1", 40, 0.1559, 0.1332, "doc P / R"),
    "D2@150": ("sub-vi150-v1", 150, 0.1725, 0.4267, "older pool (rerank 400)"),
    "D1@200": ("sub-vi200-v1", 200, 0.1460, 0.4577, "older pool"),
    "rq-chm@65": ("sub-vi200-rq-chunkheadmean-v1", 65, 0.2958, None, "chunk P, RRF chunk+head+mean"),
    "rq-chm@200": ("sub-vi200-rq-chunkheadmean-v1", 200, 0.1698, 0.5219, "doc P / R"),
    "rq-all@65": ("sub-vi200-rq-all-v1", 65, 0.2856, None, "chunk P, + sub-questions + titles"),
    "rq-all@200": ("sub-vi200-rq-all-v1", 200, 0.1672, 0.5221, "doc P / R"),
    # added 2026-10-19 (sets submitted 14-19/10); "chunks": the set is the docs of relevant_chunks (whole-doc chunks)
    "wfuse@65": ("sub-vi200-wfuse-title-v1", 65, 0.2774, None, "chunk P, title-weighted fusion"),
    "wfuse@200": ("sub-vi200-wfuse-title-v1", 200, 0.1557, 0.4808, "doc P / R"),
    "bm25doc@40": ("probe-bm25doc-bodylucene-k40-v1", 40, 0.1472, 0.1264, "doc P / R, doc-level BM25"),
    "prf@65": ("sub-vi200-prf-m10a1-full65-v1", 65, 0.2876, None, "chunk P, Rocchio PRF order"),
    "prf@200": ("sub-vi200-prf-m10a1-full65-v1", 200, 0.1641, 0.5000, "doc P / R"),
    "adaptk@k": ("sub-vi-adaptk-full65-v1", 200, 0.2180, 0.4803, "doc P / R, k per query (lists <= 200)"),
    "gorder@k": ("sub-vi-gorder-adaptk-full65-v1", 200, 0.2217, 0.4990, "doc P / R, proxy-g order, k per query"),
    "gorder@c65": ("sub-vi-gorder-adaptk-full65-v1", 65, 0.3006, None, "chunk P", "chunks"),
    # added 2026-10-20: calibration sets of new signals and the +41k-doc run
    "e5head@200": ("sub-vi200-e5head-full65-v1", 200, 0.1369, 0.4208, "doc P / R, e5 doc head only"),
    "e5head@65": ("sub-vi200-e5head-full65-v1", 65, 0.2324, None, "chunk P"),
    "tmmean@200": ("sub-vi200-tmmean-full65-v1", 200, 0.2035, 0.4096, "doc P / R, teammate doc mean only"),
    "tmmean@65": ("sub-vi200-tmmean-full65-v1", 65, 0.2489, None, "chunk P"),
    "docs41k@k": ("sub-vi-gorder-adaptkn18-docs41k-v1", 200, 0.2213, 0.5089, "doc P / R, +41k docs"),
}
N_SETS_V1 = 13  # the sets of the 6-feature fit behind the 0.3156 / 0.3241 submissions
MQ_CHANNELS = {"sub": ("sub_head", "sub_mean"), "gtitle": ("title_head", "title_mean")}
FEATURES = ["head_rel", "mean_rel", "dense_rr", "bm25_rr", "rerank", "in_rerank"]


def load_sets(cfg) -> dict[str, dict[int, list[int]]]:
    out = {}
    for name, (run, k, *rest) in LB_SETS.items():
        s = read_submission(cfg.paths.runs_dir / run / "submission.json")
        if len(rest) > 3 and rest[3] == "chunks":
            out[name] = {x["id"]: [int(c["doc_id"]) for c in x["relevant_chunks"]][:k] for x in s}
        else:
            out[name] = {x["id"]: [int(d) for d in x["relevant_docs"][:k]] for x in s}
    return out


def _pair_cos(vi: pd.DataFrame, qv: np.ndarray, qpos: dict, dense_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """cosine(query, doc head) and cosine(query, doc mean) per (query, doc) row of vi, from the doc_*_vi.f16.npy
    memmaps of dense_dir (NaN when the doc has no vector); rows gathered in sorted blocks to keep memory low."""
    doc_ids = np.load(dense_dir / "doc_vi_ids.npy")
    H = np.load(dense_dir / "doc_head_vi.f16.npy", mmap_mode="r")
    M = np.load(dense_dir / "doc_mean_vi.f16.npy", mmap_mode="r")
    rows = vi["doc_id"].map(pd.Series(np.arange(len(doc_ids)), index=doc_ids))
    ok = rows.notna().to_numpy()
    r_ok = rows[ok].astype(np.int64).to_numpy()
    q_ok = vi["query_id"].map(qpos).to_numpy()[ok]
    hc, mc = np.full(len(vi), np.nan, np.float32), np.full(len(vi), np.nan, np.float32)
    ok_idx = np.flatnonzero(ok)
    for s in range(0, len(r_ok), 100_000):
        o = np.argsort(r_ok[s: s + 100_000])
        rr, qq = r_ok[s: s + 100_000][o], qv[q_ok[s: s + 100_000][o]].astype(np.float32)
        idx = ok_idx[s: s + 100_000][o]
        hc[idx] = np.einsum("ij,ij->i", H[rr].astype(np.float32), qq)
        mc[idx] = np.einsum("ij,ij->i", M[rr].astype(np.float32), qq)
    return hc, mc


TM_DENSE = "gloryvu-bgem3"  # teammate chunk vectors -> doc vectors (teammate_docvec.py build)


def build_features(title_models: list[str] | None = None, extra_channels: list[str] | None = None,
                   tm: bool = False) -> None:
    """title_models: probe models (probe_eval.py) whose query-vs-title cosine becomes a column title_cos_<tag>."""
    title_models = title_models or []
    cfg = get_config()
    meta = MetaStore(cfg.paths.index_dir / DB_FILE)
    src = cfg.paths.runs_dir / RUN
    # routes: best channel rank per doc
    c = pd.read_parquet(src / "candidates.parquet", columns=["query_id", "chunk_id", "routes"])
    c["doc_id"] = c["chunk_id"] // CHUNK_ID_STRIDE
    for ch in ("dense_vi", "bm25_vi"):
        c[ch] = c["routes"].str.extract(rf"{ch}:(\d+)")[0].astype(float)
    pairs = c.groupby(["query_id", "doc_id"])[["dense_vi", "bm25_vi"]].min().reset_index()
    rr = pd.read_parquet(src / "reranked.parquet", columns=["query_id", "doc_id", "score", "rrf"])
    rr = rr.groupby(["query_id", "doc_id"]).agg(rerank=("score", "max"), rrf=("rrf", "max")).reset_index()
    head = pd.read_parquet(cfg.paths.dev_dir / "head_top300.parquet")[["query_id", "doc_id"]]
    # doc-level channels of the retrieval bench (original query vs head / doc mean, top-300)
    bdir = cfg.paths.dev_dir / "bench"
    # extra_channels: other bench channels whose docs join the universe only (no feature), so that a calibration set
    # ranked by them is fully covered by g (e.g. tm_mean_q, teammate_docvec.py)
    files = [bdir / "head_q.parquet", bdir / "mean_q.parquet", *[bdir / f"title_{t}.parquet" for t in title_models],
             *[bdir / f"{c}.parquet" for c in extra_channels or []]]
    bench = [pd.read_parquet(f)[["query_id", "doc_id"]] for f in files if f.exists()]
    sets = load_sets(cfg)
    extra = pd.DataFrame([(q, d) for s in sets.values() for q, ds in s.items() for d in ds],
                         columns=["query_id", "doc_id"])
    u = pd.concat([pairs[["query_id", "doc_id"]], rr[["query_id", "doc_id"]], head, extra, *bench]).drop_duplicates()
    u = u.merge(pairs, how="left", on=["query_id", "doc_id"]).merge(rr, how="left", on=["query_id", "doc_id"])
    lang = meta.documents(u["doc_id"].unique()).set_index("doc_id")["lang"]
    u["lang"] = u["doc_id"].map(lang)
    # exact query vs head (first chunk) and vs doc-mean cosines, Vietnamese docs
    from medret.retrieve.query_encoder import QueryEncoder

    q = pd.read_parquet(src / "queries.parquet")
    qv = QueryEncoder(cfg.embed.model, cache_path=cfg.paths.data_dir / "cache" / "qvec.sqlite").encode(
        q["query"].tolist())
    qpos = {qid: i for i, qid in enumerate(q["id"])}
    # head / doc-mean matrices written by retrieval_bench.py (float16 memmaps, one row per Vietnamese doc): gathered
    # block by block so memory stays at a few hundred MB (laptop budget)
    dense_dir = cfg.paths.index_dir / "dense" / cfg.index.dense_run
    doc_ids = np.load(dense_dir / "doc_vi_ids.npy")
    H = np.load(dense_dir / "doc_head_vi.f16.npy", mmap_mode="r")
    M = np.load(dense_dir / "doc_mean_vi.f16.npy", mmap_mode="r")
    row_of = pd.Series(np.arange(len(doc_ids)), index=doc_ids)
    vi = u[u["lang"].eq("vi")].copy()
    rows = vi["doc_id"].map(row_of)
    ok = rows.notna().to_numpy()
    r_ok = rows[ok].astype(np.int64).to_numpy()
    q_ok = vi["query_id"].map(qpos).to_numpy()[ok]
    head_cos = np.full(len(vi), np.nan, dtype=np.float32)
    mean_cos = np.full(len(vi), np.nan, dtype=np.float32)
    hc, mc = np.empty(len(r_ok), dtype=np.float32), np.empty(len(r_ok), dtype=np.float32)
    for s in range(0, len(r_ok), 100_000):
        sl = slice(s, s + 100_000)
        o = np.argsort(r_ok[sl])
        rr_sorted = r_ok[sl][o]
        qq = qv[q_ok[sl][o]].astype(np.float32)
        hc_part = np.einsum("ij,ij->i", H[rr_sorted].astype(np.float32), qq)
        mc_part = np.einsum("ij,ij->i", M[rr_sorted].astype(np.float32), qq)
        idx = np.arange(s, min(s + 100_000, len(r_ok)))[o]
        hc[idx], mc[idx] = hc_part, mc_part
    head_cos[ok], mean_cos[ok] = hc, mc
    vi["head_cos"], vi["mean_cos"] = head_cos, mean_cos
    u = u.merge(vi[["query_id", "doc_id", "head_cos", "mean_cos"]], how="left", on=["query_id", "doc_id"])
    if tm:  # same BGE-M3 query vectors, doc vectors from the teammate's chunking (calibrated by tmmean@200)
        vi["tm_head_cos"], vi["tm_mean_cos"] = _pair_cos(vi, qv, qpos, cfg.paths.index_dir / "dense" / TM_DENSE)
        u = u.merge(vi[["query_id", "doc_id", "tm_head_cos", "tm_mean_cos"]], how="left", on=["query_id", "doc_id"])
    for ch in {c for chans in MQ_CHANNELS.values() for c in chans}:  # multi-query channel ranks (top-300)
        if (bdir / f"{ch}.parquet").exists():
            r = pd.read_parquet(bdir / f"{ch}.parquet")[["query_id", "doc_id", "rank"]]
            u = u.merge(r.rename(columns={"rank": f"{ch}_rank"}), how="left", on=["query_id", "doc_id"])
    if title_models:
        spec = importlib.util.spec_from_file_location("pe", Path(__file__).parent / "probe_eval.py")
        pe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pe)
        u = u.reset_index(drop=True)
        for t in title_models:
            u[f"title_cos_{t}"] = pe.title_cosines(cfg, t, u)
            print(f"title_cos_{t}: {u[f'title_cos_{t}'].notna().mean():.0%} of pairs", flush=True)
    out = cfg.paths.dev_dir / "proxy_pairs.parquet"
    u.to_parquet(out, index=False)
    print(f"{len(u):,} pairs ({len(u) / u['query_id'].nunique():.0f}/query), vi {u['lang'].eq('vi').mean():.2f} -> {out}")


def design(u: pd.DataFrame, mq: bool = False, tm: bool = False) -> np.ndarray:
    """Per-query relative features (scales differ between queries); missing -> least favourable value."""
    g = u.groupby("query_id")
    x = pd.DataFrame(index=u.index)
    x["head_rel"] = (u["head_cos"] - g["head_cos"].transform("max")).fillna(-0.3)
    x["mean_rel"] = (u["mean_cos"] - g["mean_cos"].transform("max")).fillna(-0.3)
    x["dense_rr"] = (60 / (60 + u["dense_vi"])).fillna(0)
    x["bm25_rr"] = (60 / (60 + u["bm25_vi"])).fillna(0)
    x["rerank"] = u["rerank"].fillna(0)
    x["in_rerank"] = u["rerank"].notna().astype(float)
    cols = list(FEATURES)
    for c in sorted(c for c in u.columns if c.startswith("title_cos_")):
        x[c.replace("title_cos_", "title_rel_")] = (u[c] - g[c].transform("max")).fillna(-0.3)
        cols.append(c.replace("title_cos_", "title_rel_"))
    if tm:
        for c in ("tm_head", "tm_mean"):
            x[f"{c}_rel"] = (u[f"{c}_cos"] - g[f"{c}_cos"].transform("max")).fillna(-0.3)
            cols.append(f"{c}_rel")
    if mq:
        for name, chans in MQ_CHANNELS.items():
            x[f"{name}_rr"] = np.max([(60 / (60 + u[f"{c}_rank"])).fillna(0) for c in chans], axis=0)
            cols.append(f"{name}_rr")
    return x[cols].to_numpy(dtype=np.float64)


def fit_and_rank(loo: bool = True, drop_title: bool = False, mq: bool = False, v1_sets: bool = False,
                 save: bool = True, tm: bool = False) -> None:
    from scipy.optimize import minimize

    cfg = get_config()
    u = pd.read_parquet(cfg.paths.dev_dir / "proxy_pairs.parquet")
    if drop_title:  # the LB-validated 6-feature proxy (title signals failed on the LB, 2026-10-14)
        u = u.drop(columns=[c for c in u.columns if c.startswith("title_cos_")])
    u = u.reset_index(drop=True)
    vi = u["lang"].eq("vi").to_numpy()
    X = design(u, mq, tm)
    key = pd.MultiIndex.from_frame(u[["query_id", "doc_id"]])
    pos = pd.Series(np.arange(len(u)), index=key)
    qcode, qids = pd.factorize(u["query_id"])
    from scipy.sparse import csr_matrix

    sets = load_sets(cfg)
    mats = {}  # set -> (A: queries x pairs membership, k per query, query codes)
    for name, s in sets.items():
        rows, cols, ks, qc = [], [], [], []
        for i, (q, ds) in enumerate(s.items()):
            ix = pos.reindex(pd.MultiIndex.from_tuples([(q, d) for d in ds])).dropna().astype(int).to_numpy()
            rows += [i] * len(ix)
            cols += ix.tolist()
            ks.append(max(len(ds), 1))
            qc.append(qids.get_loc(q) if q in qids else -1)
        mats[name] = (csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(s), len(u))), np.array(ks),
                      np.array(qc))

    def predict(theta: np.ndarray) -> np.ndarray:
        g = 1 / (1 + np.exp(-(X @ theta[:-2] + theta[-2])))
        return np.where(vi, g, 0.0)  # non-Vietnamese docs: never gold in this phase

    def set_pr(g: np.ndarray, c: float, name: str) -> tuple[float, float]:
        tot = np.bincount(qcode, weights=g, minlength=len(qids)) * c
        a, ks, qc = mats[name]
        hit = a @ g
        denom = np.where(qc >= 0, tot[qc.clip(min=0)], np.inf)
        return float(np.mean(hit / ks)), float(np.mean(np.where(denom > 0, hit / denom, 0.0)))

    def loss(theta: np.ndarray, use: list[str]) -> float:
        g, c = predict(theta), 1 + np.exp(theta[-1])
        err = 0.0
        for name in use:
            p, r = set_pr(g, c, name)
            lp, lr = LB_SETS[name][2], LB_SETS[name][3]
            err += (p - lp) ** 2 + ((r - lr) ** 2 if lr is not None else 0.0)
        return err

    theta0 = np.zeros(X.shape[1] + 2)  # +2: bias, log(c - 1)
    theta0[-2] = -2.0
    names = list(LB_SETS)[:N_SETS_V1] if v1_sets else list(LB_SETS)
    best = minimize(loss, theta0, args=(names,), method="Powell", options={"maxiter": 4000, "xtol": 1e-3})
    th = best.x
    g, c = predict(th), 1 + np.exp(th[-1])
    feat_names = FEATURES + [c.replace("title_cos_", "title_rel_") for c in sorted(u.columns)
                             if c.startswith("title_cos_")] + (["tm_head_rel", "tm_mean_rel"] if tm else []) \
        + ([f"{n}_rr" for n in MQ_CHANNELS] if mq else [])
    print("weights:", dict(zip(feat_names + ["bias"], np.round(th[:-1], 3), strict=True)), f"c={c:.2f}")
    print(f"{'set':12s} {'LB P':>6s} {'fit P':>6s} {'LOO P':>6s} | {'LB R':>6s} {'fit R':>6s} {'LOO R':>6s}")
    loo_err, out_err = [], []
    for name in LB_SETS:  # sets outside the fit (marked *) are a true out-of-sample check
        held_out = name not in names
        th_loo = th if (not loo or held_out) else minimize(loss, th, args=([n for n in names if n != name],),
                                                           method="Powell", options={"maxiter": 2000, "xtol": 1e-3}).x
        gl, cl = predict(th_loo), 1 + np.exp(th_loo[-1])
        p, r = set_pr(g, c, name)
        pl, rl = set_pr(gl, cl, name)
        lp, lr = LB_SETS[name][2], LB_SETS[name][3]
        (out_err if held_out else loo_err).append(abs(pl - lp))
        print(f"{name + ('*' if held_out else ''):12s} {lp:6.3f} {p:6.3f} {pl:6.3f} | {lr or 0:6.3f} {r:6.3f} {rl:6.3f}")
    print(f"{'leave-one-set-out' if loo else 'in-sample (no LOO)'} mean |P error| {np.mean(loo_err):.4f}")
    if out_err:
        print(f"held-out sets (*) mean |P error| {np.mean(out_err):.4f}")
    if not save:
        return
    u["g"] = g
    u.to_parquet(cfg.paths.dev_dir / "proxy_pairs_scored.parquet", index=False)
    rank_strategies(u, c)


def _proxy_scores(v: pd.DataFrame, score: np.ndarray, tot: pd.Series, k_doc: int, k_full: int) -> tuple:
    """Proxy doc F2@k_doc, chunk F2@k_full (whole-doc chunks = doc P/R of the top k_full), P@k_full, R@k_doc."""
    w = v[["query_id", "g"]].assign(s=score).sort_values(["query_id", "s"], ascending=[True, False])
    w["r"] = w.groupby("query_id").cumcount()
    res = []
    for k in (k_doc, k_full):
        hit = w[w["r"] < k].groupby("query_id")["g"].sum()
        p, r = hit / k, hit / tot.reindex(hit.index)
        res.append(((5 * p * r / (4 * p + r)).fillna(0).mean(), p.mean(), r.mean()))
    return res[0][0], res[1][0], res[1][1], res[0][2]


def rank_strategies(u: pd.DataFrame, c: float, k_doc: int = 200, k_full: int = 65, n_random: int = 300) -> None:
    """Fusion weights searched on the proxy; the submitted RRF fusion (LB 0.3076) calibrates the scale."""
    cfg = get_config()
    v = u[u["lang"].eq("vi")].copy().reset_index(drop=True)
    tot = u.groupby("query_id")["g"].sum() * c
    gq = v.groupby("query_id")

    def z(col: str, fill: float) -> np.ndarray:
        x = v[col].fillna(fill)
        return ((x - gq[col].transform("mean")) / (gq[col].transform("std") + 1e-9)).fillna(-3).to_numpy()

    F = {"head": z("head_cos", -1.0), "mean": z("mean_cos", -1.0),
         "dense": (60 / (60 + v["dense_vi"])).fillna(0).to_numpy() * 5,
         "bm25": (60 / (60 + v["bm25_vi"])).fillna(0).to_numpy() * 5,
         "rerank": z("rerank", 0.0)}
    for c in sorted(c for c in v.columns if c.startswith("title_cos_")):
        F[c.replace("title_cos_", "title:")] = z(c, -1.0)
    sub = read_submission(cfg.paths.runs_dir / "sub-vi200-rq-chunkheadmean-v1" / "submission.json")
    rank_sub = {(x["id"], int(d)): r for x in sub for r, d in enumerate(x["relevant_docs"])}
    named = {
        "rerank (best before)": v["rerank"].fillna(-1).to_numpy(),
        "RRF chunk+head+mean (LB 0.3076)": -np.array([rank_sub.get((q, d), 10_000) for q, d in
                                                     zip(v["query_id"], v["doc_id"], strict=True)], dtype=float),
        "proxy g (model itself)": v["g"].to_numpy(),
    }
    rows = []
    for name, sc in named.items():
        rows.append((name, None, *_proxy_scores(v, sc, tot, k_doc, k_full)))
    rng = np.random.default_rng(0)
    for i in range(n_random):
        w = {"head": rng.uniform(0, 3), "mean": rng.uniform(0, 3), "dense": rng.uniform(0, 2),
             "bm25": rng.uniform(0, 1), "rerank": rng.uniform(0, 1.5)}
        w |= {k: rng.uniform(0, 4) for k in F if k.startswith("title:")}
        sc = sum(w[k] * F[k] for k in w)
        rows.append((f"search #{i}", {k: round(x, 2) for k, x in w.items()}, *_proxy_scores(v, sc, tot, k_doc, k_full)))
    df = pd.DataFrame(rows, columns=["name", "weights", "docF2", "chunkF2", "P65", "R200"])
    df["final"] = (df["docF2"] + df["chunkF2"]) / 2
    ref = df.loc[df["name"].str.startswith("RRF"), "final"].iloc[0]
    df["final_LB_est"] = df["final"] - ref + 0.3076
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(df[df["weights"].isna()].round(3).to_string(index=False))
    print("\ntop weight sets:")
    print(df[df["weights"].notna()].sort_values("final", ascending=False).head(12).round(3).to_string(index=False))
    df.to_parquet(cfg.paths.dev_dir / "proxy_fusion_search.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "fit"])
    ap.add_argument("--title-models", nargs="*", default=[])
    ap.add_argument("--extra-channels", nargs="*", default=[], help="bench channels added to the universe only")
    ap.add_argument("--tm-features", action="store_true", help="teammate doc head / mean cosines (compute and fit)")
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-set-out refits (slow with many features)")
    ap.add_argument("--drop-title", action="store_true", help="fit without the title_cos_* columns")
    ap.add_argument("--mq-features", action="store_true", help="add multi-query channel ranks (sub-questions, titles)")
    ap.add_argument("--v1-sets", action="store_true", help="fit on the first 13 LB sets only (the 0.3241 proxy)")
    ap.add_argument("--no-save", action="store_true", help="diagnostic fit: do not overwrite proxy_pairs_scored")
    a = ap.parse_args()
    from medret.system import lower_priority

    lower_priority()
    if a.cmd == "features":
        build_features(a.title_models, a.extra_channels, a.tm_features)
    else:
        fit_and_rank(loo=not a.no_loo, drop_title=a.drop_title, mq=a.mq_features, v1_sets=a.v1_sets,
                     save=not a.no_save, tm=a.tm_features)


if __name__ == "__main__":
    main()
