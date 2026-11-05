"""Embedding-model alignment probe, evaluation: which model's query-vs-title similarity explains the LB gold?

  python scripts/analysis/probe_eval.py [--models bgem3-title e5-large ...]
For every probe model (data/embeddings/probe-<tag>/, from configs/kaggle/probe_<tag>.json) the query-vs-title cosine
of every proxy pair (data/dev/proxy_pairs.parquet) becomes a per-query relative feature. The gold proxy
(gold_proxy.py: logistic P(gold | features), fitted to the LB P / R of 13 submitted doc sets) is refitted
  (a) on that single feature  -> fit loss and leave-one-set-out P error: how much of the gold one model explains;
  (b) on the current features + that feature -> fit loss only (no LOO: 8-parameter Powell fits are slow):
      does the model add information to BGE-M3 head / mean?
Lower is better. Memory stays low: title vectors are read block by block from the float16 shards.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from medret.config import get_config
from medret.system import lower_priority

QUERY_FMT = {"bgem3-title": 0, "e5-large": 1, "e5-large-instruct": 2, "qwen3-emb-0.6b": 3, "qwen3-emb-4b": 3,
             "e5i-head": 2,  # e5i-head: e5-large-instruct on doc heads (make_embed_probe.py --heads)
             "aitvn-head": 0}  # AITeamVN/Vietnamese_Embedding (BGE-M3 fine-tune) on doc heads, raw query


def _proxy_module():
    spec = importlib.util.spec_from_file_location("gp", Path(__file__).parent / "gold_proxy.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    return gp


def title_cosines(cfg, tag: str, u: pd.DataFrame) -> np.ndarray:
    """cosine(query, title) per pair under one probe model; NaN where the doc has no vector."""
    d = cfg.paths.data_dir / "embeddings" / f"probe-{tag}"
    ids = np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "shard-*.ids.npy")))])
    files = sorted(glob.glob(str(d / "shard-*.vec.npy")))
    sizes = [np.load(f, mmap_mode="r").shape[0] for f in files]
    dim = np.load(files[0], mmap_mode="r").shape[1]
    vec = np.lib.format.open_memmap(d / "all.f16.npy", mode="w+", dtype=np.float16, shape=(sum(sizes), dim)) \
        if not (d / "all.f16.npy").exists() else None
    if vec is not None:
        pos = 0
        for f, n in zip(files, sizes, strict=True):
            vec[pos: pos + n] = np.load(f, mmap_mode="r")
            pos += n
        vec.flush()
    vec = np.load(d / "all.f16.npy", mmap_mode="r")
    row = pd.Series(np.arange(len(ids)), index=ids)
    fmt = QUERY_FMT[tag]
    # one row per query (not per pair: 1.1M x 2560 floats would be ~11 GB for Qwen3-Embedding-4B)
    qids = [q for q in u["query_id"].unique() if -(fmt * 10_000 + q) in row.index]
    QM = np.stack([vec[row[-(fmt * 10_000 + q)]].astype(np.float32) for q in qids])
    qrow = pd.Series(np.arange(len(qids)), index=qids)
    out = np.full(len(u), np.nan, dtype=np.float32)
    r = u["doc_id"].map(row)
    ok = np.flatnonzero(r.notna().to_numpy() & u["query_id"].isin(qrow.index).to_numpy())
    rr = r.to_numpy()[ok].astype(np.int64)
    qq = u["query_id"].map(qrow).to_numpy()[ok].astype(np.int64)
    for s in range(0, len(ok), 50_000):
        o = np.argsort(rr[s: s + 50_000])
        part = np.einsum("ij,ij->i", vec[rr[s: s + 50_000][o]].astype(np.float32), QM[qq[s: s + 50_000][o]])
        out[ok[s: s + 50_000][o]] = part
    return out


def fit(gp, u: pd.DataFrame, X: np.ndarray, sets, names: list[str], loo: bool = True) -> tuple[float, float]:
    vi = u["lang"].eq("vi").to_numpy()
    qcode, qids = pd.factorize(u["query_id"])
    pos = pd.Series(np.arange(len(u)), index=pd.MultiIndex.from_frame(u[["query_id", "doc_id"]]))
    mats = {}
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

    def pr(theta, name):
        g = np.where(vi, 1 / (1 + np.exp(-(X @ theta[:-2] + theta[-2]))), 0.0)
        tot = np.bincount(qcode, weights=g, minlength=len(qids)) * (1 + np.exp(theta[-1]))
        a, ks, qc = mats[name]
        hit = a @ g
        den = np.where(qc >= 0, tot[qc.clip(min=0)], np.inf)
        return float(np.mean(hit / ks)), float(np.mean(np.where(den > 0, hit / den, 0.0)))

    def loss(theta, use):
        err = 0.0
        for n in use:
            p, r = pr(theta, n)
            lp, lr = gp.LB_SETS[n][2], gp.LB_SETS[n][3]
            err += (p - lp) ** 2 + ((r - lr) ** 2 if lr is not None else 0.0)
        return err

    th0 = np.zeros(X.shape[1] + 2)
    th0[-2] = -2.0
    best = minimize(loss, th0, args=(names,), method="Powell", options={"maxiter": 3000, "xtol": 1e-3})
    if not loo:
        return float(best.fun), float("nan")
    errs = []
    for n in names:
        th = minimize(loss, best.x, args=([m for m in names if m != n],), method="Powell",
                      options={"maxiter": 1500, "xtol": 1e-3}).x
        errs.append(abs(pr(th, n)[0] - gp.LB_SETS[n][2]))
    return float(best.fun), float(np.mean(errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-reference", action="store_true", help="skip the BGE-M3 head / current-proxy rows")
    ap.add_argument("--no-combined", action="store_true", help="only the single-feature fits (8-param fits ~25 min)")
    a = ap.parse_args()
    lower_priority()
    cfg = get_config()
    gp = _proxy_module()
    u = pd.read_parquet(cfg.paths.dev_dir / "proxy_pairs.parquet").reset_index(drop=True)
    sets = gp.load_sets(cfg)
    names = list(gp.LB_SETS)
    base = gp.design(u)
    tags = a.models or [t for t in QUERY_FMT if (cfg.paths.data_dir / "embeddings" / f"probe-{t}").exists()]
    g = u.groupby("query_id")
    rows = []
    # reference: BGE-M3 head (first chunk) alone, and the current 6-feature proxy
    head_rel = (u["head_cos"] - g["head_cos"].transform("max")).fillna(-0.3).to_numpy()[:, None]
    import time

    t0 = time.time()
    if not a.skip_reference:
        rows.append(("BGE-M3 head (first chunk), alone", *fit(gp, u, head_rel, sets, names)))
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
        # 6-feature proxy: fit loss only (its leave-one-set-out error, 0.007, is known from gold_proxy.py fit)
        rows.append(("current proxy (6 features)", *fit(gp, u, base, sets, names, loo=False)))
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    for t in tags:
        cos = pd.Series(title_cosines(cfg, t, u))
        rel = (cos - cos.groupby(u["query_id"]).transform("max")).fillna(-0.3).to_numpy()[:, None]
        cover = float(np.isfinite(cos).mean())
        rows.append((f"{t} title, alone (cover {cover:.0%})", *fit(gp, u, rel, sets, names)))
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
        if not a.no_combined:
            rows.append((f"current proxy + {t} title", *fit(gp, u, np.hstack([base, rel]), sets, names, loo=False)))
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["features", "fit_loss", "LOO_mean_abs_P_err"])
    print(df.round(5).to_string(index=False))
    out = cfg.paths.dev_dir / "probe_eval.csv"
    df.to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
