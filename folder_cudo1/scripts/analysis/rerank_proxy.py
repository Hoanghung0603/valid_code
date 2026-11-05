"""P(doc is gold | cross-encoder scores) fitted to the LB doc P/R of submitted doc sets, then k per query by expected F2.

  python scripts/analysis/rerank_proxy.py fit
  python scripts/analysis/rerank_proxy.py k --weights 1 0.5 1.5    # -> data/dev/rerank_proxy_k.parquet (fused_sub --k-file)
Learning from label proportions, as gold_proxy.py, but on the Kaggle rerank candidates (vi top-200 of the 0.3345 order,
zh top-100 of the teammate ranking): p(d) = sigmoid(b0 + b1 logit(s4B) + b2 logit(s0.6B) + b3 [zh] + b4 [no text]
+ b5 log(1 + source rank) for vi + b6 for zh) (the source rank carries the proxy g order of the vi candidates).
Every doc of the fitted sets is a candidate, so the sets differ in exactly the scores the model uses (S4 = 0.6B order,
S6 / S7 = fusions with 4B, the 0.3345 and rrfwp lists = proxy-g orders): the fit interpolates. Gold outside the
candidates: G_q = sum_cand p / (1 - f), f fitted. LB values are macro over queries, so are the predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from medret.config import get_config
from medret.submit import iter_submission

SETS = {  # run: (doc P, doc R) on the LB
    "sub-vi-rrfwp-zh30docs-v1": (0.2006, 0.5922),
    "sub-vi-rrfwp-win2x10-v1": (0.2169, 0.5262),
    "sub-rr06-200-chunksbest-v1": (0.1942, 0.6122),
    "sub-fuse3-chunksfused-v1": (0.2058, 0.6097),
    "sub-fuse3w-4bfull-v1": (0.2097, 0.6226),
    "sub-fuse3w-4bfull-kproxy-v1": (0.2434, 0.5672),  # S8: k per query from this proxy (5-set fit), 2026-10-27
    # S9: deeper pool, k from the deep fit (k mean 178): the first LB set with docs ranked 200-399 (fit with --deep)
    "sub-deep-kproxy-nproxy-v1": (0.2191, 0.6005),
    "sub-twins-kproxy-nproxy-v1": (0.2326, 0.5822),  # S10: twin pages scored by 4B (fit with --deep --twins)
    # S11 itself (sub-long-kproxy-nproxy-v1, LB 0.2374 / 0.5768) joined this list after it was built: its k and N
    # came from the fit on the 8 sets above, so it stays out here to reproduce the submission exactly
}
SCORES = {"s4": "data/kaggle/rerank/qwen3-4b-full/scores.parquet", "s06": "data/kaggle/rerank/qwen3-0.6b/scores.parquet"}
ALL = {"s06": "data/kaggle/rerank/qwen3-0.6b-all/scores.parquet", "s4": "data/kaggle/rerank/qwen3-4b-all/scores.parquet"}
ALL_TW = {**ALL, "s4": "data/kaggle/rerank/qwen3-4b-all-tw/scores.parquet"}  # + 81k twin pairs (2026-10-28)
# + the top-150 per query rescored with 4,000 chars / 1,024 tokens (merge_long_scores.py, 2026-10-28)
ALL_LONG = {**ALL, "s4": "data/kaggle/rerank/qwen3-4b-all-tw-long/scores.parquet"}
# deeper pool (vi 200-399, zh 100-199; 4B only on the 0.6B top-75 per query), used when present
DEEP = {"s4": "data/kaggle/rerank/qwen3-4b-deep/scores.parquet", "s06": "data/kaggle/rerank/qwen3-0.6b-deep/scores.parquet"}


def _scores(key: str, deep: bool, cols: list[str] | None = None) -> pd.DataFrame:
    files = [SCORES[key]] + ([DEEP[key]] if deep and Path(DEEP[key]).exists() else [])
    return pd.concat(pd.read_parquet(f, columns=cols) for f in files)


def candidates(deep: bool = False, twins: bool = False, long: bool = False) -> pd.DataFrame:
    """deep: also the deeper pool's pairs that have a 4B score (the others were cut by 0.6B). twins: 4B scores of
    the text-less zh pairs read from their http/https twin's page (qwen3-4b-all-tw, implies deep). long: 4B scores
    of the top-150 read with 4,000 chars (qwen3-4b-all-tw-long, implies twins)."""
    twins = twins or long
    a = _scores("s06", deep or twins).rename(columns={"score": "s06"})
    if twins:
        b = pd.read_parquet((ALL_LONG if long else ALL_TW)["s4"], columns=["query_id", "doc_id", "score", "has_text"])
    else:
        b = _scores("s4", deep, ["query_id", "doc_id", "score", "has_text"])
    b = b.rename(columns={"score": "s4", "has_text": "t4"})
    c = a.merge(b, on=["query_id", "doc_id"])
    eps = 1e-4
    for k, t in (("s4", c["t4"]), ("s06", c["has_text"])):  # per model: the 0.6B never read the twins' text
        x = c[k].clip(eps, 1 - eps)
        c[f"l{k}"] = np.log(x / (1 - x)) * t
    c["zh"] = (c["lang"] == "zh").astype(float)
    c["notext"] = (~c["t4"]).astype(float)
    return c.reset_index(drop=True)


NF = 8  # b0, 4B, 0.6B, zh, no text, log source rank (vi), log source rank (zh), deeper pool


def features(c: pd.DataFrame) -> np.ndarray:
    lr = np.log1p(c["src_rank"].to_numpy(float))
    # deeper pool (vi rank >= 200, zh >= 100): S9 showed the log-rank term alone over-rates these docs (2026-10-27)
    deep = ((c["src_rank"] >= 200) & (c["zh"] == 0)) | ((c["src_rank"] >= 100) & (c["zh"] == 1))
    return np.stack([np.ones(len(c)), c["ls4"], c["ls06"], c["zh"], c["notext"], lr * (1 - c["zh"]), lr * c["zh"],
                     deep.astype(float)], 1)


def load_sets(c: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = pd.Series(np.arange(len(c)), index=pd.MultiIndex.from_arrays([c["query_id"], c["doc_id"]]))
    out = {}
    cfg = get_config()
    for run in SETS:
        rows, miss = [], 0
        for x in iter_submission(cfg.paths.runs_dir / run / "submission.json"):
            for d in x["relevant_docs"]:
                k = (int(x["id"]), int(d))
                if k in idx.index:
                    rows.append(idx[k])
                else:
                    miss += 1
        out[run] = np.array(rows)
        print(f"  {run}: {len(rows):,} docs, {miss} outside the candidates")
    return out


def predict(theta: np.ndarray, X: np.ndarray, qcode: np.ndarray, nq: int, sets: dict) -> dict:
    p = 1 / (1 + np.exp(-(X @ theta[:NF])))
    f = 1 / (1 + np.exp(-theta[NF]))
    g = np.bincount(qcode, weights=p, minlength=nq) / (1 - f)
    res = {}
    for run, rows in sets.items():
        hits = np.bincount(qcode[rows], weights=p[rows], minlength=nq)
        n = np.bincount(qcode[rows], minlength=nq)
        ok = n > 0
        res[run] = (float(np.mean(hits[ok] / n[ok])), float(np.mean(hits[ok] / g[ok])))
    return res


def _out(deep: bool, kind: str, twins: bool = False, long: bool = False) -> Path:
    tag = f"{'_deep' if deep else ''}{'_tw' if twins and not long else ''}{'_long' if long else ''}"
    return get_config().paths.dev_dir / f"rerank_proxy{tag}{kind}"


def fit(deep: bool = False, loo: bool = True, twins: bool = False, long: bool = False) -> None:
    """deep: the deeper pool's 4B-scored pairs join the candidates (only G_q changes: every fitted set lies in the
    first pool), written to rerank_proxy_deep.json."""
    c = candidates(deep, twins, long)
    X = features(c)
    qcode, quniq = pd.factorize(c["query_id"])
    sets = load_sets(c)

    def loss(t, sub=sets):
        pr = predict(t, X, qcode, len(quniq), sub)
        err = sum((pr[r][0] - SETS[r][0]) ** 2 + (pr[r][1] - SETS[r][1]) ** 2 for r in sub)
        return err + 1e-4 * np.sum(t[1:NF] ** 2)

    best = None
    opts = {"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-9}
    starts = ([-0.9, 0.34, 0.23, -0.1, 0.18, -0.17, -0.5, 0, -1.7], [-0.5, 0.3, 0, 0, 0, -0.3, -0.3, -0.5, -1.6],
              [-1, 0.2, 0.1, 0, 0, -0.2, -0.1, -1, -1])
    for x0 in starts:
        r = minimize(loss, np.array(x0, float), method="Nelder-Mead", options=opts)
        if best is None or r.fun < best.fun:
            best = r
    pr = predict(best.x, X, qcode, len(quniq), sets)
    print("theta (b0, 4B, 0.6B, zh, notext, log rank vi, log rank zh, deep, logit f):", np.round(best.x, 3).tolist())
    for r in SETS:
        print(f"  {r:32s} LB P/R {SETS[r][0]:.4f}/{SETS[r][1]:.4f}  fit {pr[r][0]:.4f}/{pr[r][1]:.4f}")
    # leave-one-out: does it predict a set it has not seen? (6 refits: most of the run time)
    for held in SETS if loo else []:
        sub = {k: v for k, v in sets.items() if k != held}
        r2 = minimize(loss, best.x, args=(sub,), method="Nelder-Mead", options={"maxiter": 3000})
        ph = predict(r2.x, X, qcode, len(quniq), {held: sets[held]})[held]
        print(f"  held out {held:32s} LB {SETS[held][0]:.4f}/{SETS[held][1]:.4f}  predicted {ph[0]:.4f}/{ph[1]:.4f}")
    out = _out(deep, ".json", twins, long)
    out.write_text(json.dumps({"theta": best.x.tolist(), "sets": SETS, "fit": pr, "deep": deep}, indent=1))
    print("->", out)


def doc_probs(deep: bool = False, twins: bool = False, long: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """p per candidate (query_id, doc_id, lang, p) and the expected gold count per query under the fitted proxy."""
    th = np.array(json.loads(_out(deep, ".json", twins, long).read_text())["theta"])
    c = candidates(deep, twins, long)
    c["p"] = 1 / (1 + np.exp(-(features(c) @ th[:NF])))
    gq = c.groupby("query_id")["p"].sum() / (1 - 1 / (1 + np.exp(-th[NF])))
    return c[["query_id", "doc_id", "lang", "p"]], gq


def f2(h: float, k: int, g: float) -> float:
    p, r = h / k, h / g
    return 5 * p * r / (4 * p + r) if h > 0 else 0.0


def choose_k(base_run: str, weights: list[float], deep: bool = False, k_min: int = 10, step: int = 5,
             twins: bool = False, long: bool = False) -> None:
    """k per query maximising the expected doc F2 along the fused order (fused_sub.py) under the fitted p; with deep,
    the order runs over both pools (data/kaggle/rerank/qwen3-*-all: first pool + deeper pool scores)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fused_sub", Path(__file__).parent / "fused_sub.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    cfg = get_config()
    probs, gq = doc_probs(deep, twins, long)
    pmap = probs.set_index(["query_id", "doc_id"])["p"]
    base = {x["id"]: [int(d) for d in x["relevant_docs"]]
            for x in iter_submission(cfg.paths.runs_dir / base_run / "submission.json")}
    src = ALL_LONG if long else ALL_TW if twins else ALL if deep else SCORES
    files = [src["s06"], src["s4"]]
    order = fs.fused_order(base, files, weights).groupby("query_id")["doc_id"].agg(list)
    rows, cur, best = [], [], []
    for q, ds in order.items():
        cum = np.cumsum([pmap.get((q, d), 0.0) for d in ds])
        k0 = len(base[q])
        kb = max(range(k_min, len(ds) + 1, step), key=lambda k: f2(cum[k - 1], k, gq[q]))
        rows.append((int(q), kb))
        cur.append(f2(cum[k0 - 1], k0, gq[q]))
        best.append(f2(cum[kb - 1], kb, gq[q]))
    ks = pd.DataFrame(rows, columns=["query_id", "k"])
    out = _out(deep, "_k.parquet", twins, long)
    ks.to_parquet(out, index=False)
    print(f"expected doc F2: base list sizes {np.mean(cur):.4f} -> per-query k {np.mean(best):.4f}; "
          f"k mean {ks['k'].mean():.1f}, p10/50/90 {np.percentile(ks['k'], [10, 50, 90])} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "k"])
    ap.add_argument("--base", default="sub-vi-rrfwp-zh30docs-v1")
    ap.add_argument("--weights", nargs="+", type=float, default=[1, 0.5, 1.5])
    ap.add_argument("--deep", action="store_true", help="include the deeper candidate pool")
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-out check")
    ap.add_argument("--twins", action="store_true", help="4B scores of the text-less zh pairs from their twin page")
    ap.add_argument("--long", action="store_true", help="4B scores of the top-150 read with 4,000 chars (implies twins)")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.deep, not a.no_loo, a.twins, a.long)
    else:
        choose_k(a.base, a.weights, a.deep, twins=a.twins, long=a.long)


if __name__ == "__main__":
    main()
