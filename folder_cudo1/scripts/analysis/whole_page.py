"""Whole-page BGE-M3 channels (dense, sparse, hybrid) from the teammate's page embeddings, compared with ours.

  python scripts/analysis/whole_page.py channels   # -> data/dev/bench/wp_{dense,sparse,hybrid}_q.parquet (top-300)
  python scripts/analysis/whole_page.py compare    # overlap with our channels, proxy-expected P / R of each
Source: Kaggle private datasets gloryvu/vibiomir-bgem3-pages (part A, 307,962 docs) and -pages-b (part B, the other
427,948 Vietnamese docs + 44 re-fetched ones that replace their part-A rows), downloaded to data/external/gloryvu_pages/
{a,b}: dense-NNNN.npy (fp16, L2-normalised CLS of BAAI/bge-m3 over the whole page main content, max 8192 tokens),
ids-NNNN.parquet (row order), sparse-NNNN.parquet (doc_id, tok, w: bge-m3 lexical weights), queries_dense.npy and
queries_sparse.parquet (the 1,200 queries with the same text normalisation as the pages).
Scores: dense = q.d; sparse = sum over shared tokens of q_w * d_w; hybrid = dense + 0.3 sparse (bge-m3 paper).
Computed block by block over the page files (one file's sparse matrix in RAM at a time), keeping the top-300 per query.
"""

from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from medret.config import get_config
from medret.system import lower_priority

SRC = "data/external/gloryvu_pages"
VOCAB = 250_002
TOP = 300
CHANNELS = ("wp_dense_q", "wp_sparse_q", "wp_hybrid_q")


def _csr(ids_order: np.ndarray, sp_df: pd.DataFrame) -> sp.csr_matrix:
    """Sparse rows in the order of ids_order (the dense file's row order)."""
    sp_df = sp_df.set_index("doc_id").reindex(ids_order)
    toks = sp_df["tok"].tolist()
    ws = sp_df["w"].tolist()
    lens = np.array([0 if t is None or (isinstance(t, float)) else len(t) for t in toks])
    indptr = np.r_[0, np.cumsum(lens)]
    ind = np.concatenate([np.asarray(t, np.int32) for t, n in zip(toks, lens, strict=True) if n])
    val = np.concatenate([np.asarray(w, np.float32) for w, n in zip(ws, lens, strict=True) if n])
    return sp.csr_matrix((val, ind, indptr), shape=(len(ids_order), VOCAB))


BLOCK = 25_000


def _merge(best: dict, c: str, sc: np.ndarray, ids: np.ndarray) -> None:
    """Fold one (queries x docs) score block into the running top-TOP of channel c."""
    bs, bi = best[c]
    k = min(TOP, sc.shape[1])
    top = np.argpartition(-sc, k - 1, axis=1)[:, :k] if sc.shape[1] > k else np.tile(np.arange(k), (len(sc), 1))
    bs = np.concatenate([bs, np.take_along_axis(sc, top, 1)], 1)
    bi = np.concatenate([bi, ids[top]], 1)
    if bs.shape[1] > TOP:
        sel = np.argpartition(-bs, TOP - 1, axis=1)[:, :TOP]
        bs, bi = np.take_along_axis(bs, sel, 1), np.take_along_axis(bi, sel, 1)
    best[c] = (bs, bi)


def _queries():
    q = pd.read_parquet(f"{SRC}/a/queries.parquet")
    Qd = np.load(f"{SRC}/a/queries_dense.npy").astype(np.float32)
    qs = pd.read_parquet(f"{SRC}/a/queries_sparse.parquet").set_index("id").reindex(q["id"])
    lens = np.array([len(t) for t in qs["tok"]])
    Qs = sp.csr_matrix((np.concatenate([np.asarray(w, np.float32) for w in qs["w"]]),
                        np.concatenate([np.asarray(t, np.int32) for t in qs["tok"]]), np.r_[0, np.cumsum(lens)]),
                       shape=(len(q), VOCAB))
    return q["id"].to_numpy(), Qd, Qs


def channels() -> None:
    cfg = get_config()
    qids, Qd, Qs = _queries()
    QsT = Qs.T.tocsc()
    b_ids = set()
    for f in glob.glob(f"{SRC}/b/ids-*.parquet"):
        b_ids.update(pd.read_parquet(f)["doc_id"].tolist())
    best = {c: (np.full((len(qids), 0), -np.inf, np.float32), np.zeros((len(qids), 0), np.int64)) for c in CHANNELS}
    t0 = time.time()
    for part in ("a", "b"):
        for df in sorted(glob.glob(f"{SRC}/{part}/dense-*.npy")):
            n = df.split("-")[-1].split(".")[0]
            ids = pd.read_parquet(f"{SRC}/{part}/ids-{n}.parquet")["doc_id"].to_numpy(np.int64)
            keep = np.ones(len(ids), bool) if part == "b" else ~np.isin(ids, list(b_ids))
            D = np.load(df, mmap_mode="r")
            S_all = _csr(ids, pd.read_parquet(f"{SRC}/{part}/sparse-{n}.parquet"))
            for s in range(0, len(ids), BLOCK):  # (queries x 25k docs) score blocks: ~120 MB each
                sel_rows = np.flatnonzero(keep[s: s + BLOCK]) + s
                if not len(sel_rows):
                    continue
                bid = ids[sel_rows]
                dense = Qd @ np.asarray(D[sel_rows], dtype=np.float32).T
                sparse = np.asarray((S_all[sel_rows] @ QsT).T.todense(), dtype=np.float32)
                for c, sc in zip(CHANNELS, (dense, sparse, dense + 0.3 * sparse), strict=True):
                    _merge(best, c, sc, bid)
            print(f"{part}/{n}: {keep.sum():,} docs, {time.time() - t0:.0f}s", flush=True)
    for c, (bs, bi) in best.items():
        o = np.argsort(-bs, axis=1)
        docs = np.take_along_axis(bi, o, 1)
        res = pd.DataFrame({"query_id": np.repeat(qids, docs.shape[1]), "doc_id": docs.ravel(),
                            "rank": np.tile(np.arange(docs.shape[1]), len(qids))})
        res.to_parquet(cfg.paths.dev_dir / "bench" / f"{c}.parquet", index=False)
        print(f"channel {c}: {len(res):,} rows", flush=True)


def compare(k: int = 200, scored: str = "data/dev/proxy_v6_6f/proxy_pairs_scored.parquet") -> None:
    cfg = get_config()
    b = cfg.paths.dev_dir / "bench"
    u = pd.read_parquet(scored, columns=["query_id", "doc_id", "g"])
    g, gq = u.set_index(["query_id", "doc_id"])["g"], u.groupby("query_id")["g"].sum()

    def top(name):
        f = pd.read_parquet(b / f"{name}.parquet")
        return f[f["rank"] < k].sort_values(["query_id", "rank"]).groupby("query_id")["doc_id"].agg(list)

    names = ("head_q", "mean_q", "tm_mean_q", *CHANNELS)
    tops = {n: top(n) for n in names}
    def overlap(x: str, y: str) -> float:
        return float(np.mean([len(set(tops[x][q]) & set(tops[y][q])) / k for q in tops[x].index]))

    for a_ in CHANNELS:
        print(a_, " ".join(f"| overlap {b_} {overlap(a_, b_):.3f}" for b_ in ("head_q", "mean_q", "tm_mean_q")))
    for n, t in tops.items():
        P, R, C = [], [], []
        for q, ds in t.items():
            v = np.array([g.get((q, d), np.nan) for d in ds])
            C.append(np.isfinite(v).mean())
            P.append(np.nansum(v) / len(ds))
            R.append(np.nansum(v) / gq.get(q, np.inf))
        print(f"{n:12s} proxy-expected P@{k} {np.mean(P):.3f} R@{k} {np.mean(R):.3f} (pairs with g: {np.mean(C):.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["channels", "compare"])
    ap.add_argument("--scored", default="data/dev/proxy_v6_6f/proxy_pairs_scored.parquet")
    a = ap.parse_args()
    lower_priority()
    channels() if a.cmd == "channels" else compare(scored=a.scored)


if __name__ == "__main__":
    main()
