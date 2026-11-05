"""Per-query doc count k and whole-document chunk count N under one global size budget (expected-F2, proxy g).

  python scripts/analysis/allocate_sub.py simulate                 # proxy gains of uniform vs adaptive k / N
  python scripts/analysis/allocate_sub.py build --name X --mult 1.0 [--adaptive-k] [--adaptive-n] [--order g]
Order = the best LB run (sub-vi200-rq-chunkheadmean-v1, 0.3076; the 6-feature proxy was fitted on it, so its g
values on this order are interpolation, not extrapolation). Expected gold per doc = proxy g; gold per query
G_q = sum of g over the query's candidate universe. Doc F2(k) and chunk F2(N) use the expected hits (whole-doc
chunks: chunk P / R ~ doc P / R of the top N). Budget = uncompressed JSON bytes of the whole-doc chunks,
estimated as 1.36 bytes per char (UTF-8 Vietnamese + JSON escapes, measured on earlier submissions).
--order g: rank the Vietnamese docs of the proxy universe by g itself instead of the base order (top-200 per
query). Its features all appear in the LB-scored sets (interpolation), but the proxy then scores its own ranking,
so the simulated gain (+0.011 final, 2026-10-19) is an upper estimate.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from medret.config import get_config
from medret.index.meta import DB_FILE, MetaStore
from medret.submit import read_submission
from medret.system import lower_priority

BASE = "sub-vi200-rq-chunkheadmean-v1"
BYTES_PER_CHAR = 1.36
STEP = 5


def f2(hit: float, k: int, g: float) -> float:
    if k == 0 or hit <= 0 or g <= 0:
        return 0.0
    p, r = hit / k, hit / g
    return 5 * p * r / (4 * p + r)


def load(cfg, max_chars: int, order_src: str = "base", top: int = 200, scored: str | None = None):
    """scored: proxy output to use (default data/dev/proxy_pairs_scored.parquet; the 0.3241 run used the v1 copy in
    data/dev/proxy_v1_20261019/)."""
    path = scored or cfg.paths.dev_dir / "proxy_pairs_scored.parquet"
    u = pd.read_parquet(path, columns=["query_id", "doc_id", "g", "lang"])
    if order_src == "g" or order_src.startswith("rrf:"):
        v = u[u["lang"] == "vi"].sort_values(["query_id", "g"], ascending=[True, False])
        order = {int(q): [int(d) for d in ds[:top]] for q, ds in v.groupby("query_id")["doc_id"].agg(list).items()}
        if order_src.startswith("rrf:"):  # e.g. rrf:tm_mean_q -> RRF(k=60) of the g order and bench channel ranks
            ranks = [pd.DataFrame([(q, d, r) for q, ds in order.items() for r, d in enumerate(ds)],
                                  columns=["query_id", "doc_id", "rank"])]
            ranks += [pd.read_parquet(cfg.paths.dev_dir / "bench" / f"{c}.parquet")[["query_id", "doc_id", "rank"]]
                      for c in order_src[4:].split(",")]
            f = pd.concat(ranks)
            f = f.assign(s=1 / (60 + f["rank"] + 1)).groupby(["query_id", "doc_id"])["s"].sum().reset_index()
            f = f.sort_values(["query_id", "s"], ascending=[True, False])
            order = {int(q): [int(d) for d in ds[:top]] for q, ds in f.groupby("query_id")["doc_id"].agg(list).items()}
    else:
        sub = read_submission(cfg.paths.runs_dir / BASE / "submission.json")
        order = {x["id"]: [int(d) for d in x["relevant_docs"]] for x in sub}
    gq = u.groupby("query_id")["g"].sum()
    g = u.set_index(["query_id", "doc_id"])["g"]
    docs = {d for ds in order.values() for d in ds}
    n = MetaStore(cfg.paths.index_dir / DB_FILE).documents(docs).set_index("doc_id")["n_chars"].clip(upper=max_chars)
    G, L = {}, {}
    for q, ds in order.items():
        G[q] = np.array([g.get((q, d), 0.0) for d in ds])
        L[q] = np.array([n.get(d, 0) for d in ds], dtype=float)
    return order, G, L, gq


def best_k(gv: np.ndarray, gtot: float, ks=range(50, 201, 10)) -> int:
    cum = np.cumsum(gv)
    return max(ks, key=lambda k: f2(cum[min(k, len(gv)) - 1], min(k, len(gv)), gtot))


def allocate_n(G, L, gq, budget_chars: float, adaptive: bool, n_max: int = 200) -> dict[int, int]:
    """Uniform: the largest N with sum of chars <= budget. Adaptive: greedy STEP-doc increments by expected chunk
    F2 gain per char, across all queries."""
    qs = list(G)
    if not adaptive:
        for n in range(n_max, 0, -STEP):
            if sum(L[q][:n].sum() for q in qs) <= budget_chars:
                return {q: n for q in qs}
        return {q: 0 for q in qs}
    alloc = {q: 0 for q in qs}
    cum = {q: np.concatenate([[0.0], np.cumsum(G[q])]) for q in qs}
    lc = {q: np.concatenate([[0.0], np.cumsum(L[q])]) for q in qs}

    def gain(q):
        a, b = alloc[q], min(alloc[q] + STEP, len(G[q]))
        if b == a:
            return None
        df = f2(cum[q][b], b, gq[q]) - f2(cum[q][a], a, gq[q])
        return df / max(lc[q][b] - lc[q][a], 1.0), b

    heap = []
    for q in qs:
        r = gain(q)
        if r:
            heapq.heappush(heap, (-r[0], q, r[1]))
    used = 0.0
    while heap:
        neg, q, b = heapq.heappop(heap)
        cost = lc[q][b] - lc[q][alloc[q]]
        if used + cost > budget_chars or -neg <= 0:
            continue
        used += cost
        alloc[q] = b
        r = gain(q)
        if r:
            heapq.heappush(heap, (-r[0], q, r[1]))
    return alloc


def evaluate(G, gq, ks: dict[int, int], ns: dict[int, int]) -> tuple[float, float]:
    doc = np.mean([f2(G[q][: ks[q]].sum(), min(ks[q], len(G[q])), gq[q]) for q in G])
    chunk = np.mean([f2(G[q][: ns[q]].sum(), min(ns[q], len(G[q])), gq[q]) for q in G])
    return float(doc), float(chunk)


def simulate(max_chars: int) -> None:
    cfg = get_config()
    order, G, L, gq = load(cfg, max_chars)
    base_chars = sum(L[q][:65].sum() for q in G)
    print(f"base: k=200, N=65 -> {base_chars * BYTES_PER_CHAR / 1e6:.0f} MB of chunk JSON")
    k200 = {q: 200 for q in G}
    kad = {q: best_k(G[q], gq[q]) for q in G}
    ref = sum(evaluate(G, gq, k200, {q: 65 for q in G})) / 2
    kv = list(kad.values())
    print(f"adaptive k: mean {np.mean(kv):.0f}, p10/p50/p90 {np.percentile(kv, [10, 50, 90])}")
    print(f"{'budget':>9s} {'N':>12s} {'k':>9s} {'docF2':>7s} {'chunkF2':>8s} {'final':>7s} {'vs base':>8s}")
    for mult in (1.0, 1.3, 1.5):
        budget = base_chars * mult
        for ad_n in (False, True):
            ns = allocate_n(G, L, gq, budget, ad_n)
            for kname, ks in (("200", k200), ("adaptive", kad)):
                d, c = evaluate(G, gq, ks, ns)
                tag = "adaptive" if ad_n else f"uniform {next(iter(ns.values()))}"
                print(f"x{mult:<8.1f} {tag:>12s} {kname:>9s} {d:7.3f} {c:8.3f} {(d + c) / 2:7.3f} {(d + c) / 2 - ref:+8.3f}")


def split_half(w: str, reach: int = 300) -> int:
    """Split point near the middle of w, on a line break (else a space) so no word is cut."""
    mid = len(w) // 2
    for sep, r in (("\n", reach), (" ", reach // 3)):
        cands = [i for i in (w.rfind(sep, mid - r, mid), w.find(sep, mid, mid + r)) if i > 0]
        if cands:
            return min(cands, key=lambda i: abs(i - mid)) + 1
    return mid


def build(name: str, mult: float, adaptive_k: bool, adaptive_n: bool, max_chars: int, dict_mb: int,
          order_src: str = "base", scored: str | None = None, n_max: int = 200, windows: int = 0,
          win_min_chars: int = 1500) -> None:
    """Write the submission: doc order (base run or proxy g), k per query, N whole-doc chunks per query within
    mult x the budget of 65 whole docs. windows > 0: the first `windows` whole-doc docs of each query (highest g)
    also get their whole-doc text as two halves (organisers, 2026-10-19: overlapping chunks that are not
    near-duplicates are scored separately), when that text has at least win_min_chars characters."""
    from medret.schemas import ChunkRef, QueryResult
    from medret.select.selector import full_doc_span
    from medret.submit import pack_submission, validate_submission, write_submission

    cfg = get_config()
    order, G, L, gq = load(cfg, max_chars, order_src, scored=scored)
    base_chars = sum(L[q][:65].sum() for q in G)
    ks = {q: best_k(G[q], gq[q]) for q in G} if adaptive_k else {q: 200 for q in G}
    ns = allocate_n(G, L, gq, base_chars * mult, adaptive_n, n_max=n_max)
    d, c = evaluate(G, gq, ks, ns)
    print(f"proxy: docF2 {d:.3f} chunkF2 {c:.3f}; mean k {np.mean(list(ks.values())):.0f}, "
          f"mean N {np.mean(list(ns.values())):.1f}")
    rr = pd.read_parquet(cfg.paths.runs_dir / "pool-vi-m1000" / "reranked.parquet",
                         columns=["query_id", "doc_id", "score", "start", "end"])
    best = rr.sort_values("score", ascending=False).drop_duplicates(["query_id", "doc_id"])
    best = best.set_index(["query_id", "doc_id"])[["start", "end"]]
    need = {d for q, ds in order.items() for d in ds[: ns[q]]}
    texts = MetaStore(cfg.paths.index_dir / DB_FILE).documents(need, with_text=True).set_index("doc_id")["text"]
    results, win_g = [], []
    for q in sorted(order):
        docs, chunks = order[q][: ks[q]], []
        for i, doc in enumerate(order[q][: ns[q]]):
            t = texts.get(doc, "")
            se = (int(best.at[(q, doc), "start"]), int(best.at[(q, doc), "end"])) if (q, doc) in best.index else (0, 0)
            lo, hi = full_doc_span(len(t), *se, max_chars)
            if t[lo:hi].strip():
                chunks.append(ChunkRef(str(doc), t[lo:hi]))
                if i < windows and hi - lo >= win_min_chars:  # verbatim halves of the same doc slice
                    cut = lo + split_half(t[lo:hi])
                    chunks += [ChunkRef(str(doc), t[lo:cut]), ChunkRef(str(doc), t[cut:hi])]
                    win_g.append(G[q][i])
        results.append(QueryResult(int(q), [str(x) for x in docs], chunks))
    if windows:
        mg = float(np.mean(win_g)) if win_g else 0.0
        print(f"windows: {2 * len(win_g):,} half chunks on {len(win_g):,} docs ({len(win_g) / len(order):.1f}/query), "
              f"mean proxy g of those docs {mg:.3f}: the halves raise chunk P if a half of a gold doc matches a gold "
              f"chunk more often than {0.2595 / max(mg, 1e-9):.2f} (chunk P of the best run / mean g)")
    out = cfg.paths.runs_dir / name
    out.mkdir(parents=True, exist_ok=True)
    path = write_submission(results, out / "submission.json", cfg.submission.doc_id_type)
    del texts, rr, best, results  # free ~2-3 GB before validation parses the JSON back
    gc.collect()
    errors = validate_submission(path, sorted(order))
    print("validate:", "OK" if not errors else errors[:5])
    (out / "config.json").write_text(json.dumps({
        "base_order": BASE if order_src == "base" else f"{order_src} (proxy g, vi)", "budget_mult": mult,
        "adaptive_k": adaptive_k, "adaptive_n": adaptive_n,
        "max_chars": max_chars, "lzma_dict_mb": dict_mb, "proxy_scored": str(scored or "proxy_pairs_scored.parquet"),
        "n_max": n_max, "query_order": "overlap", "windows": windows, "win_min_chars": win_min_chars,
        "mean_k": float(np.mean(list(ks.values()))), "mean_n": float(np.mean(list(ns.values()))),
        "proxy_docF2": d, "proxy_chunkF2": c}, indent=1), encoding="utf-8")
    if errors:
        return
    if path.stat().st_size < 300e6:  # 450 MB in-process crashed (2026-10-24)
        z = pack_submission(path, out / f"{name}.zip", dict_mb=dict_mb, reorder=True)
        print(f"zip: {z} ({z.stat().st_size / 1e6:.1f} MB)")
    else:  # ~1 GB JSON: this process still holds GBs of freed objects, the encoder needs ~3 GB more
        print(f"large JSON: run `python scripts/analysis/allocate_sub.py pack --name {name}` in a new process")


def pack(name: str, dict_mb: int, match_finder: str = "bt4") -> None:
    """Reorder and zip a built run in two child processes, so neither step inherits the other's memory."""
    cfg = get_config()
    out = cfg.paths.runs_dir / name
    path, z = out / "submission.json", out / f"{name}.zip"
    subprocess.run([sys.executable, "-c", f"from medret.submit import reorder_submission; reorder_submission(r'{path}')"],
                   check=True)
    subprocess.run([sys.executable, "-c", f"from medret.submit import pack_submission; "
                    f"pack_submission(r'{path}', r'{z}', dict_mb={dict_mb}, match_finder='{match_finder}')"],
                   check=True)
    print(f"zip: {z} ({z.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["simulate", "build", "pack"])
    ap.add_argument("--max-chars", type=int, default=7000)
    ap.add_argument("--name", default=None)
    ap.add_argument("--mult", type=float, default=1.5, help="whole-doc budget as a multiple of the base (N=65)")
    ap.add_argument("--adaptive-k", action="store_true")
    ap.add_argument("--adaptive-n", action="store_true")
    ap.add_argument("--dict-mb", type=int, default=256)
    ap.add_argument("--mf", default="bt4", choices=["bt4", "hc4"],
                    help="LZMA match finder: hc4 needs ~1 GB less RAM (2026-10-26: bt4 hit the RAM guard)")
    ap.add_argument("--order", default="base", help="doc order: base (best LB run), g (proxy) or rrf:<bench channels>")
    ap.add_argument("--scored", default=None, help="proxy output parquet (default data/dev/proxy_pairs_scored.parquet)")
    ap.add_argument("--n-max", type=int, default=200, help="max whole-doc chunks per query (adaptive N)")
    ap.add_argument("--windows", type=int, default=0, help="top docs per query that also get two half chunks")
    ap.add_argument("--win-min-chars", type=int, default=1500)
    a = ap.parse_args()
    lower_priority()
    if a.cmd == "simulate":
        simulate(a.max_chars)
    elif a.cmd == "pack":
        pack(a.name, a.dict_mb, a.mf)
    else:
        build(a.name, a.mult, a.adaptive_k, a.adaptive_n, a.max_chars, a.dict_mb, a.order, a.scored, a.n_max,
              a.windows, a.win_min_chars)


if __name__ == "__main__":
    main()
