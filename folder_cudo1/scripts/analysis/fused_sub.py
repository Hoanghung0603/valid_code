"""Submission from a fused doc order (base list + cross-encoder ranks), whole-doc chunks chosen along that order.

  python scripts/analysis/fused_sub.py --docs-from sub-vi-rrfwp-zh30docs-v1 --n-from sub-vi-rrfwp-zh30c25snap-v1 \
         --scores data/kaggle/rerank/qwen3-0.6b/scores.parquet data/kaggle/rerank/qwen3-4b/scores.parquet \
         --extra-docs 'data/parsed/documents/*/part-gloryvu-20261023-zh.parquet' --name sub-fuse3-chunksfused-v1
  python scripts/analysis/allocate_sub.py pack --name sub-fuse3-chunksfused-v1
Order: RRF (k=60) of the doc's position in --docs-from's relevant_docs (candidates outside it follow, vi then zh, by
source rank) and its rank under each score file (launch_rerank.py output; pages without text keep the base position as
their rank; pairs a model did not score add nothing). Docs: the first k of that order, k per query = len of the base
list, so the doc half changes by order only. Chunks: walking the fused order, the first N_q docs with text get a
whole-doc chunk, N_q = number of docs with chunks in --n-from for that query (same budget); vi: a window of <= 7,000
chars centred on the doc's best reranked chunk (as allocate_sub), plus its two halves for the first 10 vi docs (>= 1,500
chars); zh: the page text cut to 2,000 chars (Q&A pages, median 700). Until 2026-10-26 the whole-doc chunks followed the
proxy g order while the docs followed the fused one; chunks only match gold chunks of gold docs, so choosing them along
the better order should raise chunk recall with the doc recall.
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from medret.config import get_config
from medret.index.meta import DB_FILE, MetaStore
from medret.select.selector import full_doc_span
from medret.submit import iter_submission, validate_submission

_spec = importlib.util.spec_from_file_location("allocate_sub", Path(__file__).parent / "allocate_sub.py")
_alloc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_alloc)


def fused_order(base: dict[int, list[int]], score_files: list[str], weights: list[float] | None = None) -> pd.DataFrame:
    """weights: base list first, then one per score file (default all 1)."""
    w = weights or [1.0] * (1 + len(score_files))
    s = pd.read_parquet(score_files[0], columns=["query_id", "doc_id", "lang", "src_rank", "has_text"])
    pos = pd.DataFrame([(q, d, r) for q, ds in base.items() for r, d in enumerate(ds)],
                       columns=["query_id", "doc_id", "a_rank"])
    s = s.merge(pos, on=["query_id", "doc_id"], how="left")
    rest = s["a_rank"].isna()
    s.loc[rest, "a_rank"] = 1000 + (s.loc[rest, "lang"] == "zh") * 1000 + s.loc[rest, "src_rank"]
    s["key"] = w[0] / (61 + s["a_rank"])
    for k, f in enumerate(score_files):
        sc = pd.read_parquet(f, columns=["query_id", "doc_id", "score", "has_text"]).rename(
            columns={"score": f"s{k}", "has_text": f"t{k}"})
        s = s.merge(sc, on=["query_id", "doc_id"], how="left")
        r = s[s[f"t{k}"].fillna(False).astype(bool)].groupby("query_id")[f"s{k}"].rank(ascending=False, method="first") - 1
        s[f"r{k}"] = r
        no_text = s[f"s{k}"].notna() & ~s[f"t{k}"].fillna(False).astype(bool)
        s.loc[no_text, f"r{k}"] = s.loc[no_text, "a_rank"]
        s["key"] += (w[k + 1] / (61 + s[f"r{k}"])).fillna(0.0)
    return s.sort_values(["query_id", "key"], ascending=[True, False])


def proxy_n(order: dict, lang: dict, n_base: dict, max_chars: int, zh_chars: int, deep: bool, cfg,
            twins: bool = False, long: bool = False) -> dict:
    spec = importlib.util.spec_from_file_location("rerank_proxy", Path(__file__).parent / "rerank_proxy.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    probs, gq = rp.doc_probs(deep, twins, long)
    pmap = probs.set_index(["query_id", "doc_id"])["p"]
    docs = {d for ds in order.values() for d in ds}
    store = MetaStore(cfg.paths.index_dir / DB_FILE)
    n_chars = store.documents(docs).set_index("doc_id")["n_chars"]
    store.con.close()
    G, L = {}, {}
    for q, ds in order.items():
        G[q] = np.array([pmap.get((q, d), 0.0) for d in ds])
        L[q] = np.array([min(n_chars.get(d, zh_chars), zh_chars if lang.get(d) == "zh" else max_chars) for d in ds],
                        dtype=float)
    budget = sum(L[q][: n_base.get(q, 0)].sum() for q in order)
    ns = _alloc.allocate_n(G, L, gq.to_dict(), budget, adaptive=True, n_max=300)
    v = np.array(list(ns.values()))
    print(f"N per query from the rerank proxy: mean {v.mean():.1f} (base {np.mean(list(n_base.values())):.1f}), "
          f"p10/50/90 {np.percentile(v, [10, 50, 90])}, budget {budget / 1e6:.0f}M chars")
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-from", required=True)
    ap.add_argument("--n-from", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="RRF weights: base list, then one per score file (default all 1)")
    ap.add_argument("--extra-docs", default=None)
    ap.add_argument("--n-proxy", action="store_true",
                    help="whole-doc count per query by expected chunk F2 per char under rerank_proxy.py's p, same char "
                         "budget as --n-from's counts along this order (allocate_sub.allocate_n)")
    ap.add_argument("--deep", action="store_true", help="rerank_proxy candidates include the deeper pool")
    ap.add_argument("--twins", action="store_true",
                    help="docs without text take the page text of their http/https twin (canonical_id, exact_equivalent "
                         "in the team snapshot's corpus_annotated.parquet); proxy with the twins' 4B scores")
    ap.add_argument("--long", action="store_true", help="--n-proxy with the proxy fitted on the long-context 4B scores")
    ap.add_argument("--annotated", default="data/external/gloryvu_20261023/corpus_annotated.parquet")
    ap.add_argument("--k-file", default=None,
                    help="parquet (query_id, k): docs per query, e.g. from rerank_proxy.py, instead of the base list size")
    ap.add_argument("--max-chars", type=int, default=7000)
    ap.add_argument("--zh-chars", type=int, default=2000)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--win-min-chars", type=int, default=1500)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    cfg = get_config()
    runs = cfg.paths.runs_dir
    base = {x["id"]: [int(d) for d in x["relevant_docs"]] for x in iter_submission(runs / a.docs_from / "submission.json")}
    n_q = {x["id"]: len({c["doc_id"] for c in x["relevant_chunks"]})
           for x in iter_submission(runs / a.n_from / "submission.json")}
    s = fused_order(base, a.scores, a.weights)
    kq = ({int(q): int(k) for q, k in pd.read_parquet(a.k_file).itertuples(index=False)} if a.k_file
          else {q: len(ds) for q, ds in base.items()})
    order = s.groupby("query_id")["doc_id"].agg(list).to_dict()
    lang = dict(zip(s["doc_id"], s["lang"], strict=False))
    if a.n_proxy:
        n_q = proxy_n(order, lang, n_q, a.max_chars, a.zh_chars, a.deep, cfg, a.twins, a.long)
    # texts for every doc that can get a chunk: the first N_q + slack of each order
    need = {d for q, ds in order.items() for d in ds[: n_q.get(q, 0) + 40]}
    twin_of = {}
    if a.twins:  # read the twin's page for docs that have no text under their own id
        ann = pd.read_parquet(a.annotated, columns=["id", "canonical_id", "dup_type"])
        ann = ann[(ann["dup_type"] == "exact_equivalent") & (ann["id"] != ann["canonical_id"])]
        twin_of = {int(d): int(c) for d, c in zip(ann["id"], ann["canonical_id"], strict=True) if d in need}
        need = need | set(twin_of.values())
    # one query, one connection: a second MetaStore opened right after the first crashed DuckDB on 2026-10-28
    store = MetaStore(cfg.paths.index_dir / DB_FILE)
    text = store.documents(need, with_text=True).set_index("doc_id")["text"]
    store.con.close()
    if a.extra_docs:
        lack = sorted(need - set(text.index))
        extra = [pq.read_table(f, columns=["doc_id", "text"], filters=[("doc_id", "in", lack)]).to_pandas()
                 for f in glob.glob(a.extra_docs)]
        if extra:
            text = pd.concat([text, pd.concat(extra).set_index("doc_id")["text"]])
    rr = pd.read_parquet(runs / "pool-vi-m1000" / "reranked.parquet",
                         columns=["query_id", "doc_id", "score", "start", "end"])
    best = rr.sort_values("score", ascending=False).drop_duplicates(["query_id", "doc_id"])
    best = best.set_index(["query_id", "doc_id"])[["start", "end"]]
    del rr
    gc.collect()
    out = runs / a.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "submission.json"
    qids, n_chunks, n_zh = sorted(order), 0, 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("[")
        for i, q in enumerate(qids):
            docs = order[q][: kq.get(q, len(base.get(q, [])))]
            chunks, got, n_win = [], 0, 0
            for d in order[q]:
                if got >= n_q.get(q, 0):
                    break
                t = text.get(d, "")
                if (not isinstance(t, str) or not t.strip()) and d in twin_of:
                    t = text.get(twin_of[d], "")  # the same page crawled under the twin id (verbatim page text)
                if not isinstance(t, str) or not t.strip():
                    continue
                if lang.get(d) == "zh":
                    chunks.append({"doc_id": int(d), "chunk_text": t[: a.zh_chars]})
                    n_zh += 1
                else:
                    se = (int(best.at[(q, d), "start"]), int(best.at[(q, d), "end"])) if (q, d) in best.index else (0, 0)
                    lo, hi = full_doc_span(len(t), *se, a.max_chars)
                    w = t[lo:hi]  # verbatim document text
                    chunks.append({"doc_id": int(d), "chunk_text": w})
                    if n_win < a.windows and len(w) >= a.win_min_chars:
                        cut = _alloc.split_half(w)
                        chunks += [{"doc_id": int(d), "chunk_text": w[:cut]}, {"doc_id": int(d), "chunk_text": w[cut:]}]
                        n_win += 1
                got += 1
            n_chunks += len(chunks)
            item = {"id": int(q), "relevant_docs": [int(d) for d in docs], "relevant_chunks": chunks}
            f.write((", " if i else "") + json.dumps(item, ensure_ascii=False))
        f.write("]")
    del text
    gc.collect()
    errors = validate_submission(path, qids)
    info = {"docs_from": a.docs_from, "n_from": a.n_from, "scores": a.scores, "weights": a.weights, "k_file": a.k_file,
            "twins": a.twins, "long": a.long,
            "max_chars": a.max_chars,
            "zh_chars": a.zh_chars, "windows": a.windows, "chunks_per_query": n_chunks / len(qids),
            "zh_chunks_per_query": n_zh / len(qids)}
    (out / "config.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"{len(qids)} queries; {info['chunks_per_query']:.1f} chunks/query ({info['zh_chunks_per_query']:.1f} zh); "
          f"validate: {'OK' if not errors else errors[:5]}")


if __name__ == "__main__":
    main()
