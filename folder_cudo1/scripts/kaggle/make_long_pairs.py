"""Pairs for a long-context rerank pass: the first `top` candidates of the current fused order, with page texts bundled.

  python scripts/kaggle/make_long_pairs.py --top 150 --chars 4000 --out data/kaggle/rerank_pairs_long
  python scripts/kaggle/make_long_pairs.py --top 100 --queries 20 --chars 4000 --out data/kaggle/rerank_pairs_long_pilot
Order = fused_sub.fused_order (base list of the 0.3345 run x1, 0.6B rank x0.5, 4B rank x1.5, 4B with the twin pages) as
in the 0.3724 run. Texts = page title + page text of the team snapshot (Oct 23; vi clean chunks, zh chunks), cut to
`chars` characters, read through the http/https twin (text_id) when the doc has no page under its own id; bundled as
texts-*.parquet so the kernel needs no access to the snapshot (rerank_runner.py reads them first). Motivation
(2026-10-28): the reranks so far read 1,600 chars (512 tokens) of each page.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from medret.config import PROJECT_ROOT, get_config
from medret.data.dataset import load_queries
from medret.submit import iter_submission

SNAP = PROJECT_ROOT / "data" / "external" / "gloryvu_20261023"
SCORES = ["data/kaggle/rerank/qwen3-0.6b-all/scores.parquet", "data/kaggle/rerank/qwen3-4b-all-tw/scores.parquet"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=150)
    ap.add_argument("--chars", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=0, help="first n queries only (pilot)")
    ap.add_argument("--per-shard", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = get_config()
    spec = importlib.util.spec_from_file_location("fused_sub", PROJECT_ROOT / "scripts" / "analysis" / "fused_sub.py")
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    base = {x["id"]: [int(d) for d in x["relevant_docs"]]
            for x in iter_submission(cfg.paths.runs_dir / "sub-vi-rrfwp-zh30docs-v1" / "submission.json")}
    o = fs.fused_order(base, SCORES, [1, 0.5, 1.5])
    o = o[o.groupby("query_id").cumcount() < a.top][["query_id", "doc_id", "src_rank", "lang"]]
    if a.queries:
        o = o[o["query_id"].isin(sorted(o["query_id"].unique())[: a.queries])]
    ann = pd.read_parquet(SNAP / "corpus_annotated.parquet", columns=["id", "canonical_id"]).set_index("id")["canonical_id"]
    o["text_id"] = o["doc_id"].map(ann).fillna(o["doc_id"]).astype("int64")
    ids = sorted(set(o["text_id"]))
    parts = []
    for f in sorted(glob.glob(str(SNAP / "chunks_vi-*.parquet")) + glob.glob(str(SNAP / "chunks_zh-*.parquet"))):
        parts.append(pq.read_table(f, columns=["doc_id", "chunk_idx", "context", "page_title"],
                                   filters=[("doc_id", "in", ids)]).to_pandas())
    ch = pd.concat(parts).sort_values(["doc_id", "chunk_idx"])
    head = ch.groupby("doc_id").agg(title=("page_title", "first"), text=("context", "\n".join))
    texts = pd.DataFrame({"doc_id": head.index.astype("int64"),
                          "text": [(f"{t}\n{x}" if isinstance(t, str) and t.strip() else x)[: a.chars]
                                   for t, x in zip(head["title"], head["text"], strict=True)]})
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    qids = sorted(o["query_id"].unique())
    for k in range(0, len(qids), a.per_shard):
        o[o["query_id"].isin(qids[k: k + a.per_shard])].to_parquet(a.out / f"pairs-{k // a.per_shard:04d}.parquet",
                                                                   index=False)
    texts.to_parquet(a.out / "texts-0000.parquet", index=False)
    load_queries(cfg)[["id", "query"]].to_parquet(a.out / "queries.parquet", index=False)
    miss = 1 - o["text_id"].isin(set(texts["doc_id"])).mean()
    print(f"{len(o):,} pairs, {len(qids)} queries, {len(texts):,} texts ({miss:.1%} of pairs without text), "
          f"{texts['text'].str.len().mean():.0f} chars on average -> {a.out}")


if __name__ == "__main__":
    main()
