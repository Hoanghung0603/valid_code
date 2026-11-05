"""Pairs datasets chosen from earlier rerank scores: the 4B part of the deeper pool, and the twin pages.

  python scripts/kaggle/make_score_pairs.py deep4b --out data/kaggle/rerank_pairs_deep4b   # -> rerank_qwen3-4b-deep.json
  python scripts/kaggle/make_score_pairs.py twins --out data/kaggle/rerank_pairs_twins     # -> rerank_qwen3-4b-twins.json
deep4b (S9, 2026-10-27): of the deeper pool (launch_rerank.py make-pairs --vi-top 400 --zh-top 200 --exclude
data/kaggle/rerank_pairs, scored by 0.6B: qwen3-0.6b-deep), the top-75 pairs per query by 0.6B score among those
with page text, in score order (90,000 pairs). Texts are read on Kaggle from the team snapshot.
twins (S10, 2026-10-28): every zh candidate without page text (qwen3-0.6b-all, has_text false) whose id is an
`exact_equivalent` http/https twin of another id in the snapshot's corpus_annotated.parquet (text_id = that id;
81,000 pairs, all www.120ask.com), with the twins' texts bundled ("<page title>\\n<page text>" of the snapshot,
cut to 1,600 chars as the runner does). Both were built ad hoc at the time; this script reproduces the uploaded
datasets row for row.
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from medret.config import PROJECT_ROOT, get_config
from medret.data.dataset import load_queries

R = PROJECT_ROOT / "data" / "kaggle" / "rerank"
SNAP = PROJECT_ROOT / "data" / "external" / "gloryvu_20261023"
COLS = ["query_id", "doc_id", "src_rank", "lang"]


def deep4b(top: int = 75) -> pd.DataFrame:
    s = pd.read_parquet(R / "qwen3-0.6b-deep" / "scores.parquet")
    s = s[s["has_text"]].sort_values(["query_id", "score"], ascending=[True, False])
    return s.groupby("query_id").head(top)[COLS]


def twins() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_parquet(R / "qwen3-0.6b-all" / "scores.parquet")
    ann = pd.read_parquet(SNAP / "corpus_annotated.parquet", columns=["id", "canonical_id", "dup_type"])
    eq = ann[(ann["dup_type"] == "exact_equivalent") & (ann["id"] != ann["canonical_id"])].set_index("id")["canonical_id"]
    c = s[(s["lang"] == "zh") & ~s["has_text"]].copy()
    c["text_id"] = c["doc_id"].map(eq)
    c = c[c["text_id"].notna()].astype({"text_id": "int64"})[COLS + ["text_id"]]
    ids = sorted(set(c["text_id"]))
    parts = [pq.read_table(f, columns=["doc_id", "chunk_idx", "context", "page_title"],
                           filters=[("doc_id", "in", ids)]).to_pandas()
             for f in sorted(glob.glob(str(SNAP / "chunks_zh-*.parquet")))]
    ch = pd.concat(parts).sort_values(["doc_id", "chunk_idx"])
    head = ch.groupby("doc_id").agg(title=("page_title", "first"), text=("context", "\n".join))
    texts = pd.DataFrame({"doc_id": head.index.astype("int64"),
                          "text": [(f"{t}\n{x}" if isinstance(t, str) and t.strip() else x)[:1600]
                                   for t, x in zip(head["title"], head["text"], strict=True)]})
    return c[c["text_id"].isin(set(texts["doc_id"]))], texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["deep4b", "twins"])
    ap.add_argument("--per-shard", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    texts = None
    pairs = deep4b() if a.cmd == "deep4b" else None
    if a.cmd == "twins":
        pairs, texts = twins()
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    qids = sorted(pairs["query_id"].unique())
    for k in range(0, len(qids), a.per_shard):
        pairs[pairs["query_id"].isin(qids[k: k + a.per_shard])].to_parquet(
            a.out / f"pairs-{k // a.per_shard:04d}.parquet", index=False)
    if texts is not None:
        texts.to_parquet(a.out / "texts-0000.parquet", index=False)
    load_queries(get_config())[["id", "query"]].to_parquet(a.out / "queries.parquet", index=False)
    print(f"{len(pairs):,} pairs, {len(qids)} queries -> {a.out}")


if __name__ == "__main__":
    main()
