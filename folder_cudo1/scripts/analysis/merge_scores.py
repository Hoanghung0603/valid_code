"""Merged Qwen3-Reranker score files read by rerank_proxy.py and fused_sub.py, from the per-run Kaggle outputs.

  python scripts/analysis/merge_scores.py
  python scripts/analysis/merge_long_scores.py        # then: qwen3-4b-all-tw + qwen3-4b-long -> qwen3-4b-all-tw-long
Each Kaggle run (launch_rerank.py run <config>) writes data/kaggle/rerank/<run>/scores.parquet; the merges are
  qwen3-0.6b-all  = qwen3-0.6b (first pool) + qwen3-0.6b-deep (deeper pool)
  qwen3-4b-full   = qwen3-4b (first half of the first pool) + qwen3-4b-tail (second half, configs -tail-p1/-p2)
  qwen3-4b-all    = qwen3-4b-full + qwen3-4b-deep (top-75 of the deeper pool by 0.6B)
  qwen3-4b-all-tw = the pairs of qwen3-4b-all that qwen3-4b-twins does not have, then every pair of qwen3-4b-twins
                    (text-less zh docs read through their http/https twin), which replaces the 4B score of the
                    39,780 such pairs already in qwen3-4b-all and adds 41,220
The parts of each concatenation are disjoint (checked in _write).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

R = Path("data/kaggle/rerank")
KEY = ["query_id", "doc_id"]
COLS = ["query_id", "doc_id", "src_rank", "lang", "score", "has_text"]


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(R / name / "scores.parquet")


def _write(df: pd.DataFrame, name: str) -> None:
    assert not df.duplicated(KEY).any(), f"{name}: duplicate pairs"
    (R / name).mkdir(parents=True, exist_ok=True)
    df[COLS].to_parquet(R / name / "scores.parquet", index=False)
    print(f"{name}: {len(df):,} pairs")


def main() -> None:
    _write(pd.concat([_read("qwen3-0.6b"), _read("qwen3-0.6b-deep")], ignore_index=True), "qwen3-0.6b-all")
    full = pd.concat([_read("qwen3-4b"), _read("qwen3-4b-tail")], ignore_index=True)
    _write(full, "qwen3-4b-full")
    al = pd.concat([full, _read("qwen3-4b-deep")], ignore_index=True)
    _write(al, "qwen3-4b-all")
    # row order matters: fused_sub.py breaks score ties by row order (rank method "first")
    tw = _read("qwen3-4b-twins")
    in_tw = al.set_index(KEY).index.isin(tw.set_index(KEY).index)
    _write(pd.concat([al[~in_tw], tw.assign(has_text=tw["has_text"].astype(bool))], ignore_index=True),
           "qwen3-4b-all-tw")
    print(f"twins: {in_tw.sum():,} pairs replaced, {len(tw) - in_tw.sum():,} added")


if __name__ == "__main__":
    main()
