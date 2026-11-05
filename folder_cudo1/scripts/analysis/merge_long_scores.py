"""4B scores with the long-context run (4,000 chars, 1,024 tokens) replacing the 1,600-char scores where it exists.

  python scripts/analysis/merge_long_scores.py   # -> data/kaggle/rerank/qwen3-4b-all-tw-long/scores.parquet
Base: qwen3-4b-all-tw (both pools + twin pages). Long: qwen3-4b-long, the top-150 per query of the S10 fused order
(scripts/kaggle/make_long_pairs.py); twins read their twin's page as in the base. Pairs outside the top-150 keep the
1,600-char score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

R = Path("data/kaggle/rerank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(R / "qwen3-4b-all-tw/scores.parquet"))
    ap.add_argument("--long", default=str(R / "qwen3-4b-long/scores.parquet"))
    ap.add_argument("--out", default=str(R / "qwen3-4b-all-tw-long/scores.parquet"))
    a = ap.parse_args()
    base = pd.read_parquet(a.base)
    lg = pd.read_parquet(a.long, columns=["query_id", "doc_id", "score", "has_text"])
    m = base.merge(lg, on=["query_id", "doc_id"], how="left", suffixes=("", "_long"))
    hit = m["score_long"].notna()
    m.loc[hit, "score"] = m.loc[hit, "score_long"]
    m.loc[hit, "has_text"] = m.loc[hit, "has_text_long"].astype(bool)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m[base.columns].to_parquet(out, index=False)
    print(f"{hit.sum():,} of {len(m):,} pairs take the long score ({m.loc[hit, 'query_id'].nunique()} queries; "
          f"{len(lg) - hit.sum()} long pairs not in the base) -> {out}")


if __name__ == "__main__":
    main()
