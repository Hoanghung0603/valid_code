"""Bench channel (data/dev/bench/<name>.parquet: query_id, doc_id, rank) from a slice of a teammate's probe submission.

  python scripts/analysis/probe_channel.py --zip probe_recall_vi1000_en300_zh300.zip --start 1300 --end 1600 \
         --name tm_zh_q
The teammate's probe lists, per query, 1,000 vi docs, then 300 en, then 300 zh, each block in rank order of chunk-level
BGE-M3 dense retrieval (2026-10-24). Its LB doc recall (0.885 vs 0.699 for vi1000), together with our best + en30 run
(doc recall unchanged at 0.5089), puts about 13 gold docs per query in the zh block. Our own zh ranking would need the
~100k zh docs we have not crawled, so this channel takes the teammate's order as is.
"""

from __future__ import annotations

import argparse
import json
import zipfile

import pandas as pd

from medret.config import get_config

EN_IDS = (4_421_055, 7_466_161)  # PubMed id range of corpus v2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    with zipfile.ZipFile(a.zip) as z:
        sub = json.loads(z.read(z.namelist()[0]).decode("utf-8"))
    rows = [(int(x["id"]), int(d), r) for x in sub for r, d in enumerate(x["relevant_docs"][a.start:a.end])]
    df = pd.DataFrame(rows, columns=["query_id", "doc_id", "rank"])
    en = df["doc_id"].between(*EN_IDS).mean()
    out = get_config().paths.dev_dir / "bench" / f"{a.name}.parquet"
    df.to_parquet(out, index=False)
    print(f"{df['query_id'].nunique()} queries, {len(df) / df['query_id'].nunique():.0f} docs/query "
          f"({en:.1%} in the PubMed id range) -> {out}")


if __name__ == "__main__":
    main()
