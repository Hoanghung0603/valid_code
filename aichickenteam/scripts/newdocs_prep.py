"""Gom doc VI mới crawl (data/cache/crawl*/**/docs_out/*.parquet) chưa có trong data/docs → dataset Kaggle cho kaggle_newdocs/.

    python scripts/newdocs_prep.py --out <dir>      # new_docs.parquet, query.parquet, thr.json
thr.json: {qid: [điểm hạng 300 của d1 (GN doc), d2 (bge-m3 đoạn), d3 (GN đoạn)]} – doc mới chỉ vào được list nếu vượt ngưỡng này.
"""
import argparse
import glob
import json
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq

SRC = {"d1": "data/cache/dense_gn_top300_v2.json", "d2": "data/cache/kaggle_bgem3_out/dense_chunks_top.json",
       "d3": "data/cache/dense_chunks_top_v2.json"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--ver", type=int, default=2, help="bản list dense hiện tại (ngưỡng hạng 300 lấy từ đây)")
    a = ap.parse_args()
    if a.ver > 2:
        SRC.update(d1=f"data/cache/dense_gn_top300_v{a.ver}.json", d2=f"data/cache/dense_bgem3_chunks_v{a.ver}.json",
                   d3=f"data/cache/dense_chunks_top_v{a.ver}.json")
    have = set()
    for f in glob.glob("data/docs/corpus-*.parquet"):
        have |= set(map(str, pq.read_table(f, columns=["doc_id"]).column("doc_id").to_pylist()))
    parts = []
    for f in sorted(glob.glob("data/cache/crawl*/**/docs_out/*.parquet", recursive=True) + glob.glob("data/cache/pharmacity/part-*.parquet")):   # + Pharmacity cào lại (BTC 19/10)
        t = pd.read_parquet(f, columns=["doc_id", "source", "lang", "url", "title", "text"])
        parts.append(t[t.lang == "vi"])
    d = pd.concat(parts).assign(doc_id=lambda x: x.doc_id.astype(str)).drop_duplicates("doc_id")
    d = d[~d.doc_id.isin(have) & (d.text.fillna("").str.len() > 0)].reset_index(drop=True)
    print("new VI docs", len(d), "already had", len(have))
    os.makedirs(a.out, exist_ok=True)
    d.to_parquet(f"{a.out}/new_docs.parquet", index=False)
    shutil.copy("data/ViBioMIR/query.parquet", f"{a.out}/query.parquet")
    src = {k: json.load(open(f)) for k, f in SRC.items()}
    thr = {q: [src[k][q][-1][0] if len(src[k].get(q, [])) >= 300 else -1.0 for k in ("d1", "d2", "d3")] for q in src["d1"]}
    json.dump(thr, open(f"{a.out}/thr.json", "w"))


if __name__ == "__main__":
    main()
