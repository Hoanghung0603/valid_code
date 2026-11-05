"""Kiểm tra chỉ mục BigIndex: tốc độ 1.200 truy vấn thật, và tự nhất quán (dùng tiêu đề một doc làm truy vấn, doc đó phải nằm trong top-10)."""
import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd

from bigindex import BigIndex, shard_docs
from bm25 import tokenize


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", default=None, help="bỏ qua nếu truy vấn khác ngôn ngữ chỉ mục (vd chỉ mục zh)")
    ap.add_argument("--probe-shards", type=int, default=3)
    a = ap.parse_args()

    t = time.time()
    idx = BigIndex(Path(a.index))
    lang = idx.meta["lang"]
    print(f"nạp {time.time() - t:.1f}s: {idx.n} doc, {idx.meta['postings']} posting, {idx.meta['unique_ids']} id duy nhất")

    if a.queries:
        qs = pd.read_parquet(a.queries)["query"].tolist()
        lat, empty = [], 0
        for q in qs:
            t = time.time()
            hits = idx.search(sorted(set(tokenize(q, lang))), k=10)
            lat.append(time.time() - t)
            empty += not hits
        lat.sort()
        print(f"{len(qs)} truy vấn: trung vị {lat[len(lat) // 2] * 1000:.0f}ms, p95 {lat[int(len(lat) * .95)] * 1000:.0f}ms, "
              f"max {lat[-1] * 1000:.0f}ms, rỗng {empty}")

    rng = random.Random(0)
    shards = idx.shards[:: max(1, len(idx.shards) // a.probe_shards)][: a.probe_shards]
    ok = tot = 0
    for sp in shards:
        # zh viết liền không có dấu cách: đếm ký tự; vi: đếm từ
        long_enough = (lambda t: len(t.strip()) >= 6) if lang == "zh" else (lambda t: len(t.split()) >= 4)
        docs = [d for d in shard_docs(Path(sp), lang) if long_enough(d["title"])]
        for d in rng.sample(docs, min(100, len(docs))):
            hits = {idx.doc_id(i) for i, _ in idx.search(sorted(set(tokenize(d["title"], lang))), k=10)}
            ok += d["id"] in hits
            tot += 1
    print(f"tự nhất quán (tiêu đề -> doc trong top10): {ok}/{tot} ({ok / max(tot, 1):.0%}) trên {len(shards)} shard")


if __name__ == "__main__":
    sys.exit(main())
