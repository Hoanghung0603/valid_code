"""Tìm top-k doc cho mọi truy vấn MỘT lần và lưu ra JSON, để dựng nhiều biến thể bài nộp mà không phải tìm lại (pha tìm doc mất ~20 phút).

Đồng thời đo thời gian: ghi mỗi 100 truy vấn số giây và số posting mà truy vấn phải cộng, để biết chậm vì truy vấn nặng (term phổ biến)
hay vì đọc đĩa. Định dạng: {"<qid>": [[doc_id, chỉ số trong BigIndex, điểm], ...]} giảm dần theo điểm.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

from bigindex import BigIndex
from bm25 import tokenize


def postings_touched(idx: BigIndex, tokens: list[str]) -> int:
    hs = {zlib.crc32(t.encode("utf-8")) & idx.mask for t in tokens}
    return sum(int(idx.indptr[h + 1]) - int(idx.indptr[h]) for h in hs)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True, help="parquet (id, query) hoặc JSONL của translate_queries (id, zh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N truy vấn đầu (để đo tốc độ)")
    a = ap.parse_args()

    if a.queries.endswith(".parquet"):
        import pandas as pd
        q = pd.read_parquet(a.queries)
        items = list(zip(q["id"].astype(int), q["query"]))
    else:
        from bipipeline import load_zh_map
        items = sorted(load_zh_map(Path(a.queries)).items())
    if a.limit:
        items = items[: a.limit]

    idx = BigIndex(Path(a.index))
    lang = idx.meta["lang"]
    out: dict[str, list] = {}
    t0 = t = time.time()
    post = 0
    for n, (qid, text) in enumerate(items, 1):
        toks = sorted(set(tokenize(text, lang)))
        post += postings_touched(idx, toks)
        hits = idx.search(toks, k=a.k)
        out[str(int(qid))] = [[idx.doc_id(i), i, round(s, 4)] for i, s in hits]
        if n % 100 == 0 or n == len(items):
            print(f"{n}/{len(items)}: {time.time() - t:.1f}s cho {'100' if n % 100 == 0 else n % 100} câu cuối, "
                  f"TB {post / (100 if n % 100 == 0 else n % 100) / 1e6:.2f}M posting/câu, tổng {time.time() - t0:.0f}s", flush=True)
            t, post = time.time(), 0
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"ghi {a.out}: {len(out)} câu, k={a.k}", flush=True)


if __name__ == "__main__":
    main()
