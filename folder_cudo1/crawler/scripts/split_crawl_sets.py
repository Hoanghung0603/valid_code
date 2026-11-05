"""Chia các URL còn phải crawl thành N bộ để nhiều máy cùng crawl -> build/crawl_sets/set_N.parquet (id, url, group).

"Còn phải crawl" = URL trong corpus, trừ host trong crawl.hosts.BLOCKED / IMPERSONATE, trừ id đã xong trong các
thư mục shard (--done, mặc định build/pages) và trong build/crawl.sqlite. Mỗi site được cắt đều cho N bộ.

Ví dụ:
  scripts/split_crawl_sets.py --sets 3
  scripts/split_crawl_sets.py --sets 3 --ids my_canonical_ids.parquet --done build/pages other/pages
"""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

import pandas as pd

from config import CORPUS, CRAWL_DB, CRAWL_SETS, PAGES
from crawl.hosts import BLOCKED, IMPERSONATE
from crawl.sets import done_ids, remaining, split_sets

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sets", type=int, default=3)
ap.add_argument("--done", nargs="*", default=[str(PAGES)], help="thư mục shard đã crawl (<nhóm>/part-*.parquet)")
ap.add_argument("--ids", help="parquet có cột id: chỉ chia các id này (vd danh sách đã bỏ trùng / bỏ lạc đề)")
ap.add_argument("--out", default=str(CRAWL_SETS))
args = ap.parse_args()

corpus = pd.read_parquet(CORPUS, columns=["id", "url"])
only = pd.read_parquet(args.ids, columns=["id"]).id if args.ids else None
todo = remaining(corpus, done_ids(args.done, CRAWL_DB), exclude_hosts=set(BLOCKED) | IMPERSONATE, only_ids=only)
print(f"còn {len(todo):,} URL trong {todo.group.nunique()} nhóm site")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for k, s in enumerate(split_sets(todo, args.sets), 1):
    s.to_parquet(out / f"set_{k}.parquet", index=False)
    top = s.groupby("group").size().sort_values(ascending=False).head(4)
    print(f"set_{k}.parquet: {len(s):,} URL  " + ", ".join(f"{g} {n:,}" for g, n in top.items()))
