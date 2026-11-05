"""Crawl một bộ URL (từ split_crawl_sets.py) bằng crawl.polite: HTML thô vào <out>/<nhóm site>/part-*.parquet.

Lịch sự: tối đa 1 request/s mỗi site (cả 39.net tính là 1 site), tôn trọng robots.txt, bị chặn thì nghỉ và lùi dần
tới 30 phút rồi thử lại; không tìm cách vượt captcha / chặn bot. Chạy lại cùng lệnh để resume.

Ví dụ:
  scripts/crawl_set.py --todo build/crawl_sets/set_2.parquet --out build/pages_set2
  scripts/crawl_set.py --todo build/crawl_sets/set_2.parquet --out build/pages_test --limit-per-group 2   # thử nhanh
  scripts/crawl_set.py --todo build/crawl_sets/set_2.parquet --out build/pages_set2 --status
Dừng êm: tạo file <out>/STOP hoặc Ctrl+C. Gộp kết quả: chép <out>/<nhóm>/part-*.parquet vào build/pages/<nhóm>/.
"""
import _bootstrap  # noqa: F401
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from config import PAGES
from crawl.polite import crawl_groups
from crawl.sets import FINAL, shard_labels

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--todo", required=True, help="set_N.parquet (id, url, group)")
ap.add_argument("--out", default=str(PAGES))
ap.add_argument("--status", action="store_true", help="in tiến độ theo site rồi thoát")
ap.add_argument("--interval", type=float, default=1.0, help="giây tối thiểu giữa 2 request bắt đầu, mỗi site")
ap.add_argument("--connect-timeout", type=float, default=5)
ap.add_argument("--read-timeout", type=float, default=15)
ap.add_argument("--retries", type=int, default=3)
ap.add_argument("--backoff", type=float, default=30, help="lần nghỉ đầu khi bị chặn; gấp đôi mỗi lần")
ap.add_argument("--max-backoff", type=float, default=1800)
ap.add_argument("--neterr-pause", type=float, default=10, help="nghỉ sau timeout / lỗi kết nối")
ap.add_argument("--stall-errors", type=int, default=5, help="số lỗi mạng liên tiếp thì coi như bị chặn")
ap.add_argument("--groups", nargs="*", help="chỉ crawl các nhóm site này")
ap.add_argument("--limit-per-group", type=int, default=0)
ap.add_argument("--insecure", action="store_true", help="bỏ kiểm tra TLS (một số chứng chỉ .gov.vn)")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
todo = pd.read_parquet(args.todo, columns=["id", "url", "group"])
stored = shard_labels(out)
done = stored[stored.label.astype(str).str.startswith(FINAL)].drop_duplicates("id")

if args.status:
    t = todo.groupby("group").size().rename("total").to_frame()
    t["done"] = done[done.id.isin(todo.id)].groupby("group").size()
    t["ok"] = done[(done.label == "ok") & done.id.isin(todo.id)].groupby("group").size()
    t = t.fillna(0).astype(int)
    t["pct"] = (100 * t.done / t.total).round(1)
    print(t.sort_values("total", ascending=False).to_string())
    print(f"\nTỔNG {t.done.sum():,}/{t.total.sum():,} ({100 * t.done.sum() / max(1, t.total.sum()):.1f}%), ok {t.ok.sum():,}")
    p = out / "_progress.json"
    if p.exists():
        j = json.loads(p.read_text())
        print(f"ghi tiến độ lần cuối {time.strftime('%Y-%m-%d %H:%M', time.localtime(j['t']))}, {j['per_min']:.0f}/phút")
    raise SystemExit

todo = todo[~todo.id.isin(done.id)]
if args.groups:
    todo = todo[todo.group.isin(args.groups)]
if args.limit_per_group:
    todo = todo.groupby("group").head(args.limit_per_group)
groups = {g: d.sort_values("id")[["id", "url"]].to_dict("records") for g, d in todo.groupby("group")}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(out / "crawl.log", encoding="utf-8"), logging.StreamHandler()])
logging.getLogger("httpx").setLevel(logging.WARNING)
try:
    asyncio.run(crawl_groups(groups, out, args))
except KeyboardInterrupt:
    logging.info("đã dừng; shard đã ghi được giữ lại, chạy lại cùng lệnh để tiếp tục")
