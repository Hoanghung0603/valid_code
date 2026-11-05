"""Fetch Long Chau pages from Common Crawl WARC records into crawl_home-format shards (no request to Long Chau).

Each record is the newest status-200 capture in D:/r2ai_data/crawl_home/cc_longchau_index.json. Output:
<out>/html/cc_shard_00000.jsonl.zst (id, final_url, http_code, content_type, encoding, body_b64) and
<out>/pairs.csv (id, url, cc_timestamp, warc_truncated, home_shard_file; the last is empty for URLs not crawled at home).

    python -I cc_fetch.py validate <out> [n]   n random URLs that the home crawl also has (default 300)
"""
import base64
import csv
import gzip
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx
import zstandard as zstd

sys.stdout.reconfigure(encoding="utf-8")
D = "D:/r2ai_data/crawl_home/"


def key(u: str) -> str:
    return urlparse(u.strip()).path.rstrip("/")


def fetch(http: httpx.Client, rec: dict) -> tuple[dict, bytes, str]:
    o, n = int(rec["offset"]), int(rec["length"])
    for attempt in range(4):
        try:
            r = http.get("https://data.commoncrawl.org/" + rec["filename"], headers={"Range": f"bytes={o}-{o + n - 1}"})
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == 3:
                raise
    warc = gzip.decompress(r.content)
    whead, rest = warc.split(b"\r\n\r\n", 1)
    hhead, body = rest.split(b"\r\n\r\n", 1)
    truncated = re.search(rb"WARC-Truncated:\s*(\S+)", whead)
    headers = {k.strip().lower(): v.strip() for k, _, v in
               (line.decode("latin-1").partition(":") for line in hhead.split(b"\r\n")[1:])}
    return headers, body, truncated.group(1).decode() if truncated else ""


def main() -> None:
    mode, out = sys.argv[1], Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    assert mode == "validate"
    recs = json.load(open(D + "cc_longchau_index.json", encoding="utf-8"))
    home = {}
    for r in csv.DictReader(open(D + "status.csv", encoding="utf-8")):
        if "longchau" in r["url"]:
            home[r["id"]] = r  # the last row per id wins
    both = sorted((r for r in home.values() if r["status"] == "ok" and key(r["url"]) in recs), key=lambda r: int(r["id"]))
    random.seed(26)
    todo = random.sample(both, n)
    (out / "html").mkdir(parents=True, exist_ok=True)
    http = httpx.Client(timeout=60)
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda r: (r, *fetch(http, recs[key(r["url"])])), todo))
    cctx = zstd.ZstdCompressor(level=3)
    with open(out / "html" / "cc_shard_00000.jsonl.zst", "wb") as f, cctx.stream_writer(f) as w, \
            open(out / "pairs.csv", "w", newline="", encoding="utf-8") as pf:
        pw = csv.writer(pf)
        pw.writerow(["id", "url", "cc_timestamp", "warc_truncated", "home_shard_file"])
        for r, headers, body, trunc in got:
            ct = headers.get("content-type", "text/html")
            m = re.search(r"charset=([\w-]+)", ct)
            rec = recs[key(r["url"])]
            w.write((json.dumps({"id": int(r["id"]), "final_url": rec["url"], "http_code": 200, "content_type": ct,
                                 "encoding": (m.group(1) if m else "utf-8").lower(),
                                 "body_b64": base64.b64encode(body).decode()}, ensure_ascii=False) + "\n").encode("utf-8"))
            pw.writerow([r["id"], r["url"], rec["timestamp"], trunc, r["shard_file"]])
    years = sorted(recs[key(r["url"])]["timestamp"][:6] for r in todo)
    print(f"{len(got)} pages, truncated {sum(1 for g in got if g[3])}, capture months {years[0]}..{years[-1]} (median {years[len(years) // 2]})")


if __name__ == "__main__":
    main()
