"""Long Chau pages in Common Crawl, read from the columnar index (the CDX server keeps answering 504).

Adds to D:/r2ai_data/crawl_home/cc_longchau_index.json (same records as cc_lc.py: url, timestamp, filename,
offset, length), keeping the newest status-200 capture per URL path; skips snapshots in cc_done.txt.
"""
import gzip
import json
import re
import sys
import time
from urllib.parse import urlparse

import duckdb
import httpx

sys.stdout.reconfigure(encoding="utf-8")
D = "D:/r2ai_data/crawl_home/"
BASE = "https://data.commoncrawl.org/"


def key(u: str) -> str:
    return urlparse(u.strip()).path.rstrip("/")


http = httpx.Client(timeout=120)
cols = [c["id"] for c in http.get("https://index.commoncrawl.org/collinfo.json").json()][:24]
recs = json.load(open(D + "cc_longchau_index.json", encoding="utf-8"))
done = set(open(D + "cc_done.txt").read().split())
con = duckdb.connect()
con.sql("SET memory_limit='1GB'; SET threads=8; SET enable_progress_bar=false")
con.sql("INSTALL httpfs; LOAD httpfs")

limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(cols)  # snapshots per run, so one run fits a tool timeout
for c in [c for c in cols if c not in done][:limit]:
    t = time.time()
    paths = gzip.decompress(http.get(f"{BASE}crawl-data/{c}/cc-index-table.paths.gz").content).decode().split()
    files = [BASE + p for p in paths if "/subset=warc/" in p]
    for attempt in range(4):
        try:
            rows = con.sql(f"""select url, fetch_time::varchar, warc_filename, warc_record_offset, warc_record_length
                               from read_parquet({files})
                               where url_surtkey >= 'vn,com,nhathuoclongchau)/' and url_surtkey < 'vn,com,nhathuoclongchau)0'
                                 and fetch_status = 200""").fetchall()
            break
        except Exception as e:
            print(c, "retry", attempt, str(e)[:120], flush=True)
            time.sleep(60 * (attempt + 1))
    else:
        print(c, "FAILED", flush=True)
        continue
    n_new = 0
    for url, ft, fn, off, ln in rows:
        k, ts = key(url), re.sub(r"\D", "", ft)[:14]  # CDX timestamp form, so old and new records compare
        if k not in recs or ts > recs[k]["timestamp"]:
            n_new += k not in recs
            recs[k] = {"url": url, "timestamp": ts, "filename": fn, "offset": str(off), "length": str(ln)}
    json.dump(recs, open(D + "cc_longchau_index.json", "w", encoding="utf-8"))
    with open(D + "cc_done.txt", "a") as f:
        f.write(c + "\n")
    print(f"{c}: {len(rows)} records, {n_new} new paths, total {len(recs)}  {time.time() - t:.0f}s", flush=True)
print("ALL_DONE")
