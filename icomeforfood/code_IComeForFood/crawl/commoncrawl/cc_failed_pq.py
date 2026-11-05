"""Which failed / empty non-Long Chau URLs of the crawl DB are archived in Common Crawl? (index only)

Reads the CC columnar index (parquet over HTTPS, row groups pruned by url_surtkey) instead of the
overloaded CDX server. Newest snapshots first; for each URL the newest status-200 capture is kept.

    python -I cc_failed_pq.py <status.duckdb> <out_dir> [n_snapshots] [sql_where]

Default selection: failed / empty non-Long Chau URLs. Pass a WHERE clause to check another set (e.g. 120ask pending).
"""
import gzip
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import httpx

sys.stdout.reconfigure(encoding="utf-8")
db, out = sys.argv[1], Path(sys.argv[2])
n_snap = int(sys.argv[3]) if len(sys.argv) > 3 else 24
out.mkdir(parents=True, exist_ok=True)
BASE = "https://data.commoncrawl.org/"


def key(u: str) -> str:
    p = urlparse(u.strip())
    return p.netloc.lower().removeprefix("www.") + p.path.rstrip("/") + (("?" + p.query) if p.query else "")


def surt_prefix(host: str) -> str:
    return ",".join(reversed(host.lower().removeprefix("www.").split("."))) + ")/"


WHERE = sys.argv[4] if len(sys.argv) > 4 else (
    "status not in ('ok', 'pending') and url not like '%nhathuoclongchau.com.vn%'")
rows = duckdb.connect(db, read_only=True).sql(f"select id, url, status from crawl_status where {WHERE}").fetchall()
want = {key(u): (i, u, s) for i, u, s in rows}
hosts = sorted({urlparse(u).netloc.lower() for _, u, _ in rows})
ranges = " or ".join(f"(url_surtkey >= '{p}' and url_surtkey < '{p[:-1]}0')" for p in sorted({surt_prefix(h) for h in hosts}))
print(f"{len(want)} failed URLs on {len(hosts)} hosts", flush=True)

http = httpx.Client(timeout=120)
cols = [c["id"] for c in http.get("https://index.commoncrawl.org/collinfo.json").json()][:n_snap]
hits_f, done_f = out / "cc_hits.json", out / "cc_done.txt"
hits = json.loads(hits_f.read_text(encoding="utf-8")) if hits_f.exists() else {}
done = set(done_f.read_text().split()) if done_f.exists() else set()
con = duckdb.connect()
con.sql("SET memory_limit='2GB'; SET threads=4; SET enable_progress_bar=false")
con.sql("INSTALL httpfs; LOAD httpfs")

for c in cols:
    if c in done:
        continue
    time.sleep(30)  # one snapshot at a time with a pause: CC 403'd this IP twice
    t = time.time()
    paths = gzip.decompress(http.get(f"{BASE}crawl-data/{c}/cc-index-table.paths.gz").content).decode().split()
    files = [BASE + p for p in paths if "/subset=warc/" in p]
    for attempt in range(4):
        try:
            recs = con.sql(f"""select url, fetch_time::varchar, warc_filename, warc_record_offset, warc_record_length,
                                      content_mime_detected from read_parquet({files})
                               where ({ranges}) and fetch_status = 200""").fetchall()
            break
        except Exception as e:
            print(c, "retry", attempt, str(e)[:120], flush=True)
            time.sleep(60 * (attempt + 1))
    else:
        print(c, "FAILED, stopping: CC may be refusing this IP", flush=True)
        sys.exit(1)
    n_new = 0
    for url, ts, fn, off, ln, mime in recs:
        k = key(url)
        if k in want and (k not in hits or ts > hits[k]["timestamp"]):
            n_new += k not in hits
            hits[k] = {"url": url, "timestamp": ts, "filename": fn, "offset": off, "length": ln, "mime": mime,
                       "snapshot": c}
    hits_f.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    with open(done_f, "a") as f:
        f.write(c + "\n")
    print(f"{c}: {len(recs)} host records, {n_new} new hits, total {len(hits)}/{len(want)}  {time.time() - t:.0f}s",
          flush=True)

print("== failed URLs with a status-200 CC capture (newest kept)")
by_host = {}
for k, (i, u, s) in want.items():
    by_host.setdefault((urlparse(u).netloc.lower(), s), []).append(k)
for (h, s), keys in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
    ts = sorted(hits[k]["timestamp"][:10] for k in keys if k in hits)
    print(f"{h:42s} {s:11s} {len(ts):6d}/{len(keys):<6d} newest capture {ts[0] if ts else '-'} .. {ts[-1] if ts else '-'}")
print("ALL_DONE")
