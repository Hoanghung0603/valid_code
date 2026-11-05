"""Fetch the Common Crawl captures (cc_failed/cc_hits.json) of corpus ids still missing in coverage_1026/missing_sub3_1028.csv.

Output in D:/r2ai_data/cc_failed: html/cc_shard_00000.jsonl.zst (id, final_url, http_code, content_type, encoding,
body_b64, cc_timestamp, cc_snapshot) and cc_fetch_status.csv. Two threads with a pause per request; the first
refusal from data.commoncrawl.org (403 or any non-206 after retries) stops the whole run.
"""
import base64
import csv
import gzip
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import httpx
import zstandard as zstd

D = Path("D:/r2ai_data")
OUT = D / "cc_failed"
PAUSE = 0.5


def key(u: str) -> str:
    p = urlparse(u.strip())
    return p.netloc.lower().removeprefix("www.") + p.path.rstrip("/") + (("?" + p.query) if p.query else "")


class Refused(Exception):
    pass


def fetch(http: httpx.Client, rec: dict) -> tuple[dict, bytes]:
    o, n = int(rec["offset"]), int(rec["length"])
    for attempt in range(3):
        time.sleep(PAUSE)
        try:
            r = http.get("https://data.commoncrawl.org/" + rec["filename"], headers={"Range": f"bytes={o}-{o + n - 1}"})
        except httpx.HTTPError:
            time.sleep(30)
            continue
        if r.status_code == 206:
            break
        raise Refused(f"HTTP {r.status_code} for {rec['filename']}")
    else:
        raise Refused(f"no answer for {rec['filename']}")
    _, rest = gzip.decompress(r.content).split(b"\r\n\r\n", 1)
    hhead, body = rest.split(b"\r\n\r\n", 1)
    headers = {k.strip().lower(): v.strip() for k, _, v in
               (line.decode("latin-1").partition(":") for line in hhead.split(b"\r\n")[1:])}
    return headers, body


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    hits = json.loads((OUT / "cc_hits.json").read_text(encoding="utf-8"))
    rows = duckdb.sql(f"""select m.id, c.url from read_csv('{D}/coverage_1026/missing_sub3_1028.csv') m
                          join '{D}/raw/links_corpus_vi_en_zh.parquet' c using (id) order by m.id""").fetchall()
    todo = [(i, u, hits[key(u)]) for i, u in rows if key(u) in hits]
    print(f"{len(todo)} captures to fetch", flush=True)
    (OUT / "html").mkdir(exist_ok=True)
    http = httpx.Client(timeout=60)
    shard = "cc_shard_00001.jsonl.zst"
    n_ok = 0
    with open(OUT / "html" / (shard + ".tmp"), "wb") as f, zstd.ZstdCompressor(level=3).stream_writer(f) as w, \
            open(OUT / "cc_fetch_status_1029.csv", "w", newline="", encoding="utf-8") as sf, ThreadPoolExecutor(2) as ex:
        sw = csv.writer(sf)
        sw.writerow(["id", "url", "cc_timestamp", "cc_snapshot", "status", "shard_file"])
        try:
            for (i, u, rec), (headers, body) in zip(todo, ex.map(lambda t: fetch(http, t[2]), todo)):
                ct = headers.get("content-type", "text/html")
                m = re.search(r"charset=([\w-]+)", ct)
                w.write((json.dumps({"id": i, "final_url": rec["url"], "http_code": 200, "content_type": ct,
                                     "encoding": (m.group(1) if m else "utf-8").lower(),
                                     "body_b64": base64.b64encode(body).decode(), "cc_timestamp": rec["timestamp"],
                                     "cc_snapshot": rec["snapshot"]}, ensure_ascii=False) + "\n").encode("utf-8"))
                sw.writerow([i, u, rec["timestamp"], rec["snapshot"], "ok", shard])
                n_ok += 1
                if n_ok % 200 == 0:
                    print(f"done {n_ok}/{len(todo)}", flush=True)
        except Refused as e:
            print(f"STOPPED: CC refused this IP after {n_ok} records: {e}", flush=True)
            ex.shutdown(cancel_futures=True)
    (OUT / "html" / (shard + ".tmp")).replace(OUT / "html" / shard)
    print(f"ALL_DONE {n_ok}/{len(todo)} ok", flush=True)


if __name__ == "__main__":
    main()
