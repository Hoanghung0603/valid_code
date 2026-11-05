"""Common Crawl copies of every Long Chau page that the home crawl has as `ok` and whose newest capture is from 2026.

Same output as cc_fetch.py validate (raw WARC bodies, no rebuild), split into shards of 1,000:
<out>/html/cc_shard_NNNNN.jsonl.zst and <out>/pairs.csv (id, url, cc_timestamp, warc_truncated, home_shard_file).

    python -I cc_validate_all.py <out>
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

import httpx
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
from cc_fetch import D, key  # noqa: E402

SHARD_RECORDS = 1000
PAUSE = 0.5  # seconds per request per thread: CloudFront 403'd the home IP after a day of 6-thread fetching


def fetch(http: httpx.Client, rec: dict) -> tuple[dict, bytes, str]:
    o, n = int(rec["offset"]), int(rec["length"])
    for attempt in range(6):
        time.sleep(PAUSE)
        try:
            r = http.get("https://data.commoncrawl.org/" + rec["filename"], headers={"Range": f"bytes={o}-{o + n - 1}"})
            if r.status_code == 206:
                break
            err = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:  # dropped connection / partial body: back off like a 403
            err = type(e).__name__
        time.sleep(30 * 2 ** attempt)  # 403/503 = slow down; give up after ~30 min of refusals
    else:
        raise RuntimeError(f"{err} for {rec['filename']}")
    warc = gzip.decompress(r.content)
    whead, rest = warc.split(b"\r\n\r\n", 1)
    hhead, body = rest.split(b"\r\n\r\n", 1)
    truncated = re.search(rb"WARC-Truncated:\s*(\S+)", whead)
    headers = {k.strip().lower(): v.strip() for k, _, v in
               (line.decode("latin-1").partition(":") for line in hhead.split(b"\r\n")[1:])}
    return headers, body, truncated.group(1).decode() if truncated else ""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(sys.argv[1])
    (out / "html").mkdir(parents=True, exist_ok=True)
    recs = json.load(open(D + "cc_longchau_index.json", encoding="utf-8"))
    home = {}
    for r in csv.DictReader(open(D + "status.csv", encoding="utf-8")):
        if "longchau" in r["url"]:
            home[r["id"]] = r  # the last row per id wins
    todo = sorted((r for r in home.values() if r["status"] == "ok" and key(r["url"]) in recs
                   and recs[key(r["url"])]["timestamp"] >= "20260101"), key=lambda r: int(r["id"]))
    print(f"{len(todo)} pages", flush=True)
    http = httpx.Client(timeout=60)
    cctx = zstd.ZstdCompressor(level=3)
    n_trunc = 0
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as pf, ThreadPoolExecutor(2) as ex:
        pw = csv.writer(pf)
        pw.writerow(["id", "url", "cc_timestamp", "warc_truncated", "home_shard_file"])
        for i in range(0, len(todo), SHARD_RECORDS):
            batch = todo[i:i + SHARD_RECORDS]
            got = ex.map(lambda r: (r, *fetch(http, recs[key(r["url"])])), batch)
            name = f"cc_shard_{i // SHARD_RECORDS:05d}.jsonl.zst"
            with open(out / "html" / name, "wb") as f, cctx.stream_writer(f) as w:
                for r, headers, body, trunc in got:
                    rec = recs[key(r["url"])]
                    ct = headers.get("content-type", "text/html")
                    m = re.search(r"charset=([\w-]+)", ct)
                    w.write((json.dumps({"id": int(r["id"]), "final_url": rec["url"], "http_code": 200, "content_type": ct,
                                         "encoding": (m.group(1) if m else "utf-8").lower(),
                                         "body_b64": base64.b64encode(body).decode(), "cc_timestamp": rec["timestamp"]},
                                        ensure_ascii=False) + "\n").encode("utf-8"))
                    pw.writerow([r["id"], r["url"], rec["timestamp"], trunc, r["shard_file"]])
                    n_trunc += bool(trunc)
            pf.flush()
            print(f"{name}: done {i + len(batch)}/{len(todo)}", flush=True)
    print(f"ALL_DONE {len(todo)} pages, truncated {n_trunc}")


if __name__ == "__main__":
    main()
