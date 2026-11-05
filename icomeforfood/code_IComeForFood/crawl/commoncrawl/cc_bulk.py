"""Bulk-fetch Long Chau pages from Common Crawl into crawl_home-format shards (no request to Long Chau).

For every URL of the NT (even) and TV (odd) ranked lists that the home crawl does not have as `ok`, take the
newest status-200 capture in cc_longchau_index.json. Captures before 2026-04 use the old Next.js layout, whose
article lives only in __NEXT_DATA__; those are rebuilt into a small HTML page (main chat's validated rule).

Output in <out>: html/cc_shard_NNNNN.jsonl.zst (id, final_url, http_code, content_type, encoding, body_b64,
cc_timestamp, cc_rebuilt) and cc_status.csv (one row per id once its shard is closed). Resumable: ids already in
cc_status.csv are skipped; an unfinished .tmp shard is deleted and refetched. Create <out>/STOP to stop cleanly.

    python -I cc_bulk.py <out>
"""
import base64
import csv
import gzip
import html
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import httpx
import zstandard as zstd

D = "D:/r2ai_data/crawl_home/"
SHARD_RECORDS = 1000
NEW_LAYOUT = "202604"  # captures from this month on carry the article in the HTML itself
NEXT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
COLS = ["id", "url", "list", "cc_timestamp", "cc_rebuilt", "status", "error_msg", "shard_file"]
log = logging.getLogger("cc_bulk")


def key(u: str) -> str:
    return urlparse(u.strip()).path.rstrip("/")


def from_next_data(body: bytes) -> bytes | None:
    """Old Long Chau layout (CC captures before ~2026-04): the article is only in __NEXT_DATA__ JSON."""
    m = NEXT.search(body.decode("utf-8", "replace"))
    d = (json.loads(m.group(1)) if m else {}).get("props", {}).get("pageProps", {}).get("detail", {}).get("data", {})
    if len(d.get("description") or "") < 1000:
        return None
    name, short = html.escape(d.get("name") or ""), html.escape(d.get("shortDescription") or "")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{name}</title></head><body><main><article>"
            f"<h1>{name}</h1><p>{short}</p>{d['description']}</article></main></body></html>").encode("utf-8")


def fetch(http: httpx.Client, rec: dict) -> tuple[dict, bytes]:
    o, n = int(rec["offset"]), int(rec["length"])
    for attempt in range(5):
        try:
            r = http.get("https://data.commoncrawl.org/" + rec["filename"], headers={"Range": f"bytes={o}-{o + n - 1}"})
            if r.status_code == 206:
                break
            err = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            err = type(e).__name__
        time.sleep(5 * 2 ** attempt)  # CC asks for backoff on 503 / slow down
    else:
        raise RuntimeError(err)
    warc = gzip.decompress(r.content)
    _, rest = warc.split(b"\r\n\r\n", 1)
    hhead, body = rest.split(b"\r\n\r\n", 1)
    headers = {k.strip().lower(): v.strip() for k, _, v in
               (line.decode("latin-1").partition(":") for line in hhead.split(b"\r\n")[1:])}
    return headers, body


def main() -> None:
    out = Path(sys.argv[1])
    (out / "html").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=out / "cc_bulk.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request would flood the log
    (out / "STOP").unlink(missing_ok=True)
    for tmp in (out / "html").glob("*.tmp"):
        tmp.unlink()

    recs = json.load(open(D + "cc_longchau_index.json", encoding="utf-8"))
    home_ok = {r["id"] for r in csv.DictReader(open(D + "status.csv", encoding="utf-8")) if r["status"] == "ok"}
    status_path = out / "cc_status.csv"
    done = {r["id"] for r in csv.DictReader(open(status_path, encoding="utf-8"))} if status_path.exists() else set()
    todo = []
    for name in ("nt", "tv"):  # NT's even ids first, each list in its query-ranked order
        for r in csv.DictReader(open(f"{D}longchau_ranked_{name}.csv", encoding="utf-8")):
            if r["id"] not in home_ok and r["id"] not in done and key(r["url"]) in recs:
                todo.append((r, name, recs[key(r["url"])]))
    log.info(f"todo {len(todo)} (skipped {len(done)} done, home ok {len(home_ok)})")

    new = not status_path.exists() or status_path.stat().st_size == 0
    sf = open(status_path, "a", newline="", encoding="utf-8")
    sw = csv.DictWriter(sf, fieldnames=COLS)
    if new:
        sw.writeheader()
    idx = max([int(p.name.split("_shard_")[1].split(".")[0]) for p in (out / "html").glob("cc_shard_*.jsonl.zst")],
              default=-1) + 1
    cctx = zstd.ZstdCompressor(level=3)
    http = httpx.Client(timeout=60)
    t0, n_ok, n_err, n_rebuilt = time.time(), 0, 0, 0

    def work(item):
        r, name, rec = item
        try:
            headers, body = fetch(http, rec)
        except Exception as e:
            return item, None, None, str(e)[:200]
        rebuilt = None
        if rec["timestamp"][:6] < NEW_LAYOUT:
            nb = from_next_data(body)
            if nb is not None:
                body, rebuilt = nb, "next_data"
        return item, headers, (body, rebuilt), ""

    with ThreadPoolExecutor(6) as ex:
        for start in range(0, len(todo), SHARD_RECORDS):
            if (out / "STOP").exists():
                log.info("STOP file found, finishing")
                break
            batch = todo[start:start + SHARD_RECORDS]
            shard = f"cc_shard_{idx:05d}.jsonl.zst"
            rows = []
            with open(out / "html" / (shard + ".tmp"), "wb") as f, cctx.stream_writer(f) as w:
                for (r, name, rec), headers, got, err in ex.map(work, batch):
                    row = {"id": r["id"], "url": r["url"], "list": name, "cc_timestamp": rec["timestamp"],
                           "cc_rebuilt": "", "status": "error" if err else "ok", "error_msg": err, "shard_file": ""}
                    if not err:
                        body, rebuilt = got
                        ct = "text/html; charset=utf-8" if rebuilt else headers.get("content-type", "text/html")
                        m = re.search(r"charset=([\w-]+)", ct)
                        w.write((json.dumps({"id": int(r["id"]), "final_url": rec["url"], "http_code": 200,
                                             "content_type": ct, "encoding": (m.group(1) if m else "utf-8").lower(),
                                             "body_b64": base64.b64encode(body).decode(), "cc_timestamp": rec["timestamp"],
                                             "cc_rebuilt": rebuilt}, ensure_ascii=False) + "\n").encode("utf-8"))
                        row.update(cc_rebuilt=rebuilt or "", shard_file=shard)
                        n_ok += 1
                        n_rebuilt += bool(rebuilt)
                    else:
                        n_err += 1
                    rows.append(row)
            (out / "html" / (shard + ".tmp")).replace(out / "html" / shard)
            sw.writerows(rows)
            sf.flush()
            idx += 1
            done_n = start + len(batch)
            rate = done_n / (time.time() - t0)
            log.info(f"{shard}: {len(batch)} | done {done_n}/{len(todo)} ok {n_ok} err {n_err} rebuilt {n_rebuilt} "
                     f"| {rate:.1f}/s eta {(len(todo) - done_n) / rate / 60:.0f} min")
    sf.close()
    log.info(f"finished: ok {n_ok} err {n_err} rebuilt {n_rebuilt}")


if __name__ == "__main__":
    main()
