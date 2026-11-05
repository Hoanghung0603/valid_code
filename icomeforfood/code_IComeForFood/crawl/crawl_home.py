"""Crawl selected home-connection URLs in the main crawler's raw shard format.

The Long Chau TV review (22/10) limits this run to 5,890 odd IDs. Access from a
home IP can still fail; use --pauses with no values to stop on the first block.
This script writes:
    <out>/html/<prefix>_shard_NNNNN.jsonl.zst  one JSON line per 200 response, same fields as the main
                                           crawler: id, final_url, http_code, content_type, encoding, body_b64
    <out>/status.csv                       crawl_status columns + shard_file, to merge into status.duckdb
                                           (several rows per id are possible: the last one wins)
    <out>/crawl.log                        progress per URL and every 5 minutes, blocks, pauses

Request pacing:
- a few connections per domain sharing a token bucket at --rate requests/second with jitter; cookies kept
- compressed transfer (br/gzip): a Long Chau page is ~50KB on the wire instead of ~280KB
- 403/429/503 or a Cloudflare challenge is checked against a URL that worked before: if that one still
  works, only the URL is blocked; if not, the domain pauses 10, 30, 60, 60 minutes, then probes again,
  and runs at half rate for an hour after it recovers; after 4 failed probes in a row the domain stops
- robots.txt of both sites allows the article paths (checked 20/10)

Resumable: status rows of fetched pages are written only after their shard is closed, so a crash or a
shutdown loses at most the open shard (up to 10 minutes of pages), whose URLs are fetched again on the next run. Create <out>/STOP
to stop cleanly (the open shard is closed, nothing is lost); rerun the same command to continue.

Long Chau TV list (22/10):
    python crawl/crawl_home.py --corpus crawl/longchau_tv_crawl_odd.csv \
        --out D:/r2ai_data/crawl/crawl_home_tv \
        --rate nhathuoclongchau.com.vn=0.25 --ramp 0.05 0.025 20 \
        --workers 1 --pauses --shard-prefix tv
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import zstandard as zstd
from charset_normalizer import from_bytes

try:
    from crawl.db_file_lock import db_file_lock
except ModuleNotFoundError:  # python crawl/crawl_home.py
    from db_file_lock import db_file_lock

# same browser headers as the main crawler (notebooks/01_data_pipeline.ipynb), plus compressed transfer
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,zh-CN;q=0.8,zh;q=0.7,en-US;q=0.6,en;q=0.5",
    "Referer": "https://www.google.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Upgrade-Insecure-Requests": "1",
}
STATUS_COLS = ["id", "url", "final_url", "lang_expected", "status", "http_code", "content_type", "n_bytes",
               "attempts", "fetched_at", "shard", "sha1_raw", "error_msg", "shard_file"]
BLOCK_CODES = {403, 429, 503}
GONE_CODES = {404, 410}
CHALLENGE_MARKERS = (b'cf-chl-', b'challenge-platform', b'just a moment',
                     b'verify you are human', b'checking your browser',
                     b'sorry, you have been blocked')
RETRY_DELAYS = [5, 30]
PAUSES = [600, 1800, 3600, 3600]
SHARD_RECORDS = 2000
SHARD_SECONDS = 600  # also close a shard after 10 minutes, so a shutdown loses at most that much
try:  # brotli is optional: without it the site sends gzip, which httpx always decodes
    import brotli  # noqa: F401
    HEADERS["Accept-Encoding"] = "gzip, deflate, br"
except ImportError:
    HEADERS["Accept-Encoding"] = "gzip, deflate"
_CHARSET = re.compile(r"charset=[\"']?([\w-]+)", re.I)

log = logging.getLogger("crawl_home")


def detect_encoding(raw: bytes, content_type: str) -> str:
    """Content-Type -> <meta charset> -> charset_normalizer; gb2312/gbk -> gb18030 (as the main crawler)."""
    for src in (content_type or "", raw[:4096].decode("ascii", errors="replace")):
        m = _CHARSET.search(src)
        if m:
            enc = m.group(1).lower()
            return "gb18030" if enc in ("gb2312", "gbk", "x-gbk") else enc
    best = from_bytes(raw[:32768]).best()
    return best.encoding if best else "utf-8"


def challenge_reason(url: str, final_url: str, n_bytes: int) -> str | None:
    """Same rule as the main crawler: anti-bot redirects are blocks, not pages."""
    if "verify.html" in final_url.lower():
        return "verification_redirect"
    a, b = urlparse(url).hostname or "", urlparse(final_url).hostname or ""
    if a and b and a.lower() != b.lower() and n_bytes < 3000:
        return "small_cross_host_redirect"
    return None


class Output:
    """Streams 200 responses into zstd shards; status rows of a shard are written once it is closed."""

    def __init__(self, out: Path, prefix: str = "home"):
        self.dir = out / "html"
        self.dir.mkdir(parents=True, exist_ok=True)
        for tmp in self.dir.glob("*.tmp"):
            tmp.unlink()  # an unfinished shard of a crashed run: its URLs have no status row, so they rerun
        self.prefix = prefix  # another machine uses its own prefix, so shard names never collide
        done = [int(p.name.split("_shard_")[1].split(".")[0]) for p in self.dir.glob(f"{prefix}_shard_*.jsonl.zst")]
        self.idx = max(done, default=-1) + 1
        self.status_path = out / "status.csv"
        new = not self.status_path.exists()
        cut = not new and self.status_path.stat().st_size and not self.status_path.read_bytes().endswith(b"\n")
        self.status_f = open(self.status_path, "a", newline="", encoding="utf-8")
        if cut:
            self.status_f.write("\n")  # the last row was cut by a power loss: start the next one on a new line
        self.status = csv.DictWriter(self.status_f, fieldnames=STATUS_COLS)
        if new:
            self.status.writeheader()
        self.lock = threading.Lock()
        self.cctx = zstd.ZstdCompressor(level=3)
        self._open()

    def _open(self):
        self.name = f"{self.prefix}_shard_{self.idx:05d}.jsonl.zst"
        self.fh = open(self.dir / (self.name + ".tmp"), "wb")
        self.writer = self.cctx.stream_writer(self.fh)
        self.opened = time.monotonic()
        self.pending: list[dict] = []

    def _close(self):
        self.writer.close()  # also closes the file
        if self.pending:
            os.replace(self.dir / (self.name + ".tmp"), self.dir / self.name)
            for row in self.pending:
                self.status.writerow(row)
            self.status_f.flush()
            log.info(f"shard {self.name}: {len(self.pending)} pages")
            self.idx += 1
        else:
            (self.dir / (self.name + ".tmp")).unlink()

    def page(self, rec: dict, row: dict):
        with self.lock:
            self.writer.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
            row["shard_file"] = self.name
            self.pending.append(row)
            if len(self.pending) >= SHARD_RECORDS or time.monotonic() - self.opened >= SHARD_SECONDS:
                self._close()
                self._open()

    def row(self, row: dict):
        with self.lock:
            self.status.writerow(row)
            self.status_f.flush()

    def close(self):
        with self.lock:
            self._close()
        self.status_f.close()


class Domain:
    """Token bucket + block state shared by the workers of one domain."""

    def __init__(self, name: str, rate: float, ramp: tuple[float, float, float] | None = None):
        self.name, self.base_rate = name, rate
        self.lock = threading.Lock()
        self.next_at = time.monotonic()
        # start at half rate: a fresh client without the site's cookies got blocked within minutes at full rate
        self.half_until = 0.0 if ramp else time.monotonic() + 1800
        self.ramp = ramp  # (start rate, step, seconds per step): rise slowly towards rate while not blocked
        self.ramp_t0 = time.monotonic()
        self.paused_until = 0.0
        self.failed_probes = 0
        self.stopped = False
        self.good_url: str | None = None
        self.counts: Counter = Counter()
        self.recent: deque = deque(maxlen=200)

    def rate(self) -> float:
        now = time.monotonic()
        rate = self.base_rate
        if self.ramp:
            start, step, every = self.ramp
            rate = min(rate, start + step * int((now - self.ramp_t0) // every))
        return rate / 2 if now < self.half_until else rate

    def wait_turn(self, stop: threading.Event) -> bool:
        while not stop.is_set() and not self.stopped:
            with self.lock:
                now = time.monotonic()
                start = max(self.next_at, self.paused_until)
                if now >= start:
                    self.next_at = now + random.uniform(0.7, 1.3) / self.rate()
                    return True
            stop.wait(min(5.0, start - now))
        return False


def fetch(client: httpx.Client, url: str) -> tuple[httpx.Response | None, str | None]:
    try:
        return client.get(url, follow_redirects=True, timeout=30), None
    except httpx.HTTPError as e:
        return None, f"{type(e).__name__}: {e}"[:200]


def is_block(resp: httpx.Response) -> bool:
    if resp.status_code in BLOCK_CODES or resp.headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    if resp.status_code == 200 and "html" in resp.headers.get("content-type", "").lower():
        sample = resp.content[:200000].lower()
        return any(marker in sample for marker in CHALLENGE_MARKERS)
    return False


def worker(dom: Domain, queue: deque, client: httpx.Client, out: Output, stop: threading.Event):
    while True:
        with dom.lock:
            if not queue or dom.stopped:
                return
            item = queue.popleft()
        attempts = item.get("attempts", 0)
        while True:
            if not dom.wait_turn(stop):
                with dom.lock:
                    queue.appendleft(item)  # not fetched; rerun next time
                return
            attempts += 1
            resp, err = fetch(client, item["url"])
            row = {"id": item["id"], "url": item["url"], "final_url": item["url"], "lang_expected": "vi",
                   "attempts": attempts, "fetched_at": datetime.now(timezone.utc).isoformat(),
                   "shard": None, "sha1_raw": None, "error_msg": None, "http_code": None,
                   "content_type": None, "n_bytes": 0, "shard_file": None}
            if resp is None:
                if attempts <= len(RETRY_DELAYS):
                    stop.wait(RETRY_DELAYS[attempts - 1])
                    continue
                row.update(status="timeout" if "Timeout" in err else "dns_error", error_msg=err)
                out.row(row)
                dom.counts[row["status"]] += 1
                break
            raw = resp.content
            row.update(final_url=str(resp.url), http_code=resp.status_code,
                       content_type=resp.headers.get("content-type", ""), n_bytes=len(raw))
            if is_block(resp) or challenge_reason(item["url"], row["final_url"], len(raw)):
                if not PAUSES:
                    with dom.lock:
                        dom.stopped = True
                    row.update(status="blocked", error_msg=f"HTTP {resp.status_code}: stopped at first block",
                               sha1_raw=hashlib.sha1(raw).hexdigest())
                    out.row(row)
                    dom.counts["blocked_url"] += 1
                    log.warning("%s: first refusal/challenge; stopping this domain", dom.name)
                    break
                if handle_block(dom, client, stop):
                    attempts -= 1  # the domain was blocked, not this URL: try it again
                    continue
                row.update(status="blocked", error_msg=f"HTTP {resp.status_code}",
                           sha1_raw=hashlib.sha1(raw).hexdigest())
                out.row(row)
                dom.counts["blocked_url"] += 1
                break
            if resp.status_code == 200:
                ct = row["content_type"].lower()
                status = "ok" if ("text/html" in ct or "application/xhtml" in ct) and len(raw) >= 100 else \
                    ("non_html" if "html" not in ct else "empty")
                row.update(status=status, sha1_raw=hashlib.sha1(raw).hexdigest())
                rec = {"id": item["id"], "final_url": row["final_url"], "http_code": 200,
                       "content_type": row["content_type"], "encoding": detect_encoding(raw, row["content_type"]),
                       "body_b64": base64.b64encode(raw).decode("ascii")}
                out.page(rec, row)
                dom.good_url = item["url"]
                dom.counts[status] += 1
                dom.recent.append(1)
                break
            if resp.status_code in GONE_CODES or attempts > len(RETRY_DELAYS):
                row.update(status="http_error", error_msg=f"HTTP {resp.status_code}")
                out.row(row)
                dom.counts[f"http_{resp.status_code}"] += 1
                break
            stop.wait(RETRY_DELAYS[attempts - 1])
        completed = sum(value for key, value in dom.counts.items() if key != "pauses")
        log.info("id=%s status=%s HTTP=%s completed=%s", item["id"], row["status"],
                 row["http_code"] if row["http_code"] is not None else "-", completed)


def handle_block(dom: Domain, client: httpx.Client, stop: threading.Event) -> bool:
    """True if the domain itself is blocked (the caller retries the URL after the pause)."""
    with dom.lock:
        if time.monotonic() < dom.paused_until:
            return True  # another worker already paused the domain
        # before any page has worked in this run, the home page tells a blocked IP from a refused URL
        probe = dom.good_url or f"https://{dom.name}/"
    stop.wait(5)
    resp, _ = fetch(client, probe)
    if resp is not None and resp.status_code == 200 and not is_block(resp):
        return False  # the site still serves us: only this URL is refused
    with dom.lock:
        if time.monotonic() < dom.paused_until:
            return True  # another worker saw the same block while we probed
        if dom.failed_probes >= len(PAUSES):
            dom.stopped = True
            log.error(f"{dom.name}: still blocked after {len(PAUSES)} pauses, stopping this domain")
            return True
        pause = PAUSES[dom.failed_probes]
        dom.failed_probes += 1
        dom.paused_until = time.monotonic() + pause
        dom.half_until = dom.paused_until + 3600
        dom.counts["pauses"] += 1
    log.warning(f"{dom.name}: blocked (probe {probe} refused), pausing {pause // 60} min, then half rate")
    stop.wait(pause)
    resp, _ = fetch(client, probe)
    if resp is not None and resp.status_code == 200 and not is_block(resp):
        with dom.lock:
            dom.failed_probes = 0
            dom.ramp_t0 = time.monotonic()  # climb again from the start rate
        log.info(f"{dom.name}: recovered after pause")
    return True


def read_ids(path: str) -> set[int]:
    """IDs from a list; for crawl status CSVs, skip only successfully saved IDs."""
    with open(path, encoding="utf-8-sig") as f:
        first = f.readline().strip()
        f.seek(0)
        if "id" in first.split(","):
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or ())
            is_status = "status" in fields
            has_shard = "shard_file" in fields
            return {int(r["id"]) for r in reader
                    if (r.get("id") or "").isdigit() and
                    (not is_status or (r.get("status") == "ok" and
                     (not has_shard or bool(r.get("shard_file")))))}
        return {int(x) for x in f.read().split() if x.isdigit()}


def saved_ids_in_db(database: str) -> set[int]:
    """Read a locked DB snapshot so this queue can coexist with another writer."""
    import duckdb

    database = Path(database).resolve()
    with tempfile.TemporaryDirectory(prefix='crawl_home_db_') as folder:
        copy = Path(folder) / 'status.duckdb'
        with db_file_lock(database):
            shutil.copy2(database, copy)
            wal = Path(str(database) + '.wal')
            if wal.is_file():
                shutil.copy2(wal, Path(str(copy) + '.wal'))
        con = duckdb.connect(str(copy), read_only=True)
        try:
            return {int(row[0]) for row in con.execute(
                "SELECT id FROM crawl_status WHERE status IN ('ok','non_html') "
                'AND shard IS NOT NULL').fetchall()}
        finally:
            con.close()


def load_queue(corpus: str, domains: list[str], status_db: str | None, out: Path,
               skip_files: list[str], part: tuple[int, int] | None, part_domains: list[str]) -> dict[str, deque]:
    if corpus.endswith(".csv"):  # a CSV id,url list needs no pyarrow on a machine with a very new Python
        with open(corpus, encoding="utf-8") as f:
            t = [{"id": int(r["id"]), "url": r["url"]} for r in csv.DictReader(f)]
    else:
        import pyarrow.parquet as pq
        t = pq.read_table(corpus, columns=["id", "url"]).to_pylist()
    host = lambda u: (urlparse(u).hostname or "").lower().removeprefix("www.")  # noqa: E731
    skip: set[int] = set()
    for f in skip_files:
        skip |= read_ids(f)  # pages another machine already has
    if status_db:
        skip |= saved_ids_in_db(status_db)
    status_path = out / "status.csv"
    if status_path.exists():
        last: dict[int, dict] = {}
        with open(status_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("id") or "").isdigit() and r.get("status"):  # a row cut by a power loss is skipped
                    last[int(r["id"])] = r
        # finished: a page or a gone page; refusals and timeouts are tried again on the next run
        skip |= {i for i, r in last.items() if r["status"] in ("ok", "non_html", "empty", "http_error")}
    queues = {d: deque() for d in domains}
    for r in t:
        h = host(r["url"])
        mine = part is None or h not in part_domains or r["id"] % part[1] == part[0]
        if h in queues and r["id"] not in skip and mine:
            queues[h].append({"id": r["id"], "url": r["url"]})
    return queues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--status-db", help="main status.duckdb: IDs already saved are skipped using a locked snapshot")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rate", nargs="+", default=["nhathuoclongchau.com.vn=1.5", "tamanhhospital.vn=1.0"],
                    help="domain=requests per second")
    ap.add_argument("--workers", type=int, default=2, help="connections per domain")
    ap.add_argument("--limit", type=int, default=0, help="URLs per domain, for a trial run")
    ap.add_argument("--skip-ids", nargs="*", default=[], help="CSV (id column) or id-per-line files to skip")
    ap.add_argument("--part", help="K/N: only ids with id %% N == K, to split the work between machines")
    ap.add_argument("--pauses", nargs="*", type=int, default=[10, 30, 60, 60],
                    help="minutes to pause after each block; give none to stop the domain at the first refusal/challenge")
    ap.add_argument("--ramp", nargs=3, type=float, metavar=("START", "STEP", "MINUTES"),
                    help="start at START req/s and add STEP every MINUTES while not blocked, up to --rate")
    ap.add_argument("--shard-prefix", default="home", help="shard file prefix, one per machine")
    ap.add_argument("--part-domains", nargs="+", default=["nhathuoclongchau.com.vn"], help="domains --part applies to")
    ap.add_argument("--plan", action="store_true", help="show the remaining queue without requests or output changes")
    args = ap.parse_args()
    part = tuple(int(x) for x in args.part.split("/")) if args.part else None
    PAUSES[:] = [m * 60 for m in args.pauses]

    rates = {d: float(r) for d, r in (x.split("=") for x in args.rate)}
    if args.workers < 1 or any(rate <= 0 for rate in rates.values()):
        ap.error("--workers and every --rate must be positive")
    queues = load_queue(args.corpus, list(rates), args.status_db, args.out, args.skip_ids, part, args.part_domains)
    if args.limit:
        queues = {d: deque(list(q)[:args.limit]) for d, q in queues.items()}
    if args.plan:
        for domain, queue in queues.items():
            print(f'{domain}: {len(queue):,} URLs remaining')
        print('Offline plan: zero requests; no output files changed.')
        return

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "STOP").unlink(missing_ok=True)  # left over from the previous clean stop
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.out / "crawl.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is too much for 80K URLs
    if os.name == "nt":  # an overnight run must not stop when Windows sleeps on idle; the display may turn off
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log.info("queue: " + ", ".join(f"{d} {len(q)}" for d, q in queues.items()))

    out = Output(args.out, args.shard_prefix)
    stop = threading.Event()
    ramp = (args.ramp[0], args.ramp[1], args.ramp[2] * 60) if args.ramp else None
    domains = {d: Domain(d, rates[d], ramp) for d in queues}
    threads, clients = [], []
    for d, q in queues.items():
        for _ in range(args.workers):
            c = httpx.Client(headers=HEADERS, http2=False, limits=httpx.Limits(max_connections=1))
            clients.append(c)
            t = threading.Thread(target=worker, args=(domains[d], q, c, out, stop), daemon=True)
            t.start()
            threads.append(t)

    t0, last = time.time(), 0.0
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            if (args.out / "STOP").exists():
                log.info("STOP file found, finishing")
                stop.set()
            if time.time() - last >= 300:
                last = time.time()
                for d, dom in domains.items():
                    done = sum(v for k, v in dom.counts.items() if k != "pauses")
                    speed = done / max(1.0, time.time() - t0)
                    eta = len(queues[d]) / speed / 3600 if speed else float("inf")
                    log.info(f"{d}: done {done}, left {len(queues[d])}, {dict(dom.counts)}, "
                             f"{speed:.2f}/s, rate {dom.rate():.2f}, eta {eta:.1f}h"
                             + (" STOPPED" if dom.stopped else ""))
    except KeyboardInterrupt:
        stop.set()
    for t in threads:
        t.join()
    out.close()
    for c in clients:
        c.close()
    for d, dom in domains.items():
        log.info(f"finished {d}: {dict(dom.counts)}, left {len(queues[d])}" + (" STOPPED" if dom.stopped else ""))


if __name__ == "__main__":
    main()
