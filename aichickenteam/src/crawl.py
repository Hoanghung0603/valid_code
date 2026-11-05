"""Async, polite, resumable crawler for data/ViBioMIR/links_corpus.parquet.

Output: Doc records (docs/INTERFACES.md) in sharded zstd parquet `data/docs/corpus-XXXXX.parquet`.
State:  sqlite `data/cache/crawl.sqlite` table `crawl` (id, status, http_code, error, n_chars, ...).

Examples
  python -m src.crawl --sample-per-domain 5                 # test crawl, 5 URLs / host
  python -m src.crawl --lang vi                             # all Vietnamese URLs
  python -m src.crawl --domains 120ask.com,39.net --limit 10000
  python -m src.crawl --shard 0/4 --priority vi-first        # id % 4 == 0 subset (uniform sample)
  python -m src.crawl --ids-file my_ids.txt                  # explicit ids, in file order
  python -m src.crawl --report data/cache/crawl_stats.md     # only write stats/estimate tables from the state db

Parallel processes: split by --lang / --domains (politeness budgets are per process, so two --shard processes
on the same domains would double the per-domain rate). They can share the state db and the output dir.
"""
from __future__ import annotations

import argparse
import asyncio
import heapq
import itertools
import logging
import multiprocessing
import os
import random
import re
import signal
import sqlite3
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("crawl")

DEFAULTS = {
    "corpus": "data/ViBioMIR/links_corpus.parquet",
    "out_dir": "data/docs",
    "state_db": "data/cache/crawl.sqlite",
    "global_concurrency": 64,
    "per_domain_concurrency": 4,
    "per_domain_delay": 0.25,       # min seconds between request starts on one politeness key
    "jitter": 0.3,                  # +- fraction of delay
    "max_delay": 8.0,               # adaptive slow-down ceiling
    "politeness_key": "registered",  # "host" | "registered" (all *.39.net share one budget)
    "timeout": 25.0,
    "connect_timeout": 10.0,
    "retries": 3,
    "backoff_base": 1.5,
    "max_bytes": 4_000_000,
    "flush_every": 5000,
    "min_chars": 30,                # docs with less extracted text -> status "empty", not stored
    "extract_workers": 6,
    "backend": "curl_cffi",         # curl_cffi (Chrome TLS fingerprint) | httpx
    "impersonate": "chrome",
    "http2": False,                 # httpx backend only
    "dead_after": 25,               # consecutive failures with 0 successes -> give up host this run
    "max_runs": 3,                  # transient failures are retried on restart up to this many runs
    "user_agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/128.0.0.0 Safari/537.36"),
    "domain_overrides": {},          # {"120ask.com": {"delay": 0.1, "concurrency": 8, "headers": {}, "cookies": {}}}
    "split_hosts": [],               # registered domains whose sub-hosts get separate politeness keys
}

_JS_COOKIE = re.compile(rb"""document\.cookie\s*=\s*["']([^=;"']+)=([^;"']*)""")

# statuses that are final (never re-fetched unless --force)
FINAL = ("ok", "empty", "not_html", "redirect_home", "gone")
DOC_COLS = ("doc_id", "source", "lang", "url", "title", "text", "fetched_at")


# ============================================================================ config


def load_config(path: str | None = None) -> dict:
    cfg = dict(DEFAULTS)
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    if p.exists():
        import yaml
        with open(p) as f:
            y = yaml.safe_load(f) or {}
        cfg.update((y.get("crawl") or {}))
    return cfg


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


_MULTI_SUFFIX = {"com.cn", "com.vn", "org.vn", "gov.vn", "net.vn", "edu.vn", "org.cn", "net.cn", "gov.cn"}


def registered_domain(host: str) -> str:
    parts = host.lower().split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _MULTI_SUFFIX:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


# ============================================================================ state db


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=60, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS crawl(
            id INTEGER PRIMARY KEY, url TEXT, host TEXT, status TEXT, http_code INTEGER,
            error TEXT, n_chars INTEGER, title TEXT, final_url TEXT, encoding TEXT, method TEXT,
            lang TEXT, bytes INTEGER, elapsed REAL, runs INTEGER DEFAULT 1, fetched_at TEXT)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS crawl_host ON crawl(host)")

    def skip_ids(self, max_runs: int, retry_failed: bool) -> set[int]:
        """ids not to fetch again: final statuses + transient failures that used up their runs."""
        q = f"SELECT id FROM crawl WHERE status IN ({','.join('?' * len(FINAL))})"
        args: list = list(FINAL)
        if not retry_failed:
            q += " OR runs >= ?"
            args.append(max_runs)
        return {r[0] for r in self.db.execute(q, args)}

    def put_many(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.db.execute("BEGIN")
        self.db.executemany(
            """INSERT INTO crawl(id,url,host,status,http_code,error,n_chars,title,final_url,encoding,
                   method,lang,bytes,elapsed,runs,fetched_at)
               VALUES(:id,:url,:host,:status,:http_code,:error,:n_chars,:title,:final_url,:encoding,
                   :method,:lang,:bytes,:elapsed,1,:fetched_at)
               ON CONFLICT(id) DO UPDATE SET url=excluded.url, host=excluded.host,
                   status=excluded.status, http_code=excluded.http_code, error=excluded.error,
                   n_chars=excluded.n_chars, title=excluded.title, final_url=excluded.final_url,
                   encoding=excluded.encoding, method=excluded.method, lang=excluded.lang,
                   bytes=excluded.bytes, elapsed=excluded.elapsed, runs=crawl.runs+1,
                   fetched_at=excluded.fetched_at""",
            [{k: r.get(k) for k in ("id", "url", "host", "status", "http_code", "error", "n_chars",
                                    "title", "final_url", "encoding", "method", "lang", "bytes",
                                    "elapsed", "fetched_at")} for r in rows])
        self.db.execute("COMMIT")


# ============================================================================ output


class ShardWriter:
    """Buffers Doc dicts; each flush writes one new zstd parquet file corpus-XXXXX.parquet.
    File numbers are claimed atomically (os.link) so several processes can write concurrently."""

    def __init__(self, out_dir: Path, flush_every: int, on_flush=None):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.buf: list[dict] = []
        self.pending_state: list[dict] = []
        self.on_flush = on_flush
        self.files: list[Path] = []
        self.n_docs = 0

    def add(self, doc: dict | None, state_row: dict) -> None:
        if doc is not None:
            self.buf.append(doc)
        self.pending_state.append(state_row)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self.buf:
            import pyarrow as pa
            import pyarrow.parquet as pq
            tbl = pa.table({c: [d[c] for d in self.buf] for c in DOC_COLS},
                           schema=pa.schema([(c, pa.string()) for c in DOC_COLS]))
            tmp = self.out_dir / f".tmp-{os.getpid()}-{time.time_ns()}.parquet"
            pq.write_table(tbl, tmp, compression="zstd", compression_level=6)
            existing = [int(p.stem.split("-")[1]) for p in self.out_dir.glob("corpus-*.parquet")
                        if p.stem.split("-")[1].isdigit()]
            i = max(existing, default=-1) + 1
            while True:
                dst = self.out_dir / f"corpus-{i:05d}.parquet"
                try:
                    os.link(tmp, dst)
                    break
                except FileExistsError:
                    i += 1
            tmp.unlink()
            self.files.append(dst)
            self.n_docs += len(self.buf)
            log.info("flushed %d docs -> %s", len(self.buf), dst.name)
            self.buf = []
        # state is written only after the docs are durably on disk -> crash-safe resume
        if self.on_flush and self.pending_state:
            self.on_flush(self.pending_state)
        self.pending_state = []


def load_docs(paths=None, dedupe: bool = True, columns=None, batch_size: int = 2048) -> Iterator[dict]:
    """Iterate Doc dicts from corpus parquet shards (default data/docs/corpus-*.parquet), in shard order.
    `paths`: None | dir | file | list of files. With dedupe, a doc_id present in several shards (re-crawl
    with --force) is yielded once, from the newest shard."""
    import pyarrow.parquet as pq
    if paths is None:
        paths = sorted(_abs(DEFAULTS["out_dir"]).glob("corpus-*.parquet"))
    elif isinstance(paths, (str, Path)):
        p = Path(paths)
        paths = sorted(p.glob("corpus-*.parquet")) if p.is_dir() else [p]
    paths = [Path(p) for p in paths]
    owner: dict[str, int] = {}
    if dedupe and len(paths) > 1:
        for i, path in enumerate(paths):
            for did in pq.read_table(path, columns=["doc_id"]).column(0).to_pylist():
                owner[did] = i
    if columns is not None and dedupe and "doc_id" not in columns:
        columns = ["doc_id", *columns]
    seen: set[str] = set()
    for i, path in enumerate(paths):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            for d in batch.to_pylist():
                if dedupe:
                    did = d.get("doc_id")
                    if (owner and owner.get(did) != i) or did in seen:
                        continue
                    seen.add(did)
                yield d


# ============================================================================ target selection


def select_targets(cfg: dict, ids_file=None, domains=None, lang=None, sample_per_domain=None,
                   shard=None, limit=None, seed=0, priority="vi-first", skip: set[int] | None = None,
                   order="interleave"):
    """-> list[(id, url, host)] in priority order."""
    import polars as pl
    from .extractors.text import lang_from_host
    df = pl.read_parquet(_abs(cfg["corpus"]))
    df = df.with_columns(pl.col("url").str.extract(r"^https?://([^/:?#]+)").str.to_lowercase().alias("host"))
    if ids_file:
        p = Path(ids_file)
        if p.suffix == ".parquet":
            ids = pl.read_parquet(p)["id"].to_list()
        else:
            ids = [int(x.split(",")[0]) for x in p.read_text().split() if x.split(",")[0].strip().isdigit()]
        rank = pl.DataFrame({"id": ids, "_rank": list(range(len(ids)))}).unique("id", keep="first")
        df = df.join(rank, on="id", how="inner").sort("_rank")
    if domains:
        ds = [d.strip().lower() for d in domains.split(",") if d.strip()]
        cond = pl.lit(False)
        for d in ds:
            cond = cond | (pl.col("host") == d) | pl.col("host").str.ends_with("." + d)
        df = df.filter(cond)
    hosts = df["host"].unique().to_list()
    hl = {h: (lang_from_host(h) or "zh") for h in hosts}
    df = df.with_columns(pl.col("host").replace_strict(hl, default="zh").alias("lang"))
    if lang:
        df = df.filter(pl.col("lang").is_in(lang.split(",")))
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        df = df.filter((pl.col("id") % n) == i)
    if skip:
        df = df.filter(~pl.col("id").is_in(list(skip)))
    if sample_per_domain:
        df = (df.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("host").alias("_r"))
                .filter(pl.col("_r") < sample_per_domain).drop("_r"))
        if not ids_file:
            df = df.sort("id")
    elif not ids_file:
        df = df.sort("id")
    rows = list(zip(df["id"].to_list(), df["url"].to_list(), df["host"].to_list(), df["lang"].to_list()))
    if priority == "vi-first":
        rows.sort(key=lambda r: 0 if r[3] == "vi" else 1)          # stable
    elif priority == "zh-first":
        rows.sort(key=lambda r: 0 if r[3] == "zh" else 1)
    if limit:
        if order == "interleave":
            # take round-robin across hosts so a small --limit still covers many domains
            by_host: dict[str, deque] = defaultdict(deque)
            for r in rows:
                by_host[r[2]].append(r)
            out = []
            qs = list(by_host.values())
            while len(out) < limit and qs:
                nq = []
                for q in qs:
                    if q and len(out) < limit:
                        out.append(q.popleft())
                    if q:
                        nq.append(q)
                qs = nq
            rows = out
        else:
            rows = rows[:limit]
    return [(r[0], r[1], r[2]) for r in rows]


# ============================================================================ crawler


class PriorityGate:
    """Global concurrency limiter where lower priority numbers are served first."""

    def __init__(self, n: int):
        self.free = n
        self.waiters: list = []
        self.counter = itertools.count()

    async def acquire(self, prio: int) -> None:
        if self.free > 0 and not self.waiters:
            self.free -= 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self.waiters, (prio, next(self.counter), fut))
        await fut

    def release(self) -> None:
        while self.waiters:
            _, _, fut = heapq.heappop(self.waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self.free += 1


@dataclass
class Domain:
    key: str
    prio: int
    delay: float
    concurrency: int
    queue: deque = field(default_factory=deque)
    next_t: float = 0.0
    ok: int = 0
    fail_streak: int = 0
    dead: bool = False
    stats: Counter = field(default_factory=Counter)
    headers: dict = field(default_factory=dict)   # per-domain extra headers (e.g. UA matching a cookie)
    cookies: dict = field(default_factory=dict)   # per-domain cookies (e.g. a manually obtained cf_clearance)

    async def wait_turn(self, jitter: float) -> None:
        now = time.monotonic()
        t = max(now, self.next_t)
        self.next_t = t + self.delay * (1 + random.uniform(-jitter, jitter))
        if t > now:
            await asyncio.sleep(t - now)


def _extract_job(body: bytes, url: str, charset: str | None, min_chars: int) -> dict:
    logging.getLogger("trafilatura").setLevel(logging.CRITICAL)
    from .extractors import extract
    return extract(body, url, charset, min_chars=max(80, min_chars))


def _soft_home(orig: str, final: str) -> bool:
    a, b = urlsplit(orig), urlsplit(final)
    return (b.path in ("", "/", "/index.html", "/index.htm", "/index.php") and not b.query
            and a.path not in ("", "/") and a.geturl() != b.geturl())


class Crawler:
    def __init__(self, cfg: dict, targets, state: State, writer: ShardWriter, dump_html: Path | None = None):
        self.cfg = cfg
        self.state = state
        self.writer = writer
        self.dump_html = dump_html
        self.gate = PriorityGate(int(cfg["global_concurrency"]))
        self.domains: dict[str, Domain] = {}
        overrides = cfg.get("domain_overrides") or {}
        for prio, (id_, url, host) in enumerate(targets):
            reg = registered_domain(host)
            split = reg in (cfg.get("split_hosts") or ())
            key = reg if cfg["politeness_key"] == "registered" and not split else host
            d = self.domains.get(key)
            if d is None:
                ov = overrides.get(key) or overrides.get(host) or overrides.get(reg) or {}
                d = Domain(key=key, prio=prio, delay=float(ov.get("delay", cfg["per_domain_delay"])),
                           concurrency=int(ov.get("concurrency", cfg["per_domain_concurrency"])),
                           headers=dict(ov.get("headers") or {}), cookies=dict(ov.get("cookies") or {}))
                self.domains[key] = d
            d.queue.append((id_, url, host))
        self.total = len(targets)
        self.done = 0
        self.t0 = time.monotonic()
        self.status = Counter()
        self.bytes = 0
        self.stop = False
        # forkserver: curl/httpx threads exist in the parent; plain fork() could deadlock children
        self.pool = ProcessPoolExecutor(max_workers=int(cfg["extract_workers"]),
                                        mp_context=multiprocessing.get_context("forkserver"))
        self.host_stats: dict[str, Counter] = defaultdict(Counter)

    # ------------------------------------------------------------------ HTTP backends
    # curl_cffi impersonates Chrome's TLS/HTTP2 fingerprint: several sites (nhathuoclongchau, vov,
    # tamanhhospital, ... behind Cloudflare/PWS) answer 403 to Python's OpenSSL fingerprint.
    def _headers(self) -> dict:
        return {
            "User-Agent": self.cfg["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,zh-CN;q=0.8,zh;q=0.7,en;q=0.6",
        }

    def _client(self):
        c = self.cfg
        if c.get("backend", "curl_cffi") == "curl_cffi":
            from curl_cffi.requests import AsyncSession
            # NB: do not override User-Agent/Accept here: a UA that disagrees with the impersonated
            # TLS fingerprint gets 403 on vov.vn & co. Only add Accept-Language.
            return AsyncSession(max_clients=int(c["global_concurrency"]), impersonate=c.get("impersonate", "chrome"),
                                headers={"Accept-Language": self._headers()["Accept-Language"]},
                                timeout=(float(c["connect_timeout"]), float(c["timeout"])))
        import httpx
        limits = httpx.Limits(max_connections=int(c["global_concurrency"]) * 2,
                              max_keepalive_connections=int(c["global_concurrency"]))
        return httpx.AsyncClient(headers=self._headers(), limits=limits, http2=bool(c["http2"]),
                                 timeout=httpx.Timeout(float(c["timeout"]), connect=float(c["connect_timeout"])),
                                 follow_redirects=True, max_redirects=8, verify=False)

    async def _get(self, url: str, insecure: bool, h1: bool = False, dom: Domain | None = None
                   ) -> tuple[int, dict, bytes, str]:
        """-> (status, headers(lowercased), body, final_url). Raises on transport errors."""
        extra = {}
        if dom is not None and dom.headers:
            extra["headers"] = dom.headers
        if dom is not None and dom.cookies:
            extra["cookies"] = dom.cookies
        if self.cfg.get("backend", "curl_cffi") == "curl_cffi":
            if h1:
                extra["http_version"] = "v1"
            r = await self.client.get(url, allow_redirects=True, max_redirects=8, verify=not insecure, **extra)
            return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content, str(r.url)
        r = await self.client.get(url, **extra)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content, str(r.url)

    def _set_cookie(self, name: str, value: str, host: str) -> None:
        try:
            self.client.cookies.set(name, value, domain=host)
        except Exception:  # pragma: no cover
            pass

    async def fetch(self, url: str, dom: Domain) -> dict:
        c = self.cfg
        retries = int(c["retries"])
        last_err, code, insecure, h1 = None, None, False, False
        cookie_retry = 0
        attempt = -1
        while attempt < retries:
            attempt += 1
            if attempt:
                await asyncio.sleep(c["backoff_base"] * (2 ** (attempt - 1)) * (1 + random.random()))
                await dom.wait_turn(c["jitter"])
            await self.gate.acquire(dom.prio)
            t0 = time.monotonic()
            try:
                code, headers, body, final = await self._get(url, insecure, h1, dom)
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:160]}"
                last_err = msg
                if any(k in msg for k in ("SSL", "certificate", "Certificate", "TLS")) and not insecure:
                    insecure = True
                if "curl: (16)" in msg or "curl: (92)" in msg or "HTTP/2" in msg:
                    h1 = True  # broken HTTP/2 framing on some servers -> retry over HTTP/1.1
                if "redirect" in msg.lower() and "maximum" in msg.lower():
                    return {"status": "net_error", "error": "TooManyRedirects", "http_code": code}
                if "Could not resolve host" in msg or "DNSError" in msg:
                    retries = min(retries, attempt + 1)
                continue
            finally:
                self.gate.release()
            elapsed = time.monotonic() - t0
            ctype = headers.get("content-type", "")
            if code in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) and attempt < retries:
                last_err = f"HTTP {code}"
                if code in (429, 503):
                    dom.delay = min(c["max_delay"], max(dom.delay * 2, 0.5))
                continue
            if code == 403 and headers.get("cf-mitigated") == "challenge":
                return {"status": "blocked", "http_code": code, "error": "cloudflare challenge", "final_url": final}
            if code >= 400:
                st = "gone" if code in (404, 410) else ("blocked" if code in (403, 429) else "http_error")
                return {"status": st, "http_code": code, "error": f"HTTP {code}", "final_url": final}
            if ctype and not any(t in ctype for t in ("html", "xml", "text/plain")):
                return {"status": "not_html", "http_code": code, "error": ctype[:80], "final_url": final}
            if len(body) < 3000 and cookie_retry < 2:
                # tiny JS cookie challenge (laodong.vn & co):  document.cookie="K=V"; location.reload()
                mc = _JS_COOKIE.search(body)
                if mc:
                    cookie_retry += 1
                    self._set_cookie(mc.group(1).decode(), mc.group(2).decode(), urlsplit(final).hostname or "")
                    attempt -= 1
                    continue
            m = re.search(r"charset=([\w-]+)", ctype, re.I)
            return {"status": "fetched", "http_code": code, "body": body[: int(c["max_bytes"])],
                    "charset": m.group(1) if m else None, "final_url": final, "elapsed": elapsed}
        status = "blocked" if code in (403, 429) else ("http_error" if code and code >= 400 else "net_error")
        return {"status": status, "http_code": code, "error": last_err}

    async def handle(self, id_: int, url: str, host: str, dom: Domain) -> None:
        res = await self.fetch(url, dom)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {"id": id_, "url": url, "host": host, "fetched_at": now, "http_code": res.get("http_code"),
               "error": res.get("error"), "final_url": res.get("final_url"), "elapsed": res.get("elapsed"),
               "n_chars": 0}
        doc = None
        if res["status"] == "fetched":
            body = res["body"]
            row["bytes"] = len(body)
            self.bytes += len(body)
            if self.dump_html is not None:
                self.dump_html.mkdir(parents=True, exist_ok=True)
                (self.dump_html / f"{id_}.html").write_bytes(body)
            if _soft_home(url, res["final_url"]):
                row["status"] = "redirect_home"
            else:
                loop = asyncio.get_running_loop()
                try:
                    ex = await loop.run_in_executor(self.pool, _extract_job, body, url, res.get("charset"),
                                                    int(self.cfg["min_chars"]))
                except Exception as e:
                    ex = None
                    row.update(status="extract_error", error=f"{type(e).__name__}: {str(e)[:120]}")
                if ex is not None:
                    row.update(n_chars=len(ex["text"]), title=ex["title"][:300], encoding=ex["encoding"],
                               method=ex["method"], lang=ex["lang"])
                    if len(ex["text"]) >= int(self.cfg["min_chars"]):
                        row["status"] = "ok"
                        doc = {"doc_id": str(id_), "source": "corpus", "lang": ex["lang"], "url": url,
                               "title": ex["title"], "text": ex["text"], "fetched_at": now}
                    else:
                        row["status"] = "empty"
        else:
            row["status"] = res["status"]
        # domain health
        if row["status"] in ("ok", "empty", "redirect_home", "not_html", "gone"):
            dom.ok += 1
            dom.fail_streak = 0
        else:
            dom.fail_streak += 1
            if row["status"] == "blocked":
                dom.delay = min(self.cfg["max_delay"], max(dom.delay * 2, 0.5))
            if dom.ok == 0 and dom.fail_streak >= int(self.cfg["dead_after"]):
                dom.dead = True
                log.warning("giving up on %s for this run (%d consecutive failures, last: %s)",
                            dom.key, dom.fail_streak, row.get("error"))
        self.status[row["status"]] += 1
        self.host_stats[host][row["status"]] += 1
        self.done += 1
        self.writer.add(doc, row)

    async def domain_worker(self, dom: Domain) -> None:
        while dom.queue and not dom.dead and not self.stop:
            id_, url, host = dom.queue.popleft()
            await dom.wait_turn(self.cfg["jitter"])
            try:
                await self.handle(id_, url, host, dom)
            except Exception as e:  # pragma: no cover
                log.exception("worker error on %s: %s", url, e)

    async def progress(self, every: float = 15.0) -> None:
        while True:
            await asyncio.sleep(every)
            dt = time.monotonic() - self.t0
            log.info("%d/%d done (%.1f/s, %.1f MB in) %s", self.done, self.total, self.done / dt,
                     self.bytes / 1e6, dict(self.status.most_common()))

    async def run(self) -> dict:
        self.client = self._client()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_signal)
            except (NotImplementedError, RuntimeError):
                pass
        prog = asyncio.create_task(self.progress())
        tasks = [asyncio.create_task(self.domain_worker(d))
                 for d in sorted(self.domains.values(), key=lambda d: d.prio)
                 for _ in range(d.concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            prog.cancel()
            self.writer.flush()
            close = getattr(self.client, "aclose", None) or getattr(self.client, "close")
            res = close()
            if asyncio.iscoroutine(res):
                await res
            self.pool.shutdown(wait=True, cancel_futures=True)
        dt = time.monotonic() - self.t0
        return {"done": self.done, "seconds": dt, "rate": self.done / max(dt, 1e-9),
                "status": dict(self.status), "bytes_in": self.bytes, "docs_written": self.writer.n_docs,
                "files": [str(p) for p in self.writer.files],
                "dead_domains": [d.key for d in self.domains.values() if d.dead]}

    def _on_signal(self) -> None:
        if self.stop:
            raise KeyboardInterrupt
        log.warning("stop requested: finishing in-flight requests and flushing (Ctrl-C again to abort)")
        self.stop = True


# ============================================================================ report


def crawl_stats(state: State, since: str | None = None, ids: set[int] | None = None) -> dict:
    """Per-host statistics from the state db (optionally only rows fetched at/after `since`)."""
    q = "SELECT id, host, status, http_code, error, n_chars, method, elapsed, bytes, fetched_at FROM crawl"
    args: list = []
    if since:
        q += " WHERE fetched_at >= ?"
        args.append(since)
    rows = [r for r in state.db.execute(q, args) if ids is None or r[0] in ids]
    by = defaultdict(list)
    for r in rows:
        by[r[1]].append(r)
    out = {}
    for host, rs in by.items():
        ok = [r for r in rs if r[2] == "ok"]
        lat = sorted(r[7] for r in rs if r[7] is not None)
        errs = Counter((r[4] or "")[:70] for r in rs if r[2] != "ok" and r[4])
        out[host] = {
            "tried": len(rs), "ok": len(ok), "status": Counter(r[2] for r in rs),
            "avg_chars": (sum(r[5] or 0 for r in ok) / len(ok)) if ok else 0.0,
            "med_latency": lat[len(lat) // 2] if lat else None,
            "mean_latency": (sum(lat) / len(lat)) if lat else None,
            "avg_bytes": (sum(r[8] or 0 for r in rs if r[8]) / max(1, sum(1 for r in rs if r[8]))),
            "methods": Counter((r[6] or "-") for r in ok), "top_error": errs.most_common(1)[0][0] if errs else "",
        }
    return out


def host_sizes(cfg: dict) -> dict[str, int]:
    import polars as pl
    sizes = (pl.read_parquet(_abs(cfg["corpus"]), columns=["url"])
             .select(pl.col("url").str.extract(r"^https?://([^/:?#]+)").str.to_lowercase().alias("host"))
             .group_by("host").len())
    return dict(zip(sizes["host"].to_list(), sizes["len"].to_list()))


def estimate_full_crawl(stats: dict, sizes: dict, bytes_per_doc: float, concurrency: int = 8,
                        delay: float = 0.1, global_rps: float = 60.0, overrides: dict | None = None) -> dict:
    """Wall-clock / disk estimate for crawling every URL with the given per-key settings.
    rate(key) = min(conc / mean_latency, 1 / delay); wall = max(max_key N_key / rate_key, N / global_rps)."""
    overrides = overrides or {}
    keys: dict[str, dict] = defaultdict(lambda: {"n": 0, "lat": [], "ok": 0.0, "hosts": []})
    for host, n in sizes.items():
        k = registered_domain(host)
        st = stats.get(host)
        succ = (st["ok"] / st["tried"]) if st and st["tried"] else 0.0
        keys[k]["n"] += n
        keys[k]["ok"] += n * succ
        keys[k]["hosts"].append(host)
        if st and st.get("mean_latency"):
            keys[k]["lat"].append(st["mean_latency"])
    rows, total_ok = [], 0.0
    worst = 0.0
    for k, v in keys.items():
        ov = overrides.get(k, {})
        c, d = ov.get("concurrency", concurrency), ov.get("delay", delay)
        lat = sorted(v["lat"])[len(v["lat"]) // 2] if v["lat"] else 2.0
        rate = min(c / max(lat, 0.05), 1.0 / max(d, 1e-3))
        hours = v["n"] / rate / 3600
        worst = max(worst, hours)
        total_ok += v["ok"]
        rows.append((k, v["n"], round(v["ok"]), c, d, lat, rate, hours))
    n_all = sum(sizes.values())
    wall = max(worst, n_all / global_rps / 3600)
    return {"rows": sorted(rows, key=lambda r: -r[1]), "wall_hours": wall, "bottleneck_hours": worst,
            "global_hours": n_all / global_rps / 3600, "expected_ok": total_ok,
            "disk_gb": total_ok * bytes_per_doc / 1e9, "n_urls": n_all}


def report_tables(stats: dict, sizes: dict) -> str:
    lines = ["| host | corpus URLs | tried | ok | ok % | avg chars | med. latency (s) | failures | extractor |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for host in sorted(stats, key=lambda h: -sizes.get(h, 0)):
        st = stats[host]
        fails = ", ".join(f"{k}:{v}" for k, v in st["status"].most_common() if k != "ok")
        if st["top_error"] and fails:
            fails += f" ({st['top_error']})"
        meth = ", ".join(f"{k}:{v}" for k, v in st["methods"].most_common(2))
        lat = f"{st['med_latency']:.2f}" if st["med_latency"] is not None else "-"
        lines.append(f"| {host} | {sizes.get(host, 0):,} | {st['tried']} | {st['ok']} | "
                     f"{100 * st['ok'] / max(1, st['tried']):.0f} | {st['avg_chars']:,.0f} | {lat} | "
                     f"{fails or '-'} | {meth or '-'} |")
    return "\n".join(lines)


def build_report(cfg: dict, state: State, since: str | None = None) -> str:
    import pyarrow.parquet as pq
    stats = crawl_stats(state, since=since)
    sizes = host_sizes(cfg)
    q = "SELECT MIN(fetched_at), MAX(fetched_at), COUNT(*), SUM(status='ok'), SUM(bytes) FROM crawl"
    t_min, t_max, n, n_ok, nbytes = state.db.execute(q + (" WHERE fetched_at >= ?" if since else ""),
                                                      [since] if since else []).fetchone()
    dur = (datetime.fromisoformat(t_max) - datetime.fromisoformat(t_min)).total_seconds() if t_min else 0
    files = sorted(_abs(cfg["out_dir"]).glob("corpus-*.parquet"))
    n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    disk = sum(f.stat().st_size for f in files)
    bpd = disk / max(1, n_rows)
    tot_chars = sum(st["avg_chars"] * st["ok"] for st in stats.values())
    tried_hosts = len(stats)
    full_ok = sum(1 for st in stats.values() if st["ok"] == st["tried"])
    dead = [h for h, st in stats.items() if st["ok"] == 0]
    md = [f"## Run summary (auto-generated by `python -m src.crawl --report`)", "",
          f"- rows: {n} URLs over {tried_hosts} hosts, ok: {n_ok} ({100 * (n_ok or 0) / max(1, n):.1f}%)",
          f"- wall time: {dur:.0f}s -> {n / max(dur, 1):.2f} URLs/s overall (politeness-bound: few URLs per host)",
          f"- downloaded: {(nbytes or 0) / 1e6:.1f} MB HTML ({(nbytes or 0) / max(1, n):,.0f} B/page, not stored)",
          f"- docs in parquet: {n_rows} docs, {disk / 1e6:.2f} MB on disk -> {bpd:,.0f} B/doc (zstd); "
          f"avg text {tot_chars / max(1, n_ok or 1):,.0f} chars/doc",
          f"- hosts with 100% success: {full_ok}/{tried_hosts}; hosts with 0 ok: {len(dead)} ({', '.join(sorted(dead))})",
          "", "## Per-host results", "", report_tables(stats, sizes), ""]
    for conc, delay, grps in ((4, 0.25, 40), (8, 0.1, 80), (16, 0.05, 150)):
        est = estimate_full_crawl(stats, sizes, bpd, concurrency=conc, delay=delay, global_rps=grps)
        md += [f"### Full-crawl estimate: per-domain concurrency {conc}, delay {delay}s, global cap {grps} req/s", "",
               f"- {est['n_urls']:,} URLs, expected ok ≈ {est['expected_ok']:,.0f} docs, disk ≈ {est['disk_gb']:.1f} GB",
               f"- slowest politeness key: {est['bottleneck_hours']:.1f} h; global cap: {est['global_hours']:.1f} h "
               f"-> **wall ≈ {est['wall_hours']:.1f} h**", "",
               "| key | URLs | expected ok | conc | delay | mean lat (s) | req/s | hours |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        md += [f"| {k} | {n_:,} | {ok:,} | {c} | {d} | {lat:.2f} | {r:.1f} | {h:.1f} |"
               for k, n_, ok, c, d, lat, r, h in est["rows"][:15]]
        md.append("")
    return "\n".join(md)


# ============================================================================ CLI


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ids-file", help="txt (one id per line) / csv (id first col) / parquet with 'id'; keeps order")
    ap.add_argument("--domains", help="comma list; matches host or any subdomain (e.g. 39.net)")
    ap.add_argument("--lang", help="vi | zh | vi,zh (by host TLD)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-per-domain", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", help="i/n: only ids with id %% n == i")
    ap.add_argument("--priority", default="vi-first", choices=["vi-first", "zh-first", "none"])
    ap.add_argument("--concurrency", type=int, help="global concurrency")
    ap.add_argument("--per-domain", type=int, help="per politeness-key concurrency")
    ap.add_argument("--delay", type=float, help="per politeness-key min delay between requests (s)")
    ap.add_argument("--flush-every", type=int)
    ap.add_argument("--out-dir")
    ap.add_argument("--state-db")
    ap.add_argument("--force", action="store_true", help="re-fetch ids even if done")
    ap.add_argument("--retry-failed", action="store_true", help="retry transient failures beyond max_runs")
    ap.add_argument("--dump-html", help="debug: save raw HTML to this dir (small runs only!)")
    ap.add_argument("--dry-run", action="store_true", help="only print target selection stats")
    ap.add_argument("--report", help="write a markdown stats/estimate report from the state db and exit")
    ap.add_argument("--since", help="with --report: only rows fetched at/after this ISO time")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(a.config)
    for k, v in (("global_concurrency", a.concurrency), ("per_domain_concurrency", a.per_domain),
                 ("per_domain_delay", a.delay), ("flush_every", a.flush_every), ("out_dir", a.out_dir),
                 ("state_db", a.state_db)):
        if v is not None:
            cfg[k] = v
    state = State(_abs(cfg["state_db"]))
    if a.report:
        md = build_report(cfg, state, since=a.since)
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(md)
        print(md)
        return
    skip = set() if a.force else state.skip_ids(int(cfg["max_runs"]), a.retry_failed)
    targets = select_targets(cfg, ids_file=a.ids_file, domains=a.domains, lang=a.lang,
                             sample_per_domain=a.sample_per_domain, shard=a.shard, limit=a.limit,
                             seed=a.seed, priority=a.priority, skip=skip)
    hosts = Counter(t[2] for t in targets)
    log.info("%d targets over %d hosts (skipped %d already done)", len(targets), len(hosts), len(skip))
    if a.dry_run:
        for h, n in hosts.most_common():
            print(f"{n:>9,}  {h}")
        return
    if not targets:
        return
    writer = ShardWriter(_abs(cfg["out_dir"]), int(cfg["flush_every"]), on_flush=state.put_many)
    crawler = Crawler(cfg, targets, state, writer, dump_html=Path(a.dump_html) if a.dump_html else None)
    info = asyncio.run(crawler.run())
    log.info("finished: %s", {k: v for k, v in info.items() if k != "files"})
    print(info)


if __name__ == "__main__":
    main()
