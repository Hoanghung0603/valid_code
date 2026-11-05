"""Bounded standalone retry for b660 crawl leftovers (518 transient + 17 missing = 535).

Standalone retry batch: does NOT restart the old crawl. Original state/pages/salvage/
verification are read-only inputs; every output goes to a NEW
data/crawl/retry_<VN date>/ directory (pages + own state log + audit).
Refuses to overwrite an existing batch directory.

Enforced constraints: cnkang.com + 120ask.com only, direct origin fetch (no proxy),
total concurrency <= 4, per-host <= 2, delay >= 1s, bandwidth <= 20 Mbps,
below-normal priority, timeout 30-45s, body cap 3MB, single extract worker,
no raw-HTML persistence, blocked host stops immediately, only ok records written.

Exit codes: 0 = done; 1 = runtime/invariant failure; 2 = validation/usage error
(count mismatch, existing output, bad args); 3 = heavy-job lock stop.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pandas as pd
import trafilatura
import zstandard as zstd

# Parity with scripts/crawl.py and scripts/extract.py is checked by tests.
MAX_ATTEMPTS = 3
TIER2_HOSTS = {
    "120ask.com", "ask.39.net", "jbk.39.net", "a-hospital.com", "familydoctor.com.cn", "cnkang.com",
    "youlai.cn", "msdmanuals.cn", "baidu.com", "ask.familydoctor.com.cn", "jb39.com",
}
HARD_HOSTS = {"zysjonline.com", "nhathuoclongchau.com.vn", "tamanhhospital.vn", "vov.vn",
              "pmc-ecm-healthblog.beta.pharmacity.io", "article.iiyi.com", "bingli.iiyi.com"}
ALLOWED_HOSTS = ("cnkang.com", "120ask.com")
SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
MIN_TEXT = 200
MAX_BODY = 3 * 1024 * 1024
BLOCK_URL_MARKERS = ("verify.html", "/captcha", "waf", "security-check")
BLOCK_TEXT_MARKERS = ("滑动拼图验证", "安全验证", "访问验证", "人机验证", "Just a moment", "Attention Required")
BLOCK_STATUSES = {403, 429, 503, 521, 522, 523, 524}
PERMANENT_STATUSES = {404, 410}
FLUSH_EVERY = 200
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
EXPECT_TRANSIENT = 518
EXPECT_MISSING = 17


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(1)
    except Exception as e:
        print(f"[retry] WARN lower_priority: {e}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


async def read_capped(resp, cap: int) -> bytes:
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(64 * 1024):
        buf += chunk
        if len(buf) >= cap:
            break
    return bytes(buf[:cap])


class BandwidthLimiter:
    def __init__(self, mbps: float):
        self.rate = mbps * 1e6 / 8 if mbps and mbps > 0 else 0.0
        self.free_at = 0.0

    def reserve(self, nbytes: int, now: float) -> float:
        if not self.rate:
            return 0.0
        start = max(now, self.free_at)
        self.free_at = start + nbytes / self.rate
        return start - now

    async def throttle(self, nbytes: int):
        delay = self.reserve(nbytes, asyncio.get_running_loop().time())
        if delay > 0:
            await asyncio.sleep(delay)


def extract_inline(body: bytes, url: str) -> dict:
    out = trafilatura.extract(
        body, url=url, output_format="json", with_metadata=True,
        include_comments=False, include_tables=True, favor_recall=True,
    )
    if not out:
        return {"title": "", "date": "", "text": ""}
    d = json.loads(out)
    return {"title": d.get("title") or "", "date": d.get("date") or "", "text": d.get("text") or ""}


def _extract_job(body: bytes, url: str) -> dict:
    # Module-level indirection: ThreadPoolExecutor pickles the target by
    # qualified name, so tests can monkeypatch retry.extract_inline.
    return extract_inline(body, url)


def classify_inline(status, final_url, text, error=None) -> str:
    if error:
        return "transient"
    if status in BLOCK_STATUSES or any(m in (final_url or "").lower() for m in BLOCK_URL_MARKERS):
        return "blocked"
    if status in PERMANENT_STATUSES:
        return "permanent"
    if status != 200:
        return "transient"
    if len(text) < MIN_TEXT and any(m in text for m in BLOCK_TEXT_MARKERS):
        return "blocked"
    return "ok" if len(text) >= MIN_TEXT else "permanent"


@dataclass
class History:
    last_outcome: str = None
    nonterminal_attempts: int = 0
    total_attempts: int = 0
    ever_ok: bool = False
    ever_permanent: bool = False
    ever_blocked: bool = False
    transient_only: bool = True
    outcomes: Counter = field(default_factory=Counter)

    @property
    def terminal_state(self):
        if self.ever_ok:
            return "ok"
        return "permanent" if self.ever_permanent else None

    def record(self, outcome):
        self.last_outcome = outcome
        self.total_attempts += 1
        self.outcomes[outcome] += 1
        self.ever_ok |= outcome == "ok"
        self.ever_permanent |= outcome == "permanent"
        if outcome not in ("ok", "permanent"):
            self.nonterminal_attempts += 1
            self.transient_only &= outcome == "transient" or outcome.startswith("transient:")
        self.ever_blocked |= outcome == "blocked" or outcome.startswith("blocked:")


def read_state(state: Path):
    histories = {}
    diag = Counter(files=0, lines=0, valid_lines=0, malformed_lines=0,
                   empty_lines=0, blank_outcomes=0, truncated_lines=0)
    for path in sorted(state.glob("log-*.tsv")):
        diag["files"] += 1
        with path.open("rb") as fh:
            for raw in fh:
                diag["lines"] += 1
                if not raw.endswith(b"\n"):
                    diag["truncated_lines"] += 1
                    continue
                try:
                    line = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
                    if not line:
                        diag["empty_lines"] += 1
                        continue
                    sid, outcome = line.split("\t")
                    sid = int(sid)
                    if not outcome.strip():
                        diag["blank_outcomes"] += 1
                        continue
                    if any(ord(c) < 32 for c in outcome):
                        raise ValueError("control character in outcome")
                except (ValueError, UnicodeError):
                    diag["malformed_lines"] += 1
                    continue
                histories.setdefault(sid, History()).record(outcome)
                diag["valid_lines"] += 1
    return histories, dict(diag)


def read_missing_summary(path: Path) -> set:
    summary = json.loads(path.read_text(encoding="utf-8"))
    count, samples = summary.get("missing"), summary.get("missing_samples")
    if type(count) is not int or count < 0 or not isinstance(samples, list):
        raise ValueError("missing summary requires missing count and complete missing_samples list")
    if any(type(i) is not int for i in samples):
        raise ValueError("missing_samples must contain integer IDs")
    if len(samples) != count or len(set(samples)) != count:
        raise ValueError("missing_samples is incomplete or duplicated; refuse partial export")
    return set(samples)


def build_candidates(corpus: Path, state: Path, summary: Path,
                     expect_transient: int = EXPECT_TRANSIENT,
                     expect_missing: int = EXPECT_MISSING):
    missing = read_missing_summary(summary)
    if len(missing) != expect_missing:
        raise ValueError(f"missing count {len(missing)} != expected {expect_missing}")
    df = pd.read_parquet(corpus, columns=["id", "url"])
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("corpus IDs must be non-null and unique")
    df["host"] = df["url"].map(host_of)
    id_row = {int(i): (str(u), str(h)) for i, u, h in df.itertuples(index=False, name=None)}
    unknown = missing - id_row.keys()
    if unknown:
        raise ValueError(f"missing_samples IDs absent from corpus: {sorted(unknown)}")
    histories, diag = read_state(state)
    transient, missing_cands = [], []
    for sid in sorted(id_row):
        url, host = id_row[sid]
        if host not in ALLOWED_HOSTS:
            continue
        h = histories.get(sid, History())
        if sid in missing:
            if h.terminal_state != "ok":
                raise ValueError(f"missing id {sid} is not state-ok (terminal={h.terminal_state})")
            missing_cands.append({"id": sid, "url": url, "host": host, "last_outcome": h.last_outcome,
                                  "nonterminal_attempts": h.nonterminal_attempts,
                                  "total_attempts": h.total_attempts,
                                  "reason": "recover_missing_shard_record",
                                  "requires_approval": True, "terminal_state": h.terminal_state})
            continue
        if h.ever_ok or h.ever_permanent or h.ever_blocked:
            continue
        if not h.total_attempts:
            continue
        if h.transient_only and h.nonterminal_attempts < MAX_ATTEMPTS:
            transient.append({"id": sid, "url": url, "host": host, "last_outcome": h.last_outcome,
                              "nonterminal_attempts": h.nonterminal_attempts,
                              "total_attempts": h.total_attempts,
                              "reason": "propose_transient_retry",
                              "requires_approval": True, "terminal_state": None})
    if len(transient) != expect_transient:
        raise ValueError(f"transient candidates {len(transient)} != expected {expect_transient}")
    if len(missing_cands) != expect_missing:
        raise ValueError(f"missing candidates {len(missing_cands)} != expected {expect_missing}")
    cands = sorted(transient + missing_cands, key=lambda c: c["id"])
    if len({c["id"] for c in cands}) != len(cands):
        raise ValueError("duplicate candidate IDs")
    return cands, {"transient": len(transient), "missing": len(missing_cands),
                   "total": len(cands), "state_diagnostics": diag}


class RetryShardWriter:
    def __init__(self, out_dir: Path, prefix: str = "retry", max_records: int = 20000, level: int = 6):
        self.out_dir, self.prefix = out_dir, prefix
        self.max_records, self.level = max_records, level
        out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = 0
        self._fh = self._zw = None
        self._n = 0
        self.files = []

    def _open(self):
        path = self.out_dir / f"{self.prefix}-{self._idx:05d}.jsonl.zst"
        self._idx += 1
        self.files.append(path)
        self._fh = path.open("xb")
        self._zw = zstd.ZstdCompressor(level=self.level).stream_writer(self._fh)
        self._n = 0

    def write(self, rec: dict):
        if self._zw is None:
            self._open()
        self._zw.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self._n += 1
        if self._n % FLUSH_EVERY == 0:
            self._zw.flush(zstd.FLUSH_FRAME)
            self._fh.flush()
        if self._n >= self.max_records:
            self.close()

    def close(self):
        if self._zw is not None:
            self._zw.flush(zstd.FLUSH_FRAME)
            self._zw.close()
            self._zw = self._fh = None


def scan_strict(path: Path, on_line):
    dctx = zstd.ZstdDecompressor()
    obj = None
    frames = 0
    out_bytes = 0
    zstd_err = None
    buf = bytearray()
    pos = 0

    def feed(out: bytes):
        nonlocal pos
        if not out:
            return
        buf.extend(out)
        while True:
            nl = buf.find(10, pos)
            if nl < 0:
                break
            on_line(bytes(buf[pos:nl]), False)
            pos = nl + 1
        if pos:
            del buf[:pos]
            pos = 0

    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                data = chunk
                while data:
                    if obj is None:
                        obj = dctx.decompressobj()
                    out = obj.decompress(data)
                    out_bytes += len(out)
                    feed(out)
                    if obj.eof:
                        frames += 1
                        data = obj.unused_data
                        obj = None
                    else:
                        data = b""
        if obj is not None:
            out = obj.flush()
            out_bytes += len(out)
            feed(out)
    except zstd.ZstdError as e:
        zstd_err = f"{type(e).__name__}: {e}"[:200]
    if zstd_err is None and obj is not None and not obj.eof:
        zstd_err = "empty_or_no_frame" if (frames == 0 and out_bytes == 0) else "truncated_final_frame"
    if zstd_err is None and frames == 0:
        zstd_err = "empty_or_no_frame"
    if buf:
        on_line(bytes(buf), True)
    return {"frames": frames, "zstd_error": zstd_err, "out_bytes": out_bytes}


@dataclass
class Opts:
    corpus: Path
    state: Path
    summary: Path
    out_root: Path
    batch: str
    concurrency: int = 4
    per_host: int = 2
    delay: float = 1.0
    max_mbps: float = 20.0
    timeout: float = 30.0
    extract_workers: int = 1
    min_free_gb: float = 5.0
    lock: Path = None
    dry_run: bool = False
    expect_transient: int = EXPECT_TRANSIENT
    expect_missing: int = EXPECT_MISSING
    session_factory: object = None


def validate_opts(o: Opts):
    if o.concurrency < 1 or o.concurrency > 4:
        raise ValueError("--concurrency must be 1..4")
    if o.per_host < 1 or o.per_host > 2:
        raise ValueError("--per-host must be 1..2")
    if o.per_host > o.concurrency:
        raise ValueError("--per-host cannot exceed --concurrency")
    if o.delay < 1.0:
        raise ValueError("--delay must be >= 1s")
    if not o.max_mbps or o.max_mbps > 20.0:
        raise ValueError("--max-mbps must be within (0, 20]")
    if o.timeout < 30.0 or o.timeout > 45.0:
        raise ValueError("--timeout must be within [30, 45]")
    if o.extract_workers != 1:
        raise ValueError("--extract-workers must be 1 (low CPU)")
    for p, kind in ((o.corpus, "file"), (o.summary, "file"), (o.state, "dir")):
        if kind == "file" and not p.is_file():
            raise ValueError(f"missing input file: {p}")
        if kind == "dir" and not p.is_dir():
            raise ValueError(f"missing input dir: {p}")


async def attempt(session, limiter, pool, item):
    status = final_url = error = None
    body = b""
    try:
        async with session.get(item["url"], allow_redirects=True, max_redirects=5) as r:
            status, final_url = r.status, str(r.url)
            body = await read_capped(r, MAX_BODY)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        error = type(e).__name__
    await limiter.throttle(len(body))
    text, meta = "", {}
    if status == 200 and body:
        meta = None
        try:
            loop = asyncio.get_running_loop()
            maybe = await loop.run_in_executor(pool, _extract_job, body, item["url"])
            meta = await maybe if asyncio.iscoroutine(maybe) else maybe
            text = meta["text"]
        except Exception as e:
            meta = {}
            error = f"extract:{type(e).__name__}"
    outcome = classify_inline(status, final_url or "", text, error)
    return {"outcome": outcome, "status": status, "final_url": final_url,
            "error": error, "text_len": len(text), "body_len": len(body),
            "title": meta.get("title", ""), "date": meta.get("date", ""), "text": text}


async def run_batch(opts: Opts):
    t0 = time.time()
    out = opts.out_root / f"retry_{opts.batch}"
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refuse existing output: {out}")
    pages, statedir, audit = out / "pages", out / "state", out / "audit"
    for d in (pages, statedir, audit):
        d.mkdir(parents=True)
    lock = opts.lock
    state_before = {p.name: sha256_of(p) for p in sorted(opts.state.glob("log-*.tsv"))}
    summary_before = sha256_of(opts.summary)

    def lock_hit():
        return lock is not None and lock.exists()

    if lock_hit():
        (audit / "run.json").write_text(json.dumps(
            {"batch": opts.batch, "exit": 3, "stop": "heavy_job_lock_at_startup",
             "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))},
            ensure_ascii=False, indent=1), encoding="utf-8")
        return 3
    if shutil.disk_usage(opts.out_root).free / 1e9 < opts.min_free_gb:
        (audit / "run.json").write_text(json.dumps(
            {"batch": opts.batch, "exit": 1, "stop": "low_disk",
             "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"batch": opts.batch, "exit": 1, "stop": "low_disk"},
                         ensure_ascii=False))
        return 1

    cands, counts = build_candidates(opts.corpus, opts.state, opts.summary,
                                     opts.expect_transient, opts.expect_missing)
    if opts.dry_run:
        print(json.dumps({"dry_run": True, "batch": opts.batch, **counts}, ensure_ascii=False))
        shutil.rmtree(out)
        return 0

    by_host = {}
    for c in cands:
        by_host.setdefault(c["host"], []).append(c)
    total_sem = asyncio.Semaphore(opts.concurrency)
    host_sem = {h: asyncio.Semaphore(min(opts.per_host, len(v))) for h, v in by_host.items()}
    limiter = BandwidthLimiter(opts.max_mbps)
    pool = ThreadPoolExecutor(max_workers=1)
    writer = RetryShardWriter(pages)
    state_fh = (statedir / "log-00000.tsv").open("w", encoding="utf-8", buffering=1)
    stop = {"flag": False, "cause": ""}
    host_stopped = {h: False for h in by_host}
    results_fh = (audit / "results.jsonl").open("w", encoding="utf-8", buffering=1)
    stat = Counter(attempted=0, ok=0, blocked=0, permanent=0, transient=0, skipped=0)
    per_host = {h: Counter() for h in by_host}
    err_types = Counter()
    bytes_in = 0
    ok_ids = set()
    last_report = t0

    def write_stats():
        (audit / "stats.json").write_text(json.dumps({
            "batch": opts.batch, "elapsed_s": int(time.time() - t0),
            "planned": len(cands), "counts": dict(stat),
            "per_host": {h: dict(v) for h, v in per_host.items()},
            "error_types": dict(err_types),
            "body_mb": round(bytes_in / 1e6, 1),
            "host_stopped": {h: v for h, v in host_stopped.items()},
            "stop": stop,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    async def worker(host, items):
        nonlocal bytes_in, last_report
        for item in items:
            if stop["flag"] or host_stopped[host]:
                stat["skipped"] += 1
                per_host[host]["skipped"] += 1
                results_fh.write(json.dumps({"id": item["id"], "url": item["url"], "host": host,
                                             "reason": item["reason"], "skipped": True,
                                             "cause": stop["cause"] or "host_blocked_stop"},
                                            ensure_ascii=False) + "\n")
                continue
            if lock_hit() or (out / "STOP").exists():
                stop["flag"] = True
                stop["cause"] = "heavy_job_lock" if lock_hit() else "stop_file"
                stat["skipped"] += 1
                per_host[host]["skipped"] += 1
                results_fh.write(json.dumps({"id": item["id"], "url": item["url"], "host": host,
                                             "reason": item["reason"], "skipped": True,
                                             "cause": stop["cause"]}, ensure_ascii=False) + "\n")
                continue
            async with total_sem:
                async with host_sem[host]:
                    if lock_hit() or (out / "STOP").exists():
                        stop["flag"] = True
                        stop["cause"] = "heavy_job_lock" if lock_hit() else "stop_file"
                        stat["skipped"] += 1
                        per_host[host]["skipped"] += 1
                        results_fh.write(json.dumps({"id": item["id"], "url": item["url"], "host": host,
                                                     "reason": item["reason"], "skipped": True,
                                                     "cause": stop["cause"]}, ensure_ascii=False) + "\n")
                        continue
                    res = await attempt(session, limiter, pool, item)
            stat["attempted"] += 1
            stat[res["outcome"]] += 1
            per_host[host][res["outcome"]] += 1
            bytes_in += res["body_len"]
            if res["error"]:
                err_types[res["error"]] += 1
            if res["outcome"] == "blocked":
                host_stopped[host] = True
            outcome_log = res["outcome"] if not res["error"] else f"{res['outcome']}:{res['error']}"
            state_fh.write(f"{item['id']}\t{outcome_log}\n")
            if res["outcome"] == "ok":
                rec = {"id": item["id"], "url": item["url"], "host": host,
                       "final_url": res["final_url"], "status": res["status"],
                       "title": res["title"], "date": res["date"],
                       "text": res["text"], "ts": int(time.time())}
                writer.write(rec)
                ok_ids.add(item["id"])
            results_fh.write(json.dumps({"id": item["id"], "url": item["url"], "host": host,
                                         "reason": item["reason"],
                                         "attempts_before": item["nonterminal_attempts"],
                                         "outcome": res["outcome"], "error": res["error"],
                                         "status": res["status"], "text_len": res["text_len"],
                                         "final_url": res["final_url"]}, ensure_ascii=False) + "\n")
            if time.time() - last_report > 30:
                last_report = time.time()
                write_stats()
            # Fixed delay with ±20% jitter (matches scripts/crawl.py
            # HostCtl.on_result: delay * uniform(0.8, 1.2)).
            await asyncio.sleep(opts.delay * random.uniform(0.8, 1.2))

    factory = opts.session_factory
    if factory is None:
        timeout = aiohttp.ClientTimeout(total=opts.timeout, sock_connect=10)
        conn = aiohttp.TCPConnector(limit=opts.concurrency,
                                    limit_per_host=opts.per_host, ssl=False, ttl_dns_cache=3600)
        headers = {"User-Agent": UA, "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}
        factory = lambda: aiohttp.ClientSession(headers=headers, timeout=timeout, connector=conn)
    async with factory() as session:
        tasks = []
        for h, rows in by_host.items():
            k = min(opts.per_host, len(rows))
            for i in range(k):
                tasks.append(asyncio.create_task(worker(h, rows[i::k])))
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            stop["flag"] = True
            stop["cause"] = "keyboard_interrupt"
        except Exception:
            stop["flag"] = True
            stop["cause"] = "worker_error"
            raise
        finally:
            writer.close()
            state_fh.close()
            results_fh.close()
            pool.shutdown(wait=False, cancel_futures=True)
    write_stats()

    manifest = audit / "manifest.sha256"
    with manifest.open("x", encoding="utf-8") as f:
        for p in sorted(pages.glob("retry-*.jsonl.zst")):
            f.write(f"{sha256_of(p)}  {p.name}\n")
    strict = {"files": len(writer.files), "records": 0, "unique": 0,
              "duplicates": 0, "schema_bad": 0, "content_bad": 0,
              "zstd_errors": [], "ids": []}
    seen = set()
    for p in sorted(pages.glob("retry-*.jsonl.zst")):
        box = {"n": 0}

        def cb(raw, trunc, b=box, name=p.name):
            b["n"] += 1
            try:
                rec = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                strict["schema_bad"] += 1
                return
            if (not isinstance(rec, dict) or [k for k in SCHEMA if k not in rec]
                    or [k for k in rec if k not in SCHEMA]):
                strict["schema_bad"] += 1
                return
            rid = rec.get("id")
            if not isinstance(rid, int) or isinstance(rid, bool):
                strict["schema_bad"] += 1
                return
            if rid in seen:
                strict["duplicates"] += 1
            else:
                seen.add(rid)
            text = rec.get("text")
            if rec.get("status") != 200 or not isinstance(text, str) or len(text) < MIN_TEXT:
                strict["content_bad"] += 1

        info = scan_strict(p, cb)
        if info["zstd_error"]:
            strict["zstd_errors"].append({p.name: info["zstd_error"]})
        strict["records"] += box["n"]
    strict["unique"] = len(seen)
    strict["ids"] = sorted(seen)
    strict_ok = (not strict["zstd_errors"] and strict["duplicates"] == 0
                 and strict["schema_bad"] == 0 and strict["content_bad"] == 0)
    cand_ids = {c["id"] for c in cands}
    missing_ids = {c["id"] for c in cands if c["reason"] == "recover_missing_shard_record"}
    off_plan = seen - cand_ids
    ever_ok_violation = sorted((seen - missing_ids) & {
        c["id"] for c in cands if c.get("terminal_state") == "ok" and c["id"] not in missing_ids})
    state_after = {p.name: sha256_of(p) for p in sorted(opts.state.glob("log-*.tsv"))}
    summary_after = sha256_of(opts.summary)
    exit_code = 0
    if stop["cause"] == "heavy_job_lock":
        exit_code = 3
    elif not strict_ok or off_plan or ever_ok_violation:
        exit_code = 1
    run_rep = {
        "batch": opts.batch, "exit": exit_code,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "duration_s": round(time.time() - t0, 1),
        "config": {"concurrency": opts.concurrency, "per_host": opts.per_host,
                   "delay": opts.delay, "max_mbps": opts.max_mbps,
                   "timeout": opts.timeout, "extract_workers": 1,
                   "body_cap": MAX_BODY, "proxy": False},
        "candidates": counts,
        "counts": dict(stat),
        "original_state_sha_before": state_before,
        "original_state_sha_after": state_after,
        "original_state_unchanged": state_before == state_after,
        "summary_sha_before": summary_before,
        "summary_sha_after": summary_after,
        "summary_unchanged": summary_before == summary_after,
        "strict_output_ok": strict_ok,
        "strict": {k: v for k, v in strict.items() if k != "ids"},
        "off_plan_ids": sorted(off_plan),
        "ever_ok_violation": ever_ok_violation,
        "stop": stop,
        "manifest": str(manifest),
        "manifest_sha256": sha256_of(manifest),
    }
    (audit / "run.json").write_text(json.dumps(run_rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"batch": opts.batch, "exit": exit_code, "counts": dict(stat),
                      "strict_ok": strict_ok, "state_unchanged": state_before == state_after},
                     ensure_ascii=False))
    return exit_code


def build_opts(argv=None) -> Opts:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path(r"D:\r2ai\data\links_corpus.parquet"))
    ap.add_argument("--state", type=Path, default=Path(r"D:\r2ai\data\crawl\state"))
    ap.add_argument("--missing-summary", type=Path,
                    default=Path(r"D:\r2ai\data\crawl\verification\verify_summary.json"))
    ap.add_argument("--out-root", type=Path, default=Path(r"D:\r2ai\data\crawl"))
    ap.add_argument("--batch", default=None,
                    help="default: machine-local date %%Y%%m%%d (b660 clock is VN)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-mbps", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--extract-workers", type=int, default=1)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--lock", type=Path, default=Path(r"D:\r2ai\locks\heavy_job.lock"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--expect-transient", type=int, default=EXPECT_TRANSIENT)
    ap.add_argument("--expect-missing", type=int, default=EXPECT_MISSING)
    a = ap.parse_args(argv)
    batch = a.batch or time.strftime("%Y%m%d", time.localtime())
    return Opts(corpus=a.corpus, state=a.state, summary=a.missing_summary,
                out_root=a.out_root, batch=batch, concurrency=a.concurrency,
                per_host=a.per_host, delay=a.delay, max_mbps=a.max_mbps,
                timeout=a.timeout, extract_workers=a.extract_workers,
                min_free_gb=a.min_free_gb, lock=a.lock, dry_run=a.dry_run,
                expect_transient=a.expect_transient, expect_missing=a.expect_missing)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        opts = build_opts(argv)
        validate_opts(opts)
    except (ValueError, OSError) as exc:
        print(f"[retry] error: {exc}", flush=True)
        return 2
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    lower_priority()
    try:
        if opts.lock is not None and opts.lock.exists():
            print("[retry] BLOCKED: heavy_job.lock exists; not running.", flush=True)
            return 3
        return asyncio.run(run_batch(opts))
    except FileExistsError as exc:
        print(f"[retry] error: {exc}", flush=True)
        return 2
    except ValueError as exc:
        print(f"[retry] error: {exc}", flush=True)
        return 2
    except KeyboardInterrupt:
        print("[retry] interrupted; progress kept in batch dir.", flush=True)
        return 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
