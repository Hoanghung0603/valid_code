"""VI top-up batch for b660: finish ALL Vietnamese docs still not ok TODAY.

Read-only target selection (never modifies queue/ or done_ids.tsv):
  * VI rows (sup.is_vi_host) whose done-kind is not terminal-ok;
  * EXCLUDE ids whose latest known HTTP status is 404/410 (dead URLs —
    "loi hoan toan"; retrying them is meaningless);
  * everything else stays IN (retry / permanent-short-text / infra /
    blocked / never-attempted): a fresh-browser + rotating-proxy attempt
    is a different attempt class than the pooled attempts that failed.

Fetch per row: sup.run_browser_one with pool=None -> FRESH headless browser,
ALWAYS through a rotated Webshare proxy (RouteTable sticky + cooldown),
CDP bandwidth cap on the shared browser lane, stale-tab guards, fail-closed
status. One retry ONLY for infra/no-status errors with a fresh browser and a
new route; never retries 404/410, blocked/challenge or classified outcomes.
Extraction: crawl_browser_probe.parse_main_html + trafilatura txt fallback
(pilot parity). No captcha solving, no raw-HTML persistence, no threshold
lowering (MIN_TEXT 200 stays).

Output (NEW dir, refuses non-empty prior run):
  data\\crawl\\vi_topup_20261026\\pages\\vi-topup-*.jsonl.zst  (schema-9, 20k)
  data\\crawl\\vi_topup_20261026\\audit\\{targets.jsonl,results.jsonl,run.json}
  manifest.sha256 over the shards + sup.strict_scan_shard verification.

Yields to the recovery STOP file and heavy_job.lock: pauses until both clear.

Exit codes: 0 done; 1 finished with failures or aborted; 2 usage/validation;
3 stop/lock at startup.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crawl_async_supervisor as sup  # noqa: E402

BASE = Path(r"D:\r2ai\data\crawl\async_recovery_20261009")
DONE = BASE / "state" / "done_ids.tsv"
QUEUE = BASE / "queue"
AUDIT = BASE / "audit"
STOP_FILE = BASE / "STOP"
LOCK = Path(r"D:\r2ai\locks\heavy_job.lock")
PROXY_FILE = HERE / "proxy_secrets.txt"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

TERMINAL = set(sup.TERMINAL_KINDS)
DEAD_STATUSES = (404, 410)
SCHEMA = sup.SCHEMA


def stop_hit() -> bool:
    try:
        return STOP_FILE.exists()
    except OSError:
        return False


def lock_hit() -> bool:
    try:
        return LOCK.exists()
    except OSError:
        return False


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception as e:
        print(f"[topup] WARN lower_priority: {e}", flush=True)


def load_done_kinds() -> dict:
    kinds = {}
    with DONE.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                kinds[parts[0].strip()] = parts[1].strip()
    return kinds


def load_last_status() -> dict:
    """Latest non-null HTTP status per id from the recovery audit stream."""
    last = {}
    p = AUDIT / "results.jsonl"
    if not p.exists():
        return last
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = str(r.get("id", "")).strip()
            st = r.get("status")
            if sid and isinstance(st, int) and not isinstance(st, bool):
                last[sid] = st
    return last


def select_targets() -> tuple[list, dict]:
    kinds = load_done_kinds()
    last = load_last_status()
    targets = []
    seen = set()
    skipped_terminal = skipped_dead = 0
    per_kind = Counter()
    for path in sorted(QUEUE.glob("*.jsonl.zst")):
        for rec in sup.iter_jsonl_zst(path):
            host = str(rec.get("host") or "").lower()
            if not sup.is_vi_host(host):
                continue
            sid = str(rec.get("id", "")).strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            kind = kinds.get(sid, "")
            per_kind[kind or "none"] += 1
            if kind in TERMINAL:
                skipped_terminal += 1
                continue
            if last.get(sid) in DEAD_STATUSES:
                skipped_dead += 1
                continue
            targets.append({"id": int(sid), "url": rec.get("url") or "",
                            "host": host, "prior_kind": kind,
                            "prior_status": last.get(sid)})
    targets.sort(key=lambda t: (t["host"], t["id"]))
    stats = {"queue_vi_unique": len(seen), "targets": len(targets),
             "skipped_terminal": skipped_terminal,
             "skipped_dead_404_410": skipped_dead,
             "prior_kinds": dict(per_kind.most_common())}
    return targets, stats


def fallback_text(html: bytes, url: str) -> dict:
    try:
        import trafilatura
        txt = trafilatura.extract(html, url=url, output_format="txt",
                                  with_metadata=False, include_comments=False,
                                  include_tables=True, favor_recall=True)
        return {"title": "", "date": "", "text": txt or ""}
    except Exception as e:
        return {"title": "", "date": "", "text": "",
                "fallback_error": f"{type(e).__name__}"}


def extract_html(raw: bytes, final_url: str):
    """parse_main_html + txt fallback when text is short (pilot parity)."""
    import crawl_browser_probe as probe
    method = "json"
    fb_err = None
    title = date = ""
    try:
        ex = probe.parse_main_html(raw, final_url or "")
        title = str(ex.get("title", "") or "")
        date = str(ex.get("date", "") or "")
        text = ex.get("text", "") or ""
    except Exception as e:
        text = ""
        fb_err = f"{type(e).__name__}"
    if len(text) < 200:
        fb = fallback_text(raw, final_url or "")
        if len(fb.get("text", "")) > len(text):
            method = "txt-fallback"
            text = fb.get("text", "")
            title = title or fb.get("title", "")
            fb_err = fb.get("fallback_error") or "short-json-text-fallback"
    return method, title, date, text, fb_err


class ShardWriter:
    def __init__(self, pages: Path, size: int = sup.SHARD_SIZE):
        self.pages = pages
        self.size = size
        self.idx = 0
        self.fh = None
        self.zw = None
        self.n = 0
        self.files = []

    def _open(self):
        name = f"vi-topup-{self.idx:05d}.jsonl.zst"
        self.idx += 1
        path = self.pages / name
        if path.exists() and path.stat().st_size == 0:
            path.unlink()          # crashed prior attempt: reuse empty shard
        self.fh = path.open("xb")
        self.zw = zstd.ZstdCompressor(level=6).stream_writer(
            self.fh, closefd=False)
        self.n = 0
        self.files.append(name)

    def write(self, rec):
        if self.zw is None:
            self._open()
        self.zw.write((json.dumps(rec, ensure_ascii=False) + "\n")
                      .encode("utf-8"))
        self.n += 1
        if self.n >= self.size:
            self.close()
            self._open()

    def close(self):
        if self.zw is None:
            return
        self.zw.flush(zstd.FLUSH_FRAME)
        self.zw.close()
        try:
            self.fh.flush()
            os.fsync(self.fh.fileno())
        finally:
            self.fh.close()
        self.zw = self.fh = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Batch:
    def __init__(self, opts):
        self.opts = opts
        self.out_dir = opts.out_root / opts.batch
        self.pages = self.out_dir / "pages"
        self.audit = self.out_dir / "audit"
        self.writer = ShardWriter(self.pages)
        self.limiter = sup.BandwidthLimiter()
        self.routes = None
        self.last_start = {}
        self.host_sems = {}
        self.nbytes = 0
        self.ok = 0
        self.permanent = 0
        self.blocked = 0
        self.transient = 0
        self.dead_404 = 0
        self.attempts = 0
        self.results = []
        self.per_host = defaultdict(Counter)
        self.err_types = Counter()
        self.audit_stats = {}
        self.t0 = time.time()

    def log(self, msg: str):
        el = round(time.time() - self.t0, 1)
        print(f"[{time.strftime('%H:%M:%S')} +{el}s] {msg}", flush=True)

    async def gate(self, stats: dict):
        """Pause while STOP or heavy_job.lock is present."""
        warned = False
        waited = 0.0
        while stop_hit() or lock_hit():
            if not warned:
                self.log(f"PAUSE stop={stop_hit()} lock={lock_hit()}")
                warned = True
            if waited > self.opts.max_stop_wait:
                raise TimeoutError("stop_or_lock_wait_exceeded")
            await asyncio.sleep(20.0)
            waited += 20.0
        if warned:
            self.log("RESUME")
            stats["paused_s"] = stats.get("paused_s", 0.0) + waited

    async def handle(self, row: dict, total_sem: asyncio.Semaphore):
        host = row["host"]
        sem_h = self.host_sems.setdefault(host, asyncio.Semaphore(self.opts.per_host))
        async with total_sem:
            async with sem_h:
                wait = self.opts.delay - (time.monotonic()
                                          - self.last_start.get(host, 0.0))
                if wait > 0:
                    await asyncio.sleep(wait)
                entry = {"id": row["id"], "host": host, "url": row["url"],
                         "prior_kind": row.get("prior_kind", "")}
                for attempt in (1, 2):
                    await self.gate(self.audit_stats)
                    self.last_start[host] = time.monotonic()
                    idx, purl = self.routes.pick(host)
                    t0 = time.time()
                    try:
                        res = await asyncio.wait_for(
                            sup.run_browser_one(
                                row, self.opts.crawl_root, self.limiter,
                                self.opts.chrome_exe or None,
                                timeout_s=self.opts.timeout,
                                proxy_url=purl, headless=True, pool=None),
                            timeout=self.opts.timeout + 45.0)
                    except Exception as e:
                        res = {"ok": False, "status": None,
                               "final_url": row["url"], "title": "",
                               "raw": b"", "nbytes": 0,
                               "error": f"topup:{type(e).__name__}",
                               "infra": True}
                    elapsed = round(time.time() - t0, 2)
                    self.nbytes += res.get("nbytes", 0) or 0
                    self.attempts += 1
                    entry["attempt"] = attempt
                    entry["proxy"] = bool(purl)
                    entry["elapsed"] = elapsed
                    entry["nbytes"] = res.get("nbytes", 0) or 0
                    if res.get("ok"):
                        method, title, date, text, fb_err = extract_html(
                            res.get("raw") or b"", res.get("final_url") or "")
                        outcome = sup.classify_inline(
                            res.get("status"), res.get("final_url") or "",
                            text, title, res.get("raw") or b"")
                        entry["method"] = method
                        entry["status"] = res.get("status")
                        entry["final_url"] = res.get("final_url")
                        entry["text_len"] = len(text)
                        if outcome == "ok":
                            rec = {"id": row["id"], "url": row["url"],
                                   "host": host,
                                   "final_url": res["final_url"],
                                   "status": res["status"], "title": title,
                                   "date": date, "text": text,
                                   "ts": int(time.time())}
                            if tuple(rec) != SCHEMA:
                                raise ValueError("schema drift")
                            self.writer.write(rec)
                        if fb_err:
                            entry["fb"] = fb_err
                    else:
                        outcome = ("infra" if res.get("infra")
                                   else "fetch_error")
                        entry["status"] = res.get("status")
                        entry["error"] = str(res.get("error") or "")[:120]
                    entry["outcome"] = outcome
                    err = str(res.get("error") or "")
                    if err:
                        self.err_types[err.split(":")[0][:40]] += 1
                    if purl is not None and idx is not None:
                        self.routes.report(host, idx, ok=bool(res.get("ok")),
                                           blocked=(outcome == "blocked"))
                    self.per_host[host][outcome] += 1
                    if outcome == "ok":
                        self.ok += 1
                        break
                    if outcome in ("permanent", "blocked"):
                        if outcome == "permanent":
                            self.permanent += 1
                            if entry.get("status") in DEAD_STATUSES:
                                self.dead_404 += 1
                        else:
                            self.blocked += 1
                        break
                    self.transient += 1
                self.results.append(entry)
                done = self.ok + self.permanent + self.blocked + self.transient
                if done % 50 == 0:
                    rate = done / max(time.time() - self.t0, 1) * 60
                    self.log(f"progress done={done} ok={self.ok} "
                             f"perm={self.permanent} blocked={self.blocked} "
                             f"trans={self.transient} rate={rate:.1f}/min "
                             f"mb={self.nbytes / 1e6:.1f}")

    async def run(self, targets: list) -> dict:
        self.audit_stats = {}
        total_sem = asyncio.Semaphore(self.opts.concurrency)
        tasks = [asyncio.create_task(self.handle(t, total_sem))
                 for t in targets]
        await asyncio.gather(*tasks)
        return self.audit_stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path,
                    default=Path(r"D:\r2ai\data\crawl"))
    ap.add_argument("--batch", default="vi_topup_20261026")
    ap.add_argument("--crawl-root", type=Path,
                    default=Path(r"D:\r2ai\data\crawl"))
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--per-host", type=int, default=4)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-stop-wait", type=float, default=7200.0)
    ap.add_argument("--chrome-exe", default=CHROME_EXE)
    a = ap.parse_args(argv)
    if a.delay < 2.0:
        print("[topup] refuse delay < 2.0s", flush=True)
        return 2
    if a.concurrency > 6 or a.per_host > 4:
        print("[topup] refuse concurrency > 6 or per-host > 4", flush=True)
        return 2
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if stop_hit() or lock_hit():
        print("[topup] stop/lock present at startup -> exit 3", flush=True)
        return 3
    out_dir = a.out_root / a.batch
    if out_dir.exists():
        clash = [p for p in out_dir.rglob("*")
                 if p.is_file() and p.stat().st_size > 0
                 and p.suffix != ".log"]
        if clash:
            print(f"[topup] refuse existing output: {out_dir}", flush=True)
            return 2
    (a.out_root / a.batch).mkdir(parents=True, exist_ok=True)
    out_dir = a.out_root / a.batch
    pages, audit = out_dir / "pages", out_dir / "audit"
    pages.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    lower_priority()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    print(f"[topup] selecting targets from {QUEUE}", flush=True)
    targets, sel = select_targets()
    print("[topup] selection " + json.dumps(sel, ensure_ascii=False),
          flush=True)
    if not targets:
        print("[topup] nothing to do", flush=True)
        (audit / "run.json").write_text(json.dumps(
            {"exit": 0, "nothing": True, "selection": sel},
            ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    with (audit / "targets.jsonl").open("x", encoding="utf-8") as f:
        for t in targets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    proxies = sup.load_proxies(PROXY_FILE)
    print(f"[topup] proxies loaded: {len(proxies)}", flush=True)
    if not proxies:
        print("[topup] no proxies -> refuse (targets need rotation)",
              flush=True)
        return 2

    chrome_exe = a.chrome_exe if os.path.exists(a.chrome_exe) else None
    batch = Batch(a)
    batch.routes = sup.RouteTable(proxies)
    status = 0
    try:
        batch.audit_stats = asyncio.run(batch.run(targets))
    except TimeoutError as e:
        print(f"[topup] aborted: {e}", flush=True)
        status = 1
    except KeyboardInterrupt:
        print("[topup] interrupted", flush=True)
        status = 1
    finally:
        batch.writer.close()

    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    shard_rows = []
    strict_all_ok = True
    for name in batch.writer.files:
        p = pages / name
        v = sup.strict_scan_shard(p)
        v.pop("ids", None)
        strict_all_ok = strict_all_ok and bool(v.get("ok"))
        shard_rows.append({"file": name, "sha256": v.get("sha256"),
                           "size": v.get("size"), "records": v.get("records"),
                           "ok": v.get("ok")})
    with (pages / "manifest.sha256").open("w", encoding="utf-8") as f:
        for row in shard_rows:
            f.write(f"{row['sha256']}  {row['file']}\n")
    with (audit / "results.jsonl").open("x", encoding="utf-8") as f:
        for r in batch.results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "exit": status, "batch": a.batch,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(batch.t0)),
        "finished_at": finished_at,
        "selection": sel,
        "config": {"concurrency": a.concurrency, "per_host": a.per_host,
                   "delay": a.delay, "timeout": a.timeout,
                   "proxies": len(proxies), "chrome_exe": chrome_exe,
                   "fresh_browser_per_attempt": True},
        "counts": {"targets": len(targets), "attempts": batch.attempts,
                   "ok": batch.ok, "permanent": batch.permanent,
                   "blocked": batch.blocked, "transient": batch.transient,
                   "dead_404_new": batch.dead_404,
                   "nbytes": batch.nbytes},
        "per_host": {h: dict(c) for h, c in batch.per_host.items()},
        "error_types": dict(batch.err_types.most_common()),
        "shards": shard_rows, "strict_all_ok": strict_all_ok,
        "paused_s": batch.audit_stats.get("paused_s", 0.0),
    }
    (audit / "run.json").write_text(json.dumps(summary, ensure_ascii=False,
                                               indent=1),
                                    encoding="utf-8")
    print("[topup] summary " + json.dumps(
        {"exit": status, "ok": batch.ok, "attempts": batch.attempts,
         "strict_all_ok": strict_all_ok}, ensure_ascii=False), flush=True)
    if status != 0:
        return status
    return 0 if strict_all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
