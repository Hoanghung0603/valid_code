"""Async broad-recovery supervisor for b660 (production, multi-cycle unattended).

Owns: D:\\r2ai\\data\\crawl\\async_recovery_20261009\\ only (queue/state/pages/audit).
Never touches original state/pages, data/index|sub|llm|corpus, controller
docs/jobs, z87. Public pages only.

Architecture (parent-review contract):
  * UNATTENDED MULTI-CYCLE: after one fair pass the supervisor re-cycles
    retriable rows with cooldown, resets per-cycle caps, never asks a user.
    Exit only on STOP, all-resolved, or fatal error.
  * FAIR CONCURRENCY: feeder streams queue shards into bounded per-host
    buffers; N workers pull round-robin across hosts (concurrency in flight,
    per_host per host, exactly 1 browser service). A dead host can never
    monopolise queues: after HOST_ROW_FAIL_STREAK parked rows the host is
    quarantined *this cycle*; single-host shards are then skipped as shards
    with an aggregate deferred count (no per-row log spam).
  * HISTORY (append-only): state/done_ids.tsv lines `id<TAB>kind`, kinds:
    ok|archive_ok|recovered = terminal (output-dedup, never refetched);
    permanent|blocked|transient = genuine attempt (budget MAX_ROW_ATTEMPTS);
    infra = infrastructure-failure marker that RESETS the attempt counter for
    that id (run-1 ModuleNotFoundError / run-2 NotImplementedError style bugs
    must never burn genuine attempts). Migration is append-only: infra rows
    detected from audit/results.jsonl are re-classified by appending markers,
    history is never destructively rewritten. Quarantined/deferred rows write
    NO history line and stay retriable.
  * ROUTES: all proxy credentials loaded at runtime (no log/repo leaks),
    per-host sticky successful route, rotation only after error/blocked with
    durable cooldown; browser-over-proxy for CF IP blocks (CDP auth).
  * BANDWIDTH: one thread-safe token bucket clock, split conservative lanes
    (HTTP ~10Mbps collective for direct/proxy/archive, browser lane ~10Mbps
    with CDP emulation <= lane rate). Chunkwise blocking throttle BEFORE/while
    reads, not bare accounting. Application-level cap only — not an OS
    guarantee (documented).
  * ARCHIVE: typed (status incl. 429) metadata+content results, serial gate,
    persisted global cooldown, zero network while gated.
  * DURABILITY: ok record is published (zstd frame flush + fsync) BEFORE the
    success journal line; startup recovers success ids from strict-readable
    own output; every shard is strict-verified (decompressobj .eof, schema-9,
    status 200, text>=200, challenge fail-closed) and hashed at CLOSE time
    into append-only audit/manifest_closed.jsonl. Singleton lock is PID-alive
    based and held until the closed-verified manifest is written.
  * HEAVY LOCK: heavy_job.lock pauses (status=paused + heartbeat) and then
    RETRIES THE SAME ROW when the lock clears; STOP is honoured mid-sleep.
    Nothing is killed and no row is lost by a pause.

Exit codes: 0 all-resolved; 1 fatal; 2 usage; 3 STOP/refused-by-live-lock.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
MIN_TEXT = 200
MAX_BODY = 3 * 1024 * 1024
ARCHIVE_META_CAP = 256 * 1024
SHARD_SIZE = 20000
FLUSH_EVERY = 50
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")

# ---- fail-closed challenge / anti-bot detection (length-independent) --------
CHALLENGE_URL_MARKERS = (
    "verify.html", "/captcha", "/challenge", "cdn-cgi", "waf",
    "security-check", "chk_js", "aes.js", "geetest", "verifycode",
)
# precise phrases only: broad medical words (e.g. bare 验证/安全) must NOT match
CHALLENGE_TEXT_MARKERS = (
    "just a moment", "attention required", "checking your browser",
    "enable javascript", "please enable javascript", "cf-browser-verification",
    "cf_chl", "challenge-platform", "cdn-cgi/challenge", "turnstile",
    "hcaptcha", "g-recaptcha", "recaptcha/api", "px-captcha", "geetest",
    "verify you are human", "are you a robot", "robot check",
    "ddos protection", "security check required",
    "滑动拼图验证", "安全验证", "访问验证", "人机验证", "请开启javascript",
    "正在验证您的浏览器", "请输入验证码", "验证码", "captcha",
)
# raw-HTML signals: ONLY strong interstitial markers (a normal page embedding
# a reCAPTCHA widget must NOT be flagged; medical FP risk of loose words)
CHALLENGE_RAW_MARKERS = (
    "window._cf_chl", "cf-chl", "chl-pageid", "cdn-cgi/challenge",
    "challenge-platform", "geetest_", "/recaptcha/challenge", "hcaptcha-challenge",
    "滑动拼图验证", "just a moment", "attention required",
)
BLOCK_STATUSES = {403, 429, 503, 521, 522, 523, 524}
PERMANENT_STATUSES = {404, 410}
CF_SHAPED_STATUSES = {403, 429, 503, 521, 522, 523, 524}

ARCHIVE_429_COOLDOWN = 900.0
HOST_BROWSER_CAP = 300        # per-cycle, per-host Chrome batch (auto-resumed)
VI_BROWSER_CAP = None          # per-cycle cap for Vietnamese hosts; None = unbounded (user 22/10)
HOST_BROWSER_FAIL_STREAK = 40
GLOBAL_BROWSER_CAP = 40000    # per-cycle global Chrome batch (VI caps raised 21/10)
HOST_PROXY_FAIL_STREAK = 15   # consecutive proxy failures before a timed cooldown
PROXY_COOLDOWN_BASE = 300.0   # first proxy cooldown after a futile streak
PROXY_COOLDOWN_MAX = 3600.0   # exponential cap; success resets the level
HOST_ROW_FAIL_STREAK = 12
HEADED_HOSTS = {"nhathuoclongchau.com.vn"}
DIRECT_SKIP_HOSTS = {"laodong.vn", "baotayninh.vn", "qdnd.vn",
                       "medlatec.vn", "thaythuocvietnam.vn",
                       "nhathuoclongchau.com.vn"}
# Hosts whose browser lane ALWAYS goes through a proxy route (rotated on
# 403/429). Long Chau evidence 13/10: headed browser on origin IP eats
# 429/403 continuously (~99% of recent rows), quarantining the host.
BROWSER_PROXY_HOSTS = {"nhathuoclongchau.com.vn"}
# browser_proxy_hosts run UNBOUNDED per cycle (user 15/10: Long Chau browser
# must always run) — exempt from BROWSER_PROXY_CAP and GLOBAL_BROWSER_CAP;
# still bounded by route cooldown, host pace, browser semaphore, 20Mbps lane.
BROWSER_PROXY_ROTATE_EVERY = 15  # force next route after N uses (rải tải; 25→15)
VI_SUFFIXES = (".vn",)


def is_vi_host(host: str) -> bool:
    h = (host or "").lower()
    return h.endswith(VI_SUFFIXES)


POOL_DEAD_MARKERS = ("session with given id not found",
                     "no target with given id",
                     "connectionclosederror", "connection closed",
                     "no close frame received")


def is_pool_dead(error: str) -> bool:
    s = (error or "").lower()
    return any(m in s for m in POOL_DEAD_MARKERS)
TERMINAL_KINDS = ("ok", "archive_ok", "recovered")
RETRY_KINDS = ("retry",)
GENUINE_KINDS = ("permanent", "blocked", "transient", "parked")
MAX_ROW_ATTEMPTS = 3
INFRA_MARKERS = ("ModuleNotFoundError", "ImportError", "NotImplementedError",
                 "nodriver-import", "NameError")
HTTP_LANE_BPS = 40_000_000 / 8    # user-approved full: ~40 Mbps http lane
BROWSER_LANE_BPS = 40_000_000 / 8  # user-approved full: ~40 Mbps browser lane
BROWSER_SESSION_CAP = 20 * 1024 * 1024   # user-approved: 20MB/page (was 10MB)
BROWSER_SETTLE_MAX = 20.0                # user-approved: settle <=20s (was 30s)
BROWSER_POOL_TABS = 10                   # tab reuse per browser before recycle
MAX_BROWSER_POOLS = 8                    # LRU cap on live pools (1 Chrome each)
POOL_IDLE_CLOSE_S = 300.0                # close pools idle this long (janitor)
POOL_DEAD_STREAK_DISCARD = 3             # discard pool after N pool-dead rows
FRAME_ABSENT_STREAK_DISCARD = 3  # discard pool after N main-frame-absent
POOL_ACQUIRE_TIMEOUT = 15.0              # fail fast, don't pile on dead pool

DEFAULT_CRAWL_ROOT = Path(r"D:\r2ai\data\crawl")
DEFAULT_WORKDIR = Path(r"D:\r2ai\data\crawl\async_recovery_20261009")
DEFAULT_LOCK = Path(r"D:\r2ai\locks\heavy_job.lock")


class StopRequested(Exception):
    """STOP file observed; drain and exit 3."""


def host_of(url: str) -> str:
    from urllib.parse import urlparse
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
            os.nice(10)
    except Exception as e:
        print(f"[async] WARN lower_priority: {e}", flush=True)


def pid_alive(pid: int) -> bool:
    """True iff PID exists (Windows: OpenProcess probe; POSIX: signal 0)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_uint()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_process_tree(pid: int) -> bool:
    """Force-kill a process and its children (Windows taskkill /T /F).

    Returns True when the PID is gone afterwards. Never raises."""
    if pid <= 0:
        return True
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=20, check=False)
        else:
            os.kill(int(pid), 9)
    except Exception:
        pass
    return not pid_alive(pid)


async def stop_browser_reap(browser, wait_s: float = 2.0):
    """Stop a nodriver browser AND reap its process (leak fix).

    nodriver's stop() can silently fail to terminate the Chrome tree when the
    CDP session is already dead (observed: hundreds of leaked headless Chrome).
    After stop() this waits briefly and force-kills the recorded PID tree."""
    if browser is None:
        return
    pid = None
    try:
        pid = getattr(browser, "_process_pid", None)
        if not pid:
            pid = getattr(getattr(browser, "_process", None), "pid", None)
        pid = int(pid) if pid else None
    except (TypeError, ValueError):
        pid = None
    try:
        res = browser.stop()
        if asyncio.iscoroutine(res):
            await asyncio.wait_for(res, timeout=10.0)
    except Exception:
        pass
    if pid:
        deadline = time.monotonic() + max(0.0, wait_s)
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return
            await asyncio.sleep(0.25)
        if pid_alive(pid):
            kill_process_tree(pid)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_status(st):
    """CDP Response.status arrives as FLOAT (200.0); urllib may give str/None.

    Accept only finite integer-valued int/float in HTTP range 100..599 and
    normalise to int. Everything else (None, "200", nan, 200.5, bool) -> None
    so callers fail closed instead of faking 200."""
    if isinstance(st, bool):
        return None
    if isinstance(st, int):
        return st if 100 <= st <= 599 else None
    if isinstance(st, float):
        if math.isfinite(st) and float(st).is_integer():
            iv = int(st)
            if 100 <= iv <= 599:
                return iv
    return None


def _urls_match(a: str, b: str) -> bool:
    """scheme+host+path equality (fragment/query-insensitive) for DOM matching."""
    from urllib.parse import urlparse

    def norm(u):
        try:
            p = urlparse(u or "")
            return (p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/")
        except Exception:
            return None
    na, nb = norm(a), norm(b)
    return na is not None and na == nb


def site_key(host: str) -> str:
    """Registrable-ish key for VN hosts (www.a.com.vn -> a.com.vn)."""
    h = (host or "").split(":")[0].strip().lower().strip(".")
    parts = [p for p in h.split(".") if p]
    if len(parts) >= 3 and parts[-2] in {
            "com", "gov", "org", "net", "edu", "ac", "health"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def same_site(host: str, url: str) -> bool:
    """True when url belongs to the requested host's registrable site."""
    from urllib.parse import urlparse
    try:
        uh = urlparse(url or "").hostname or ""
    except Exception:
        return False
    return bool(uh) and site_key(host) == site_key(uh)


def url_id_conflict(url: str, final_url: str) -> bool:
    """Different numeric article ids on target vs final URL = stale-tab mix."""
    from urllib.parse import urlparse

    def ids(u):
        try:
            path = urlparse(u or "").path or ""
        except Exception:
            return set()
        return set(re.findall(r"\d{5,}", path))
    a, b = ids(url), ids(final_url)
    return bool(a and b and not (a & b))


def is_challenge(status, final_url, text, title="", raw=b"") -> bool:
    """Fail-closed challenge/anti-bot detection over url/title/text/raw.

    Length-independent: a long challenge page is still a challenge. Precise
    phrases only (no broad medical false positives)."""
    fu = (final_url or "").lower()
    if any(m in fu for m in CHALLENGE_URL_MARKERS):
        return True
    blob = "\n".join((text or "", title or "")).lower()
    if any(m in blob for m in CHALLENGE_TEXT_MARKERS):
        return True
    if raw:
        head = raw[:200_000].decode("utf-8", "ignore").lower()
        if any(m in head for m in CHALLENGE_RAW_MARKERS):
            return True
    return False


def is_cf_shaped(status, final_url, text) -> bool:
    if status in CF_SHAPED_STATUSES:
        return True
    return is_challenge(status, final_url, text)


def classify_inline(status, final_url, text, title="", raw=b"", error=None):
    """Phân loại kết quả fetch. Fail-closed on challenges irrespective of len.

    parity with scripts/extract.py classify() on non-challenge inputs."""
    if error:
        return "transient"
    st = normalize_status(status)          # CDP float 200.0 -> 200; junk -> None
    if is_challenge(st, final_url, text, title, raw):
        return "blocked"          # challenge NEVER becomes ok, even long text
    if st in BLOCK_STATUSES or any(m in (final_url or "").lower()
                                   for m in CHALLENGE_URL_MARKERS):
        return "blocked"
    if st in PERMANENT_STATUSES:
        return "permanent"
    if st != 200:
        return "transient"        # includes st None: never fake 200
    return "ok" if len(text or "") >= MIN_TEXT else "permanent"


def browser_worth_trying(status, text, error, final_url, host="") -> bool:
    if status in PERMANENT_STATUSES:
        return False
    if "toomanyredirects" in (error or "").lower():
        return True               # cookie/redirect loop: Chrome can pass it
    if host in DIRECT_SKIP_HOSTS:
        return True               # proven direct-fail: straight to browser
    if status in CF_SHAPED_STATUSES:
        return True
    if status == 200 and len(text or "") < MIN_TEXT:
        return True
    if is_cf_shaped(status, final_url, text or ""):
        return True
    return False


def redact(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"(https?://)[^/\s@]+@", r"\1***@", s)
    s = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\s*:\s*[^:\s]+\s*:\s*\S+",
               "***:***:***:***", s)
    s = re.sub(r"(?i)(user(name)?|pass(word)?|pwd|token|secret)\s*[:=]\s*\S+",
               r"\1=***", s)
    return s[:200]


# ---------------------------------------------------------------- bandwidth

class Lane:
    __slots__ = ("rate", "free_at")

    def __init__(self, rate: float):
        self.rate = rate
        self.free_at = 0.0


class BandwidthLimiter:
    """Thread-safe token-bucket clock with split lanes (shared monotonic).

    blocking_throttle() is used from proxy/archive worker THREADS (chunkwise,
    real sleeps); throttle() from the asyncio side. Both lanes share one lock
    and one clock so no lane can silently borrow the other's budget. The cap
    is application-level and conservative — not an OS-level guarantee."""

    def __init__(self, http_bps=HTTP_LANE_BPS, browser_bps=BROWSER_LANE_BPS):
        self._lock = threading.Lock()
        self.http = Lane(http_bps or 0.0)
        self.browser = Lane(browser_bps or 0.0)

    def _reserve(self, lane: Lane, nbytes: int) -> float:
        with self._lock:
            if not lane.rate:
                return 0.0
            now = time.monotonic()
            start = max(now, lane.free_at)
            lane.free_at = start + max(0, int(nbytes)) / lane.rate
            return start - now

    def reserve(self, nbytes: int, now: float | None = None, lane: str = "http") -> float:
        return self._reserve(self.http if lane == "http" else self.browser, nbytes)

    def blocking_throttle(self, nbytes: int, lane: str = "http"):
        """Chunkwise pacing for worker threads: wait BEFORE the next read."""
        delay = self._reserve(self.http if lane == "http" else self.browser, nbytes)
        if delay > 0:
            time.sleep(min(delay, 30.0))

    async def throttle(self, nbytes: int, lane: str = "http"):
        delay = self._reserve(self.http if lane == "http" else self.browser, nbytes)
        if delay > 0:
            await asyncio.sleep(min(delay, 30.0))


# -------------------------------------------------------------------- routes

def parse_proxy_line(line: str):
    from urllib.parse import urlparse
    line = (line or "").strip()
    if not line or line.startswith("#"):
        return None
    try:
        if "://" in line:
            p = urlparse(line)
            if p.scheme not in ("http", "https") or not p.hostname or not p.port:
                return None
            auth = f"{p.username}:{p.password}@" if p.username else ""
            return f"{p.scheme}://{auth}{p.hostname}:{p.port}"
        parts = line.split(":")
        if len(parts) == 4 and parts[0] and parts[1].strip().isdigit():
            ip, port, user, pwd = [x.strip() for x in parts]
            if not ip or not user or not pwd:
                return None
            return f"http://{user}:{pwd}@{ip}:{port}"
    except Exception:
        return None
    return None


def load_proxies(proxy_file: Path, max_n: int = 100):
    """Runtime-only parse of ALL credentials (default 100). Never logged."""
    out = []
    try:
        with Path(proxy_file).open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                if len(out) >= max_n:
                    break
                pu = parse_proxy_line(raw)
                if pu:
                    out.append(pu)
    except OSError:
        return []
    return out


class RouteTable:
    """Durable per-host proxy route health.

    Successful route is sticky; rotation happens ONLY after error/blocked with
    an appropriate cooldown. Alternative routes are probed (user-approved)."""

    def __init__(self, routes: list[str]):
        self.routes = list(routes)
        self.hosts: dict[str, dict] = {}

    def _h(self, host: str) -> dict:
        return self.hosts.setdefault(host, {"sticky": None, "last": -1,
                                            "cool": {}, "fail": {},
                                            "uses": {}})

    def pick(self, host: str, now: float | None = None,
             force_rotate: bool = False) -> tuple[int | None, str | None]:
        now = time.monotonic() if now is None else now
        if not self.routes:
            return None, None
        h = self._h(host)
        n = len(self.routes)
        if not force_rotate:
            st = h["sticky"]
            if st is not None and float(h["cool"].get(st, 0.0)) <= now:
                h["uses"][st] = int(h["uses"].get(st, 0)) + 1
                return st, self.routes[st]
        else:
            h.pop("_sticky_hold", None)
        start = (h["last"] + 1) % n
        if force_rotate and h.get("sticky") is not None:
            # round-robin: start AFTER the sticky route, never repeat it
            start = (int(h["sticky"]) + 1) % n
        for k in range(n):
            i = (start + k) % n
            if float(h["cool"].get(i, 0.0)) <= now:
                h["last"] = i
                h["uses"][i] = int(h["uses"].get(i, 0)) + 1
                return i, self.routes[i]
        return None, None

    def report(self, host: str, idx: int, ok: bool, blocked: bool = False):
        h = self._h(host)
        if ok:
            h["sticky"] = idx
            h["fail"][idx] = 0
            h["cool"].pop(idx, None)
            return
        fails = int(h["fail"].get(idx, 0)) + 1
        h["fail"][idx] = fails
        if blocked or fails >= 3:
            h["cool"][idx] = time.monotonic() + min(600.0, 60.0 * fails)

    def host_drop(self, host: str):
        """Drop a route for this host after a futile streak (caller decides)."""
        h = self._h(host)
        if h["sticky"] is not None:
            h["cool"][h["sticky"]] = time.monotonic() + 300.0

    def to_json(self):
        return {"hosts": {h: {"sticky": v["sticky"], "last": v["last"],
                              "cool": {str(k): float(x) for k, x in v["cool"].items()},
                              "fail": {str(k): int(x) for k, x in v["fail"].items()},
                              "uses": {str(k): int(x)
                                       for k, x in v.get("uses", {}).items()}}
                          for h, v in self.hosts.items()}}

    def load_json(self, d):
        for h, v in (d or {}).get("hosts", {}).items():
            self.hosts[h] = {"sticky": v.get("sticky"), "last": int(v.get("last", -1)),
                             "cool": {int(k): float(x) for k, x in v.get("cool", {}).items()},
                             "fail": {int(k): int(x) for k, x in v.get("fail", {}).items()},
                             "uses": {int(k): int(x) for k, x in v.get("uses", {}).items()}}

    def redacted_summary(self) -> str:
        return f"routes={len(self.routes)} hosts={len(self.hosts)}"


def proxy_for_host(host: str, routes: RouteTable, now: float | None = None):
    return routes.pick(host, now)


# --------------------------------------------------------------------- fetch

async def fetch_direct(session, limiter, url, timeout_s=30.0):
    """Direct HTTP. Chunkwise lane throttle while reading. Status never faked."""
    status = final_url = error = None
    infra = False
    body = b""

    async def _get():
        nonlocal status, final_url, body
        async with session.get(url, allow_redirects=True, max_redirects=5) as r:
            status, final_url = r.status, str(r.url)
            buf = bytearray()
            async for chunk in r.content.iter_chunked(64 * 1024):
                buf += chunk
                await limiter.throttle(len(chunk), "http")   # pace, not account
                if len(buf) >= MAX_BODY:
                    break
            body = bytes(buf[:MAX_BODY])

    try:
        await asyncio.wait_for(_get(), timeout=timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        error, infra = classify_exc(type(e).__name__)
    return {"status": status, "final_url": final_url, "body": body,
            "error": error, "infra": infra}


def classify_exc(name: str):
    """(redacted_name, is_infra). Infrastructure/env errors never burn rows."""
    return name, any(m in name for m in INFRA_MARKERS)


def _proxy_parts(proxy_url: str):
    """(proxy_base, (host, port), Basic header) — header value stays in memory
    only; every error path below logs exception NAMES, never headers/creds."""
    from urllib.parse import unquote, urlparse
    p = urlparse(proxy_url if "://" in proxy_url else "http://" + proxy_url)
    if not p.hostname or not p.port:
        return None, None, None
    base = f"http://{p.hostname}:{p.port}"
    hdr = None
    if p.username:
        cred = f"{unquote(p.username)}:{unquote(p.password or '')}"
        hdr = "Basic " + base64.b64encode(cred.encode("utf-8")).decode("ascii")
    return base, (p.hostname, p.port), hdr


class _PreemptiveProxyAuth(object):
    """Attach Proxy-Authorization to every request (no realm dance: urllib's
    ProxyBasicAuthHandler waits for a 407 'proxy' realm that many gateways
    never send -> CONNECT dies unauthenticated).

    NOTE: this deliberately does NOT subclass BaseHandler: the request-level
    header is injected manually in fetch_via_proxy (the urllib opener chain
    only accepts BaseHandler instances)."""

    def __init__(self, header: str):
        self._hdr = header

    def apply(self, req):
        if self._hdr and not req.has_header("Proxy-Authorization"):
            req.add_unredirected_header("Proxy-Authorization", self._hdr)
        return req

    def http_request(self, req):
        return self.apply(req)

    https_request = http_request


class _TunnelHTTPSConnection:
    """http.client.HTTPSConnection subclass factory with preemptive CONNECT
    auth: set_tunnel(..., headers={'Proxy-Authorization': ...})."""

    @staticmethod
    def make(tunnel_headers: dict):
        import http.client

        class _Conn(http.client.HTTPSConnection):
            def set_tunnel(self, host, port=None, headers=None):
                merged = dict(headers or {})
                merged.update(tunnel_headers or {})
                return super().set_tunnel(host, port, headers=merged)

        return _Conn


class _TunnelHTTPSHandler:
    """HTTPSHandler subclass factory doing TLS-through-HTTP-proxy CONNECT.

    MUST win https_open precedence over BOTH ProxyHandler (order 100) and
    the stock HTTPSHandler (order 500): handler_order 90 puts it first.
    (Verified: with order 100 the subclass sat between them and the stock
    handler still claimed https_open.)"""

    @staticmethod
    def make(tunnel_headers: dict, conn_cls):
        import urllib.request

        class _Handler(urllib.request.HTTPSHandler):
            handler_order = 90

            def https_open(self, req):
                return self.do_open(conn_cls, req)

        return _Handler()


def fetch_via_proxy(url: str, proxy_url: str, limiter, timeout=30,
                    max_body=MAX_BODY):
    """Proxy fetch via stdlib urllib (no `requests` on b660 venv).

    * HTTPS: CONNECT tunnel with preemptive Proxy-Authorization in the tunnel
      headers (fixes realm-mismatch/CONNECT-without-cred failures).
    * HTTP: preemptive Proxy-Authorization request header.
    * Chunkwise blocking throttle inside the worker thread; typed real status
      (HTTPError.code preserved incl. 429, CDP-style floats normalised);
      status NEVER faked to 200. NEVER raises; no secrets in errors/logs."""
    t0 = time.time()
    body = b""
    status = None
    final_url = url
    err = None
    infra = False
    try:
        import urllib.error
        import urllib.parse
        import urllib.request
        proxy_base, proxy_hp, auth_hdr = _proxy_parts(proxy_url)
        if not proxy_base:
            return {"status": None, "final_url": url, "body": b"",
                    "error": "proxy-no-route", "infra": False, "nbytes": 0,
                    "elapsed": time.time() - t0}
        handlers = [urllib.request.ProxyHandler({"http": proxy_base,
                                                 "https": proxy_base})]
        preempt = _PreemptiveProxyAuth(auth_hdr) if auth_hdr else None
        tunnel_handler = None
        if auth_hdr:
            is_https = url.lower().startswith("https")
            if is_https:
                target = (urllib.parse.urlparse(url).hostname,
                          urllib.parse.urlparse(url).port or 443)
                th = {"Proxy-Authorization": auth_hdr}
                conn_cls = _TunnelHTTPSConnection.make(th)
                tunnel_handler = _TunnelHTTPSHandler.make(th, conn_cls)
                del target
        opener = urllib.request.build_opener(*handlers)
        # Opener handler PRECEDENCE is what makes the tunnel auth real: the
        # default HTTPSHandler and ProxyHandler fight over https_open in
        # handler_order, so re-register the tunnel handler LAST via
        # add_handler (it must claim https requests first).
        if tunnel_handler is not None:
            opener.add_handler(tunnel_handler)
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"})
        if preempt is not None:
            preempt.apply(req)          # preemptive header (no realm dance)
        with opener.open(req, timeout=timeout) as r:
            status = normalize_status(getattr(r, "status", None)
                                      or getattr(r, "code", None))
            if status is None:
                return {"status": None, "final_url": url, "body": b"",
                        "error": "status-unknown", "infra": False, "nbytes": 0,
                        "elapsed": time.time() - t0}
            final_url = str(getattr(r, "url", url) or url)
            buf = bytearray()
            while len(buf) < max_body:
                chunk = r.read(min(64 * 1024, max_body - len(buf)))
                if not chunk:
                    break
                buf += chunk
                limiter.blocking_throttle(len(chunk), "http")  # real pacing
            body = bytes(buf)
    except Exception as e:
        en = type(e).__name__
        st = normalize_status(getattr(e, "code", None))
        if st is not None:            # HTTPError: keep REAL typed status
            try:
                buf = bytearray()
                while len(buf) < min(max_body, 512 * 1024):
                    chunk = e.read(64 * 1024)
                    if not chunk:
                        break
                    buf += chunk
                    limiter.blocking_throttle(len(chunk), "http")
                body = bytes(buf)
            except Exception:
                body = b""
            return {"status": st, "final_url": url, "body": body,
                    "error": None, "infra": False, "nbytes": len(body),
                    "elapsed": time.time() - t0}
        err, infra = classify_exc(en)
        return {"status": None, "final_url": url, "body": b"",
                "error": redact(err), "infra": infra, "nbytes": 0,
                "elapsed": time.time() - t0}
    return {"status": status, "final_url": final_url, "body": body,
            "error": err, "infra": infra, "nbytes": len(body),
            "elapsed": time.time() - t0}


def wayback_available(url: str, limiter=None, timeout=20):
    """Availability API (never CDX). TYPED result: (snapshot_url, status, error).

    urllib.HTTPError keeps its real code — 429 is returned as status 429 so
    the caller can gate the global archive lane. Body capped + budgeted."""
    import urllib.error
    import urllib.parse
    import urllib.request
    api = ("https://archive.org/wayback/available?url="
           + urllib.parse.quote(url, safe=""))
    try:
        req = urllib.request.Request(api, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = normalize_status(getattr(r, "status", None)
                                      or getattr(r, "code", None))
            raw = b""
            while len(raw) < ARCHIVE_META_CAP:
                chunk = r.read(min(16 * 1024, ARCHIVE_META_CAP - len(raw)))
                if not chunk:
                    break
                raw += chunk
                if limiter is not None:
                    limiter.blocking_throttle(len(chunk), "http")
            if status is None:
                return None, None, "status-unknown"
            data = json.loads(raw.decode("utf-8", "ignore") or "{}")
        snap = (data.get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available") and snap.get("url"):
            return snap["url"], status, None
        return None, status, None
    except urllib.error.HTTPError as e:
        return None, normalize_status(getattr(e, "code", None)), None
    except Exception as e:
        name, _ = classify_exc(type(e).__name__)
        return None, None, redact(name)


def fetch_snapshot_text(snapshot_url: str, limiter=None, timeout=30):
    """Wayback content fetch. TYPED (raw, status, error); 429 stays 429.

    Chunkwise budgeted reads; never fakes 200; never raises."""
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(snapshot_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status = normalize_status(getattr(r, "status", None)
                                      or getattr(r, "code", None))
            if status is None:
                return None, None, "status-unknown"
            if status != 200:
                return None, status, None
            buf = bytearray()
            while len(buf) < MAX_BODY:
                chunk = r.read(min(64 * 1024, MAX_BODY - len(buf)))
                if not chunk:
                    break
                buf += chunk
                if limiter is not None:
                    limiter.blocking_throttle(len(chunk), "http")
            return bytes(buf), status, None
    except urllib.error.HTTPError as e:
        return None, normalize_status(getattr(e, "code", None)), None
    except Exception as e:
        name, _ = classify_exc(type(e).__name__)
        return None, None, redact(name)


# ------------------------------------------------------------------- extract

async def extract_in_pool(pool, body: bytes, url: str):
    import trafilatura
    loop = asyncio.get_running_loop()

    def _job():
        out = trafilatura.extract(
            body, url=url, output_format="json", with_metadata=True,
            include_comments=False, include_tables=True, favor_recall=True)
        if not out:
            return {"title": "", "date": "", "text": ""}
        try:
            d = json.loads(out)
        except ValueError:
            txt = trafilatura.extract(
                body, url=url, output_format="txt", with_metadata=False,
                include_comments=False, include_tables=True,
                favor_recall=True) or ""
            return {"title": "", "date": "", "text": txt}
        return {"title": d.get("title") or "", "date": d.get("date") or "",
                "text": d.get("text") or ""}
    res = await loop.run_in_executor(pool, _job)
    return res


# ------------------------------------------------------------------- browser

class BrowserPool:
    """Reuse ONE nodriver browser per (headless, proxy) lane across rows.

    A fresh browser per URL costs 2-3s start/stop each; reuse cuts that to a
    tab open/close. Isolation: one tab per row (closed in finally), cookies
    cleared between hosts. A crashed browser is discarded and rebuilt lazily.
    Max BROWSER_POOL_TABS tabs per browser, then recycle the browser.
    """

    def __init__(self, crawl_root: Path, chrome_exe: str | None,
                 headless: bool, proxy_url: str | None,
                 browser_args: list):
        self.crawl_root = crawl_root
        self.chrome_exe = chrome_exe
        self.headless = headless
        self.proxy_url = proxy_url
        self.browser_args = list(browser_args)
        self.browser = None
        self.lock = None
        self.tabs_used = 0
        self.last_host = ""
        self.last_used = time.monotonic()
        self.pool_dead_streak = 0      # consecutive pool-dead acquires
        self.frame_absent_streak = 0   # consecutive main-frame-absent pages

    async def _start_browser(self, timeout_s: float):
        """Spawn a fresh Chrome. Caller MUST hold self.lock (or be _ensure).

        Pool browsers MUST carry --proxy-server when the pool has a proxy_url
        (the non-pool path in run_browser_one already does; the pool path did
        not, so every pooled row went out on the origin IP)."""
        import asyncio as _aio
        sys.path.insert(0, str(self.crawl_root / "tools" / "browser_deps"))
        import nodriver as nd
        args = list(self.browser_args)
        flag = self.proxy_server_flag()
        if flag:
            args.append(flag)
        kw = {"headless": bool(self.headless), "browser_args": args}
        if self.chrome_exe:
            kw["browser_executable_path"] = self.chrome_exe
        self.browser = await _aio.wait_for(nd.start(**kw), timeout=timeout_s)
        self.tabs_used = 0
        self.last_used = time.monotonic()

    async def _ensure(self, timeout_s: float):
        import asyncio as _aio
        if self.lock is None:
            self.lock = _aio.Lock()
        if self.browser is not None:
            return
        # RACE FIX: nd.start() must run under the pool lock. Previously every
        # concurrent worker could pass the `browser is None` check and spawn
        # its own Chrome; only the last assignment was kept and every other
        # process leaked (observed: 200+ orphaned headless Chrome).
        async with self.lock:
            if self.browser is not None:
                return
            await self._start_browser(timeout_s)

    async def acquire(self, timeout_s: float):
        await self._ensure(min(timeout_s, POOL_ACQUIRE_TIMEOUT))
        import asyncio as _aio
        async with self.lock:
            self.last_used = time.monotonic()
            if self.tabs_used >= BROWSER_POOL_TABS:
                await self._close_browser()
                await self._start_browser(min(timeout_s, POOL_ACQUIRE_TIMEOUT))
            try:
                # new_tab=True is REQUIRED: nodriver's get() defaults to
                # reusing the FIRST tab, so concurrent workers shared one tab
                # and read each other's DOM (stale final_url/text written to
                # wrong ids). One fresh tab per row keeps rows isolated.
                page = await _aio.wait_for(
                    self.browser.get("about:blank", new_tab=True),
                    timeout=min(timeout_s, POOL_ACQUIRE_TIMEOUT))
            except Exception:
                # dead session: drop the browser, rebuild once, fail this row
                await self._close_browser()
                raise
            self.tabs_used += 1
            return page

    def note_pool_dead(self):
        self.pool_dead_streak += 1
        self.frame_absent_streak = 0

    def note_frame_absent(self):
        self.frame_absent_streak += 1
        self.pool_dead_streak = 0

    def note_page_ok(self):
        self.pool_dead_streak = 0
        self.frame_absent_streak = 0

    def should_discard(self) -> bool:
        return (self.pool_dead_streak >= POOL_DEAD_STREAK_DISCARD
                or self.frame_absent_streak >= FRAME_ABSENT_STREAK_DISCARD)

    async def release(self, page, host: str):
        try:
            if page is not None:
                if host and host != self.last_host and host:
                    try:
                        await page.send(page._tab.cdp.network.clear_browser_cookies()
                                        if hasattr(page, "_tab") else None) \
                            if False else None
                    except Exception:
                        pass
                await asyncio.wait_for(page.close(), timeout=8.0)
        except Exception:
            pass
        finally:
            self.last_host = host or self.last_host

    async     def _close_browser(self):
        try:
            if self.browser is not None:
                await stop_browser_reap(self.browser)
        except Exception:
            pass
        finally:
            self.browser = None
            self.tabs_used = 0
            self.pool_dead_streak = 0
            self.frame_absent_streak = 0

    async def discard(self):
        if self.lock is None:
            await self._close_browser()
            return
        async with self.lock:
            await self._close_browser()

    def proxy_auth(self):
        """(username, password) for CDP Fetch.continue_with_auth.

        nodriver's pump passes auth as (user, pw), so this MUST stay a flat
        2-tuple (a nested shape breaks the pump and pops the proxy dialog)."""
        if not self.proxy_url:
            return None
        from urllib.parse import unquote, urlparse
        pu = urlparse(self.proxy_url if "://" in self.proxy_url
                      else "http://" + self.proxy_url)
        if pu.hostname and pu.port and pu.username:
            return (unquote(pu.username), unquote(pu.password or ""))
        return None

    def proxy_server_flag(self):
        if not self.proxy_url:
            return None
        from urllib.parse import urlparse
        pu = urlparse(self.proxy_url if "://" in self.proxy_url
                      else "http://" + self.proxy_url)
        if pu.hostname and pu.port:
            return f"--proxy-server=http://{pu.hostname}:{pu.port}"
        return None


async def run_browser_one(row, crawl_root: Path, limiter: BandwidthLimiter,
                           chrome_exe: str | None, timeout_s=60.0,
                           proxy_url: str | None = None,
                           headless: bool = True,
                           pool: "BrowserPool | None" = None):
    """Sequential nodriver fetch of ONE row. Never raises; never fakes 200.

    * root frame unknown -> FAIL CLOSED (no record).
    * Document events are collected during navigation and filtered AFTER it:
      frame==root AND loader==nav loader (dynamic challenge redirects may
      finish on a later loader -> fall back to the final Document loader).
    * network bytes counted per request id (DataReceived accumulates,
      LoadingFinished CORRECTS to max, never summed twice), throttled on the
      browser lane; CDP download_throughput <= lane rate (fail-closed cap).
    * optional proxy (CF IP blocks): --proxy-server + CDP auth for creds.
    * pool: reuse a BrowserPool tab when given; else fresh browser (legacy).
    """
    browser = None
    own_browser = pool is None
    proxy_auth = None
    if pool is not None:
        proxy_auth = pool.proxy_auth()   # pool owns the Chrome: its creds win
    page = None
    events: list = []
    req_bytes: dict = {}
    net = {"n": 0}
    capinfo: dict = {}
    pump_task = None
    pump_done = {"v": False}
    url = row["url"]
    try:
        sys.path.insert(0, str(crawl_root / "tools" / "browser_deps"))
        import nodriver as nd
    except BaseException as e:
        return {"ok": False, "status": None, "final_url": url, "title": "",
                "raw": b"", "nbytes": 0,
                "error": f"nodriver-import:{type(e).__name__}", "infra": True}
    try:
        browser_args = ["--blink-settings=imagesEnabled=false",
                        "--disable-features=MediaPreloading",
                        "--disable-features=Translate",
                        "--blink-settings=fontsEnabled=false",
                        "--disable-background-video-track",
                        "--mute-audio",
                        "--disable-component-update",
                        "--disable-background-networking"]
        if pool is None:
            if proxy_url:
                from urllib.parse import urlparse
                pu = urlparse(proxy_url if "://" in proxy_url
                              else "http://" + proxy_url)
                if pu.hostname and pu.port:
                    browser_args.append(
                        f"--proxy-server=http://{pu.hostname}:{pu.port}")
                    if pu.username:
                        proxy_auth = (pu.username, pu.password or "")
        pool_dead = {"v": False}
        if pool is not None:
            try:
                page = await pool.acquire(timeout_s)
            except Exception as e:
                return {"ok": False, "status": None, "final_url": url,
                        "title": "", "raw": b"", "nbytes": 0,
                        "error": f"pool-dead:{type(e).__name__}",
                        "infra": False, "pool_dead": True}
        else:
            kw = {"headless": bool(headless),
                  "browser_args": list(browser_args)}
            if chrome_exe:
                kw["browser_executable_path"] = chrome_exe
            browser = await asyncio.wait_for(nd.start(**kw), timeout=timeout_s)
            page = await asyncio.wait_for(browser.get("about:blank"),
                                          timeout=timeout_s)
        await asyncio.wait_for(page.sleep(0.5), timeout=timeout_s)
        cdp = nd.cdp
        try:
            await page.send(cdp.network.enable())
            await page.send(cdp.network.emulate_network_conditions(
                offline=False, latency=0,
                download_throughput=int(BROWSER_LANE_BPS),
                upload_throughput=125_000))
            capinfo = {"applied": True, "lane_bps": BROWSER_LANE_BPS}
        except Exception as e:
            return {"ok": False, "status": None, "final_url": url, "title": "",
                    "raw": b"", "nbytes": 0,
                    "error": f"cdp-cap-not-applied: {type(e).__name__}",
                    "infra": False}
        # ---- root frame must be known: fail closed otherwise
        try:
            tree = await page.send(cdp.page.get_frame_tree())
            frame = getattr(tree, "frame", None)
            fid = getattr(frame, "id_", None) or getattr(frame, "id", None)
            root_frame = str(getattr(fid, "value", fid)) if fid is not None else None
        except Exception:
            root_frame = None
        if root_frame is None:
            return {"ok": False, "status": None, "final_url": url, "title": "",
                    "raw": b"", "nbytes": 0, "error": "root-frame-unknown",
                    "infra": False}
        # ---- Fetch interception: enable() pauses EVERY request (RequestPaused)
        # and AuthRequired. Callbacks only ENQUEUE (sync, non-blocking); an
        # async pump AWAITs continue_request/continue_with_auth. Credentials
        # are sent ONLY for source == Proxy (never server auth).
        cdp_fetch = getattr(cdp, "fetch", None)
        pump_q: deque = deque()
        pump_err: dict = {"n": 0}
        pump_done = {"v": False}

        def _req_paused_cb(event, _tab=None):
            pump_q.append(("paused", event))

        def _auth_cb(event, _tab=None):
            pump_q.append(("auth", event))

        def _ev_field(event, name):
            v = getattr(event, name, None)
            return getattr(v, "value", v)

        async def _pump(page, cdp, cdp_fetch, auth):
            async def _one(kind, event):
                try:
                    rid = _ev_field(event, "request_id")
                    if rid is None:
                        return
                    if kind == "paused":
                        await page.send(cdp_fetch.continue_request(request_id=rid))
                    else:
                        ch = getattr(event, "auth_challenge", None)
                        src = str(getattr(ch, "source", "") or "")
                        if auth and src.lower().startswith("proxy"):
                            resp = cdp_fetch.AuthChallengeResponse(
                                response="ProvideCredentials",
                                username=auth[0], password=auth[1])
                        else:
                            resp = cdp_fetch.AuthChallengeResponse(
                                response="CancelAuth")
                        await page.send(cdp_fetch.continue_with_auth(
                            request_id=rid, auth_challenge_response=resp))
                except Exception:
                    pump_err["n"] += 1

            while not pump_done["v"]:
                if not pump_q:
                    await asyncio.sleep(0.02)
                    continue
                while pump_q:
                    kind, event = pump_q.popleft()
                    await _one(kind, event)
            # shutdown race: drain anything enqueued before the flag flipped
            while pump_q:
                kind, event = pump_q.popleft()
                await _one(kind, event)

        pump_task = None
        media_blocked = {"n": 0}
        # Media blocking: Network.setBlockedURLs is fail-safe (no request
        # pausing, no main-frame stall). Fetch interception is ONLY enabled
        # when proxy auth is needed; blocking via Fetch paused EVERY request
        # and stalled main frames (cdp-main-frame-absent + handshake timeouts).
        try:
            await page.send(cdp.network.set_blocked_urls(
                ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
                 "*.ico", "*.mp4", "*.webm", "*.mp3", "*.wav", "*.ogg",
                 "*.woff", "*.woff2", "*.ttf", "*.eot"]))
            capinfo["media_blocked"] = True
        except Exception:
            capinfo["media_blocked"] = False
        if cdp_fetch is not None and proxy_auth:
            try:
                await page.send(cdp_fetch.enable(
                    handle_auth_requests=True))
                page.add_handler(cdp_fetch.RequestPaused, _req_paused_cb)
                page.add_handler(cdp_fetch.AuthRequired, _auth_cb)
                pump_task = asyncio.get_running_loop().create_task(
                    _pump(page, cdp, cdp_fetch, proxy_auth))
                capinfo["proxy_auth"] = True
                capinfo["proxy_auth_source"] = "Proxy-only"
            except Exception as e:
                capinfo["proxy_auth"] = False
                capinfo["proxy_auth_error"] = redact(type(e).__name__)

        def _resp_cb(event, _tab=None):
            try:
                resp = getattr(event, "response", event)
                rurl = getattr(resp, "url", "") or ""
                st = normalize_status(getattr(resp, "status", None))
                t = getattr(event, "type_", None)
                rtype = getattr(t, "value", t) or "Document"
                fobj = getattr(event, "frame_id", None)
                fidd = str(getattr(fobj, "value", fobj)) if fobj is not None else None
                lobj = getattr(event, "loader_id", None)
                lid = str(getattr(lobj, "value", lobj)) if lobj is not None else None
                if str(rtype) != "Document":
                    return
                events.append({"url": rurl, "status": st, "frame_id": fidd,
                               "loader_id": lid})
            except Exception:
                pass

        def _rid(event):
            r = getattr(event, "request_id", None)
            return str(getattr(r, "value", r)) if r is not None else "_"

        def _data_cb(event, _tab=None):
            try:
                n = int(getattr(event, "encoded_data_length", 0) or 0)
                rid = _rid(event)
                req_bytes[rid] = req_bytes.get(rid, 0) + n      # accumulate
                net["n"] = sum(req_bytes.values())
            except Exception:
                pass

        def _fin_cb(event, _tab=None):
            try:
                rid = _rid(event)
                fin = int(getattr(event, "encoded_data_length", 0) or 0)
                req_bytes[rid] = max(req_bytes.get(rid, 0), fin)  # correct, no sum2
                net["n"] = sum(req_bytes.values())
            except Exception:
                pass

        page.add_handler(cdp.network.ResponseReceived, _resp_cb)
        page.add_handler(cdp.network.DataReceived, _data_cb)
        page.add_handler(cdp.network.LoadingFinished, _fin_cb)
        await asyncio.sleep(0.5)
        nav = await asyncio.wait_for(page.send(cdp.page.navigate(url)),
                                     timeout=timeout_s)
        # ---- interception drain: Fetch.enable pauses EVERY request, so the
        # pump must get event-loop turns BEFORE we pick the final document.
        # (A sync fake evaluate() may never yield; this bounded wait makes the
        # drain deterministic in prod and in tests.)
        if pump_task is not None:
            for _ in range(40):          # <= ~2s bounded
                if not pump_q:
                    break
                await asyncio.sleep(0.05)

        def _pick_final(dom_url=""):
            """FINAL root document: challenge redirects can open a LATER loader
            holding the real 200 — the initial nav loader is NOT authoritative.
            Prefer the root Document event matching the DOM final URL; else the
            LAST root Document event; nav loader only breaks ties."""
            root_docs = [e for e in events
                         if e.get("frame_id") == root_frame
                         and normalize_status(e.get("status")) is not None]
            if not root_docs:
                return None
            if dom_url:
                for e in reversed(root_docs):
                    if _urls_match(e.get("url"), dom_url):
                        return e
            nav_loader = None
            try:
                lid = getattr(nav, "loader_id", None)
                nav_loader = str(getattr(lid, "value", lid)) if lid is not None else None
            except Exception:
                nav_loader = None
            for e in reversed(root_docs):
                if nav_loader and e.get("loader_id") == nav_loader:
                    return e
            return root_docs[-1]      # final root document wins

        # ---- bounded settle wait: CF/JS pages need time for the FINAL
        # document AND for the DOM body to hold real content. An early
        # event whose URL already matches the row URL (or a 200 shell with
        # an empty body) must NOT end the wait: keep polling readyState,
        # body length and challenge markers until the deadline. A 503
        # challenge match must keep waiting for the final 200; a matched
        # 200 still waits for readyState complete + quiet/stable content
        # (body length stable across polls) or extracted/readiness text
        # >= MIN_TEXT. Events final-DOM URL must match; no empty early 200.
        ev = None
        dom_url = ""
        prev_sig = None
        dom_ready = ""
        dom_body_len = 0
        dom_quiet = 0
        prev_body_len = -1
        dom_challenge = False
        dom_text_ready = 0
        settle_polls = 0
        deadline = time.monotonic() + min(BROWSER_SETTLE_MAX,
                                            max(8.0, timeout_s / 2))
        while True:
            settle_polls += 1
            try:
                dom_url = str(await asyncio.wait_for(
                    page.evaluate("document.location.href"), timeout=5.0) or "")
            except Exception:
                dom_url = ""
            dom_ok = dom_url.startswith("http")
            try:
                dom_ready = str(await asyncio.wait_for(
                    page.evaluate("document.readyState"), timeout=5.0) or "")
            except Exception:
                dom_ready = ""
            try:
                dom_body_len = int(await asyncio.wait_for(
                    page.evaluate(
                        "(document.body?document.body.innerText.length:0)"),
                    timeout=5.0) or 0)
            except Exception:
                dom_body_len = 0
            try:
                dom_text_ready = int(await asyncio.wait_for(
                    page.evaluate(
                        "((document.body?document.body.innerText:'')+"
                        "' '+(document.title||'')).length"),
                    timeout=5.0) or 0)
            except Exception:
                dom_text_ready = dom_body_len
            dom_challenge = False
            try:
                dom_challenge = False
                try:
                    _t = str(await asyncio.wait_for(
                        page.evaluate("document.title"), timeout=5.0) or "")
                except Exception:
                    _t = ""
                try:
                    _cf = int(await asyncio.wait_for(
                        page.evaluate(
                            "document.querySelectorAll("
                            "'form[action*=\"__cf_chl\"],"
                            "input[name=\"__cf_chl_jschl_tk__\"],"
                            "input[name=\"cf_clearance\"]'"
                            ").length"), timeout=5.0) or 0)
                except Exception:
                    _cf = 0
                if _cf > 0:
                    dom_challenge = True
                elif _t.strip().lower() in ("just a moment",
                                            "attention required"):
                    dom_challenge = True
            except Exception:
                dom_challenge = False
            if dom_body_len == prev_body_len and dom_body_len > 0:
                dom_quiet += 1
            else:
                dom_quiet = 0
            prev_body_len = dom_body_len
            ev = _pick_final(dom_url if dom_ok else "")
            if ev is not None:
                sig = (ev.get("url"), ev.get("status"), ev.get("loader_id"),
                       dom_ready, dom_body_len)
                st = normalize_status(ev.get("status"))
                matched = dom_ok and _urls_match(ev.get("url"), dom_url)
                content_ok = (dom_text_ready >= MIN_TEXT or
                              (dom_ready == "complete" and dom_quiet >= 2
                               and dom_body_len >= MIN_TEXT))
                if matched and st == 200 and not dom_challenge and content_ok:
                    break
                if st == 200 and not matched and not dom_ok and content_ok:
                    break           # DOM unreachable but content stable
                stable_200 = (prev_sig == sig and st == 200
                              and not dom_challenge and content_ok)
                if stable_200:
                    break
                prev_sig = sig
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.5)
        if net["n"] > BROWSER_SESSION_CAP:
            try:
                await page.send(cdp.page.stop_loading())
            except Exception:
                pass
            return {"ok": False, "status": None, "final_url": url, "title": "",
                    "raw": b"", "nbytes": net["n"],
                    "error": "session-byte-cap-exceeded", "infra": False}
        if ev is None:
            return {"ok": False, "status": None, "final_url": url, "title": "",
                    "raw": b"", "nbytes": net["n"],
                    "error": "cdp-main-frame-absent", "infra": False,
                    "nav_evidence": list(events)[:8]}
        status = normalize_status(ev.get("status"))
        if status is None:
            return {"ok": False, "status": None, "final_url": url, "title": "",
                    "raw": b"", "nbytes": net["n"],
                    "error": "cdp-status-unknown", "infra": False}
        final_url = ev.get("url") or url
        # ---- stale-tab isolation guard: a pooled tab that is (or was)
        # shared can return ANOTHER row's document. Never accept a final
        # page outside the requested site, or an unrelated numeric id.
        req_host = row.get("host") or ""
        if req_host and not same_site(req_host, final_url):
            return {"ok": False, "status": None, "final_url": final_url,
                    "title": "", "raw": b"", "nbytes": net["n"],
                    "error": "browser-cross-host", "infra": False,
                    "nav_evidence": list(events)[:8]}
        if req_host and dom_ok and dom_url and not same_site(req_host, dom_url):
            return {"ok": False, "status": None, "final_url": final_url,
                    "title": "", "raw": b"", "nbytes": net["n"],
                    "error": "browser-dom-cross-host", "infra": False,
                    "nav_evidence": list(events)[:8]}
        if url_id_conflict(url, final_url):
            return {"ok": False, "status": None, "final_url": final_url,
                    "title": "", "raw": b"", "nbytes": net["n"],
                    "error": "browser-id-mismatch", "infra": False,
                    "nav_evidence": list(events)[:8]}
        html = await asyncio.wait_for(page.get_content(), timeout=timeout_s)
        raw = (html or "").encode("utf-8", "ignore")[:MAX_BODY]
        try:
            title = str(await asyncio.wait_for(page.evaluate("document.title"),
                                               timeout=10.0) or "")[:200]
        except Exception:
            title = ""
        await limiter.throttle(net["n"], "browser")   # settle browser lane
        return {"ok": True, "status": status, "final_url": final_url,
                "title": title, "raw": raw, "nbytes": net["n"], "error": None,
                "infra": False, "cap": capinfo, "nav_evidence": list(events)[:8],
                "settle": {"polls": settle_polls, "ready": dom_ready,
                           "body_len": dom_body_len,
                           "text_ready": dom_text_ready,
                            "quiet": dom_quiet,
                            "challenge": bool(dom_challenge)}}
    except Exception as e:
        errmsg = f"{type(e).__name__}: {e}"[:200]
        if pool is not None and is_pool_dead(errmsg):
            try:
                await pool.discard()
            except Exception:
                pass
            return {"ok": False, "status": None, "final_url": url, "title": "",
                    "raw": b"", "nbytes": net.get("n", 0),
                    "error": f"pool-dead:{errmsg}"[:200], "infra": False,
                    "pool_dead": True}
        if pool is not None and page is None:
            try:
                await pool.discard()
            except Exception:
                pass
        name, infra = classify_exc(type(e).__name__)
        return {"ok": False, "status": None, "final_url": url, "title": "",
                "raw": b"", "nbytes": net.get("n", 0),
                "error": f"{name}: {e}"[:200], "infra": infra}
    finally:
        try:
            pump_done["v"] = True
            if pump_task is not None:
                await asyncio.wait_for(pump_task, timeout=3.0)
        except Exception:
            pass
        if pool is not None:
            try:
                await pool.release(page, row.get("host", ""))
            except Exception:
                try:
                    await pool.discard()
                except Exception:
                    pass
        else:
            try:
                if page is not None:
                    await asyncio.wait_for(page.close(), timeout=8.0)
            except Exception:
                pass
            try:
                if browser is not None:
                    await stop_browser_reap(browser)
            except Exception:
                pass


# ----------------------------------------------------------- output contract

def iter_jsonl_zst(path: Path):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    if raw.strip():
                        yield json.loads(raw.decode("utf-8", "ignore"))
            if buf.strip():
                yield json.loads(buf.decode("utf-8", "ignore"))


def strict_scan_shard(path: Path) -> dict:
    """Contract verify ONE output shard (NOT a double-hash).

    Strict multi-frame zstd walk: decompressobj + .eof PER FRAME + unused_data
    chaining (the proven crawl_verify_closed pattern — a plain decompressobj
    silently stops at frame 1 and would hide later records). Schema-9 exact
    keys; id int; status 200; text>=200; challenge fail-closed. A truncated
    final frame is reported (records counted, ok=false)."""
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    obj = None
    frames = 0
    zstd_err = None
    records = 0
    ids = set()
    schema_bad = content_bad = challenge = malformed = 0
    buf = bytearray()

    def handle(line: bytes):
        nonlocal records, schema_bad, content_bad, challenge, malformed
        if not line.strip():
            return
        try:
            rec = json.loads(line.decode("utf-8", "ignore"))
        except Exception:
            malformed += 1
            return
        if not isinstance(rec, dict) or tuple(rec) != SCHEMA or \
                not isinstance(rec.get("id"), int):
            schema_bad += 1
            return
        if rec.get("status") != 200 or len(rec.get("text") or "") < MIN_TEXT:
            content_bad += 1
            return
        if is_challenge(rec.get("status"), rec.get("final_url"),
                        rec.get("text"), rec.get("title"), b""):
            challenge += 1
            return
        records += 1
        ids.add(rec["id"])

    def feed(out: bytes):
        if not out:
            return
        buf.extend(out)
        while True:
            nl = buf.find(10)
            if nl < 0:
                break
            handle(bytes(buf[:nl]))
            del buf[:nl + 1]

    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 16)
                if not chunk:
                    break
                data = chunk
                while data:
                    if obj is None:
                        obj = dctx.decompressobj()
                    out = obj.decompress(data)
                    feed(out)
                    if obj.eof:
                        frames += 1
                        data = obj.unused_data   # next frame's bytes
                        obj = None
                    else:
                        data = b""
        if obj is not None:
            feed(obj.flush())
    except zstd.ZstdError as e:
        zstd_err = f"{type(e).__name__}: {e}"[:120]
    if buf.strip():
        handle(bytes(buf))     # tail line without newline (partial if truncated)
    strict_eof = (zstd_err is None) and (obj is None or obj.eof) and frames > 0
    if zstd_err is None and obj is not None and not obj.eof:
        zstd_err = "truncated_final_frame"
        strict_eof = False
    if zstd_err is None and frames == 0:
        zstd_err = "empty_or_no_frame"
        strict_eof = False
    return {"file": path.name, "records": records, "ids_unique": len(ids),
            "ids": sorted(ids), "uniques_ok": records == len(ids),
            "frames": frames, "strict_eof": strict_eof, "zstd_error": zstd_err,
            "schema_bad": schema_bad, "content_bad": content_bad,
            "challenge": challenge, "malformed": malformed,
            "sha256": sha256_of(path), "size": path.stat().st_size,
            "ok": bool(strict_eof and not schema_bad and not content_bad
                      and not challenge and not malformed
                      and records == len(ids))}


class ShardWriter:
    """Rotating output writer with publish-before-journal durability.

    write_publish(): zstd frame flush + fsync of the shard happens BEFORE the
    caller writes its success journal line. close() returns the closed file so
    the caller can strict-verify + hash it into the closed manifest."""

    def __init__(self, pages_dir: Path, prefix="recov", max_records=SHARD_SIZE,
                 on_close=None):
        import zstandard as zstd
        self._zstd = zstd
        self.pages_dir = pages_dir
        self.prefix = prefix
        self.max_records = max_records
        self.on_close = on_close
        pages_dir.mkdir(parents=True, exist_ok=True)
        self._idx = 0
        while (pages_dir / f"{prefix}-{self._idx:05d}.jsonl.zst").exists():
            self._idx += 1
        self.files = sorted(p.name for p in pages_dir.glob(f"{prefix}-*.jsonl.zst"))
        self._fh = self._zw = None
        self._n = 0
        self._path = None

    def _open(self):
        self._path = self.pages_dir / f"{self.prefix}-{self._idx:05d}.jsonl.zst"
        self._idx += 1
        self._fh = self._path.open("xb")
        # closefd=False: zw.close() must NOT close our handle (we fsync+close)
        self._zw = self._zstd.ZstdCompressor(level=6).stream_writer(self._fh,
                                                                    closefd=False)
        self._n = 0
        self.files.append(self._path.name)

    def write_publish(self, rec: dict, journal=None):
        if self._zw is None:
            self._open()
        self._zw.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self._n += 1
        self.flush_fsync()                      # durable BEFORE journal
        if journal is not None:
            journal(rec)                        # success journal after publish
        if self._n >= self.max_records:
            self.close()

    def flush_fsync(self):
        if self._zw is not None:
            self._zw.flush(self._zstd.FLUSH_FRAME)
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self):
        if self._zw is None:
            return None
        path = self._path
        try:
            self._zw.flush(self._zstd.FLUSH_FRAME)
            self._zw.close()
        finally:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except (ValueError, OSError):
                pass
            try:
                self._fh.close()
            except (ValueError, OSError):
                pass
        self._zw = self._fh = None
        self._path = None
        if self.on_close is not None:
            self.on_close(path)                 # strict verify + manifest append
        return path


# ------------------------------------------------------------------- history

def load_done_history(state_dir: Path):
    """Parse append-only id<TAB>kind history.

    `infra` markers RESET that id's genuine-attempt counter (run-1/run-2
    infrastructure bugs must not burn attempts) but NEVER un-terminal a
    harvested id. `retry` markers (data-repair only) explicitly un-terminal
    an id after its bad record was removed from the output shard. Terminal
    kinds end the id; MAX_ROW_ATTEMPTS genuine kinds exhaust it."""
    attempts = Counter()
    terminal: set = set()
    p = state_dir / "done_ids.tsv"
    if not p.exists():
        return set(), attempts, terminal
    with p.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            try:
                did = int(parts[0])
            except ValueError:
                continue
            kind = parts[1] if len(parts) > 1 else ""
            if kind == "infra":
                attempts[did] = 0        # append-only repair: fresh budget
            elif kind in RETRY_KINDS:
                terminal.discard(did)    # append-only data repair: refetch
                attempts[did] = 0
            elif kind in TERMINAL_KINDS:
                terminal.add(did)
            elif kind in GENUINE_KINDS:
                attempts[did] += 1
    done = set(terminal)
    for did, n in attempts.items():
        if n >= MAX_ROW_ATTEMPTS and did not in terminal:
            done.add(did)
    return done, attempts, terminal


def migrate_infra_history(state_dir: Path, audit_dir: Path) -> int:
    """Append-only migration: infra-failed ids (from audit/results.jsonl) get
    `id<TAB>infra` markers appended so their bogus attempts reset. History is
    NEVER rewritten destructively. Idempotent via state/history_migrated.json."""
    marker = state_dir / "history_migrated.json"
    if marker.exists():
        return 0
    results = audit_dir / "results.jsonl"
    infra_ids: set = set()
    scanned = 0
    if results.exists():
        with results.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                scanned += 1
                if "ModuleNotFoundError" not in line and "NotImplementedError" not in line \
                        and "nodriver-import" not in line and "ImportError" not in line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                err = str(row.get("error") or "")
                if any(m in err for m in INFRA_MARKERS) and not row.get("ok"):
                    try:
                        infra_ids.add(int(row["id"]))
                    except Exception:
                        continue
    if infra_ids:
        with (state_dir / "done_ids.tsv").open("a", encoding="utf-8") as fh:
            for did in sorted(infra_ids):
                fh.write(f"{did}\tinfra\n")
            fh.flush()
            os.fsync(fh.fileno())
    atomic_json(marker, {"migrated": len(infra_ids), "scanned_lines": scanned,
                         "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return len(infra_ids)


def recover_output_ids(pages_dir: Path) -> set:
    """Success IDs recovered from strict-readable OWN output (crash safety).

    ONLY strict-validated records count (schema-9, status 200, text>=200,
    non-challenge, complete line inside a valid frame): an invalid/truncated
    record must never mark a row terminal, and a valid record must never be
    refetched (no accidental refetch of valid frames)."""
    ids: set = set()
    for p in sorted(pages_dir.glob("recov-*.jsonl.zst")):
        try:
            verdict = strict_scan_shard(p)
        except Exception:
            continue
        for did in verdict.get("ids", []):
            ids.add(did)
    return ids


def atomic_json(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(state_dir: Path) -> dict:
    cp = state_dir / "checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save_checkpoint(state_dir: Path, data: dict):
    atomic_json(state_dir / "checkpoint.json", data)


def iter_queue_shard(path: Path):
    for row in iter_jsonl_zst(path):
        yield row


# --------------------------------------------------------------------- state

class HostState:
    def __init__(self, delay=2.0, min_delay=1.0, max_delay=30.0):
        self.delay = delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.consec_block = 0
        self.cooldown_until = 0.0
        self.next_ok_at = 0.0
        self.counts = Counter()
        self.bad_cycles = 0            # durable: quarantine backoff across cycles
        self.genuine_parks = 0         # durable: real full-chain failures
        self.deferred_rows = 0         # durable: budget-skipped rows seen
        self.full_chain_seen = False   # durable: ≥1 row with all methods tried
        # per-cycle caps (reset every cycle)
        self.browser_attempts = 0
        self.browser_ok = 0
        self.browser_fail_streak = 0
        self.proxy_fail_streak = 0
        self.proxy_blocked_until = 0.0
        self.proxy_block_level = 0
        self.row_fail_streak = 0
        self.quarantined = False

    def begin_cycle(self):
        """Reset per-cycle caps. Successes are NOT erased (done_ids keep them);
        durable health (delay/backoff/bad_cycles/genuine parks) is preserved."""
        self.browser_attempts = 0
        self.browser_ok = 0
        self.browser_fail_streak = 0
        self.proxy_fail_streak = 0
        self.proxy_blocked_until = 0.0
        self.proxy_block_level = 0
        self.row_fail_streak = 0
        self.quarantined = False

    def probe_budget(self) -> int:
        return max(2, HOST_ROW_FAIL_STREAK >> min(self.bad_cycles, 4))

    def browser_allowed(self, host: str = "",
                          browser_proxy_hosts: set | None = None) -> bool:
        if (host not in (browser_proxy_hosts or ())
                and self.browser_fail_streak >= HOST_BROWSER_FAIL_STREAK):
            return False
        if host in (browser_proxy_hosts or ()):
            return True  # unbounded per-cycle (always browser, e.g. Long Chau)
        cap = VI_BROWSER_CAP if is_vi_host(host) else HOST_BROWSER_CAP
        return cap is None or self.browser_attempts < cap

    def proxy_allowed(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now >= self.proxy_blocked_until

    def on_proxy_ok(self):
        self.proxy_fail_streak = 0
        self.proxy_blocked_until = 0.0
        self.proxy_block_level = 0

    def on_proxy_fail(self, now: float | None = None) -> float:
        """Count a proxy failure. A futile streak opens a timed, exponential
        cooldown (instead of banning the lane for the rest of a long cycle)."""
        now = time.monotonic() if now is None else now
        self.proxy_fail_streak += 1
        if self.proxy_fail_streak < HOST_PROXY_FAIL_STREAK:
            return 0.0
        self.proxy_block_level += 1
        cool = min(PROXY_COOLDOWN_MAX,
                   PROXY_COOLDOWN_BASE * (2 ** (self.proxy_block_level - 1)))
        self.proxy_blocked_until = now + cool
        self.proxy_fail_streak = 0
        return cool

    def on_result(self, outcome, now):
        self.counts[outcome] += 1
        if outcome == "ok":
            self.consec_block = 0
            self.delay = max(self.min_delay, self.delay * 0.9)
        else:
            if outcome == "blocked":
                self.consec_block += 1
                self.delay = min(self.max_delay, self.delay * 1.5)
                if self.consec_block >= 3:
                    self.cooldown_until = now + min(120.0, 20.0 * self.consec_block)
            else:
                self.consec_block = 0
                self.delay = min(self.max_delay, self.delay * 1.1)

    def on_row_ok(self):
        self.row_fail_streak = 0
        self.quarantined = False

    def on_row_parked(self):
        self.genuine_parks += 1
        self.row_fail_streak += 1
        if self.row_fail_streak >= self.probe_budget():
            self.quarantined = True

    def on_row_deferred(self):
        """Budget-skip (cap/cooldown/gate): keeps the row PENDING, never burns
        URL budget and never quarantines a productive host."""
        self.deferred_rows += 1

    def note_full_chain(self):
        self.full_chain_seen = True

    def end_cycle(self):
        if self.quarantined:
            self.bad_cycles += 1
        else:
            self.bad_cycles = max(0, self.bad_cycles - 1)

    def to_json(self):
        return {"d": round(self.delay, 3), "cb": self.consec_block,
                "cd": round(self.cooldown_until, 1), "bc": self.bad_cycles,
                "gp": self.genuine_parks, "dr": self.deferred_rows,
                "fc": self.full_chain_seen,
                "ct": dict(self.counts)}

    @classmethod
    def from_json(cls, d):
        hs = cls(delay=float(d.get("d", 2.0)))
        hs.consec_block = int(d.get("cb", 0))
        hs.cooldown_until = float(d.get("cd", 0.0))
        hs.bad_cycles = int(d.get("bc", 0))
        hs.genuine_parks = int(d.get("gp", 0))
        hs.deferred_rows = int(d.get("dr", 0))
        hs.full_chain_seen = bool(d.get("fc", False))
        hs.counts = Counter(d.get("ct", {}))
        return hs


class Gate:
    """STOP (graceful) + heavy_job.lock (pause & retry SAME row) + archive gate.

    STOP is honoured mid-sleep. Lock pauses set status=paused + heartbeat and
    resume the SAME row — never skipping it. Archive 429 gate is global,
    serialised and persisted."""

    def __init__(self, workdir: Path, lock: Path | None, archive_gate_until: float = 0.0):
        self.workdir = workdir
        self.lock = Path(lock) if lock else None
        self.archive_gate_until = float(archive_gate_until or 0.0)
        self.paused = False
        self._archive_lock = threading.Lock()

    def stop_hit(self) -> bool:
        try:
            return (self.workdir / "STOP").exists()
        except OSError:
            return False

    def lock_hit(self) -> bool:
        try:
            return self.lock is not None and self.lock.exists()
        except OSError:
            return False

    def raise_if_stop(self):
        if self.stop_hit():
            raise StopRequested()

    async def sleep(self, secs: float, on_tick=None):
        """Interruptible sleep: STOP aborts within ~0.5s."""
        end = time.monotonic() + max(0.0, secs)
        while True:
            self.raise_if_stop()
            if self.lock_hit():
                return                      # caller re-enters gate()
            left = end - time.monotonic()
            if left <= 0:
                return
            if on_tick is not None:
                on_tick()
            await asyncio.sleep(min(0.5, left))

    async def gate(self, on_pause=None):
        """Before EVERY method attempt: STOP -> abort; lock -> pause here and
        return normally so the SAME row/method is retried after unlock."""
        self.raise_if_stop()
        while self.lock_hit():
            self.paused = True
            if on_pause is not None:
                on_pause()
            self.raise_if_stop()
            await asyncio.sleep(5.0)
        self.paused = False
        self.raise_if_stop()

    def archive_open(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now >= self.archive_gate_until

    def archive_close(self, seconds: float):
        with self._archive_lock:
            self.archive_gate_until = max(self.archive_gate_until,
                                          time.time() + seconds)


# ---------------------------------------------------------------------- opts

class Opts:
    def __init__(self, workdir=DEFAULT_WORKDIR, crawl_root=DEFAULT_CRAWL_ROOT,
                 lock=DEFAULT_LOCK, proxy_file=None,
                 concurrency=8, per_host=2, vi_per_host=4, delay=2.0,
                 max_mbps=20.0, timeout=30.0, extract_workers=3,
                 min_free_gb=5.0, browser_max=1, archive_enabled=True,
                 max_urls=0, max_cycles=0, cycle_cooldown=60.0,
                 headed_hosts=None, vi_first=True,
                 browser_proxy_hosts=None):
        self.workdir = Path(workdir)
        self.crawl_root = Path(crawl_root)
        self.lock = Path(lock) if lock else None
        self.proxy_file = Path(proxy_file) if proxy_file else None
        self.concurrency = concurrency
        self.per_host = per_host
        self.vi_per_host = vi_per_host
        self.delay = delay
        self.max_mbps = max_mbps
        self.timeout = timeout
        self.extract_workers = extract_workers
        self.min_free_gb = min_free_gb
        self.browser_max = browser_max
        self.archive_enabled = archive_enabled
        self.max_urls = max_urls
        self.max_cycles = max_cycles
        self.cycle_cooldown = cycle_cooldown
        self.headed_hosts = set(headed_hosts or HEADED_HOSTS)
        self.browser_proxy_hosts = set(browser_proxy_hosts
                                       or BROWSER_PROXY_HOSTS)
        self.vi_first = vi_first


def per_host_limit(opts, host: str) -> int:
    if is_vi_host(host) and opts.vi_per_host:
        return max(1, min(int(opts.vi_per_host), int(opts.concurrency)))
    return int(opts.per_host)


def validate_opts(o: Opts):
    if o.concurrency < 1 or o.concurrency > 16:
        raise ValueError("--concurrency must be 1..16")
    if o.per_host < 1 or o.per_host > 8:
        raise ValueError("--per-host must be 1..8")
    if o.vi_per_host < 1 or o.vi_per_host > 8:
        raise ValueError("--vi-per-host must be 1..8")
    if o.per_host > o.concurrency:
        raise ValueError("--per-host cannot exceed --concurrency")
    if o.vi_per_host > o.concurrency:
        raise ValueError("--vi-per-host cannot exceed --concurrency")
    if o.delay < 1.0:
        raise ValueError("--delay must be >= 1s")
    if not o.max_mbps or o.max_mbps > 20.0:
        raise ValueError("--max-mbps must be within (0, 20]")
    if o.timeout < 20.0 or o.timeout > 60.0:
        raise ValueError("--timeout must be within [20, 60]")
    if o.extract_workers < 1 or o.extract_workers > 3:
        raise ValueError("--extract-workers must be 1..3 (low CPU)")
    if o.browser_max < 0 or o.browser_max > 8:
        raise ValueError("--browser-max must be 0..8")
    if o.min_free_gb < 2.0:
        raise ValueError("--min-free-gb must be >= 2")
    if o.max_cycles < 0:
        raise ValueError("--max-cycles must be >= 0")
    if o.cycle_cooldown < 5.0:
        raise ValueError("--cycle-cooldown must be >= 5s")


def lock_hit(lock) -> bool:
    try:
        return lock is not None and Path(lock).exists()
    except OSError:
        return False


def stop_hit(workdir: Path) -> bool:
    try:
        return (Path(workdir) / "STOP").exists()
    except OSError:
        return False


# ------------------------------------------------------------------ supervise

def acquire_singleton(lockfile: Path, my_pid: int, tries: int = 5) -> str:
    """Exclusive O_CREAT|O_EXCL creation (atomic_json replace is racy).

    Returns: "acquired" | "held" (live foreign owner) | "busy" (race lost).
    Stale lock (owner PID dead) is removed and the create retried."""
    for _ in range(tries):
        try:
            fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"pid": my_pid,
                                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
            return "acquired"
        except FileExistsError:
            try:
                meta = json.loads(lockfile.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            other = int(meta.get("pid") or 0)
            # ANY live owner (even our own pid, e.g. double supervise) holds it
            if other and pid_alive(other):
                return "held"
            try:
                lockfile.unlink()          # stale: owner died (PID not alive)
            except OSError:
                return "busy"
    return "busy"


async def supervise(opts: Opts) -> int:
    t0 = time.time()
    wd = opts.workdir
    qdir, statedir = wd / "queue", wd / "state"
    pagesdir, auditdir = wd / "pages", wd / "audit"
    for d in (qdir, statedir, pagesdir, auditdir):
        d.mkdir(parents=True, exist_ok=True)
    # ---- singleton: PID-alive based, EXCLUSIVE O_EXCL create (no replace
    # race); HELD until the closed-verified manifest has been written.
    lockfile = statedir / "supervisor.lock"
    my_pid = os.getpid()
    got = acquire_singleton(lockfile, my_pid)
    if got != "acquired":
        print(f"[async] singleton {got} (live owner or race); exit 3", flush=True)
        return 3
    runlog = (auditdir / "supervisor.log").open("a", encoding="utf-8", buffering=1)

    def log(msg):
        runlog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    def heartbeat():
        try:
            lockfile.touch()
        except OSError:
            pass

    # ---- append-only infra migration + history + crash recovery
    migrated = migrate_infra_history(statedir, auditdir)
    done_ids, row_attempts, terminal_ids = load_done_history(statedir)
    recovered = recover_output_ids(pagesdir)
    new_recovered = 0
    if recovered:
        with (statedir / "done_ids.tsv").open("a", encoding="utf-8") as fh:
            for did in sorted(recovered):
                if did not in done_ids:
                    done_ids.add(did)
                    terminal_ids.add(did)
                    row_attempts[did] = 0
                    fh.write(f"{did}\trecovered\n")
                    new_recovered += 1
            if new_recovered:
                fh.flush()
                os.fsync(fh.fileno())
    # ---- routes (ALL credentials), hosts, archive gate (persisted)
    proxies = load_proxies(opts.proxy_file, 100) if opts.proxy_file else []
    routes = RouteTable(proxies)
    cp = load_checkpoint(statedir)
    routes.load_json(cp.get("routes", {}))
    exhausted_hosts: set = set(cp.get("exhausted_hosts", []))
    hosts_state: dict = {}
    for h, hj in cp.get("hosts", {}).items():
        try:
            hosts_state[h] = HostState.from_json(hj)
        except Exception:
            continue
    gate = Gate(wd, opts.lock,
                archive_gate_until=float(cp.get("archive_gate_until", 0.0)))
    queue_total = 0
    shard_rows_hint = {}
    shard_host = {}
    qman = wd / "queue_manifest.json"
    if qman.exists():
        try:
            qm = json.loads(qman.read_text(encoding="utf-8"))
            queue_total = int(qm.get("total", 0))
            for s in qm.get("shards", []):
                shard_rows_hint[s.get("file")] = int(s.get("rows", 0))
                if s.get("host") and s.get("host") != "mixed":
                    shard_host[s.get("file")] = s.get("host")
        except (OSError, ValueError):
            pass
    shards = sorted(qdir.glob("*.jsonl.zst"))
    if not shards:
        print("[async] error: no queue shards in queue/", flush=True)
        return 2
    stat = Counter()
    per_host = defaultdict(Counter)
    per_method = defaultdict(Counter)
    per_host_method = defaultdict(lambda: defaultdict(Counter))
    err_types = Counter()
    bytes_in = {"n": 0}
    browser_used = {"n": 0}
    cycle_no = {"n": 0}
    stop = {"flag": False, "cause": ""}

    def save_ckpt(extra=None):
        data = {"hosts": {h: hs.to_json() for h, hs in hosts_state.items()},
                "routes": routes.to_json(),
                "exhausted_hosts": sorted(exhausted_hosts),
                "archive_gate_until": gate.archive_gate_until,
                "cycle": cycle_no["n"]}
        if extra:
            data.update(extra)
        save_checkpoint(statedir, data)

    def write_status():
        # UNIQUE counters (independent of cycle repeats) vs CUMULATIVE attempts
        harvested_unique = len(terminal_ids)
        exhausted_unique = max(0, len(done_ids) - harvested_unique)
        pending_unique = max(0, queue_total - len(done_ids)) if queue_total else -1
        atomic_json(auditdir / "status.json", {
            "elapsed_s": int(time.time() - t0),
            "cycle": cycle_no["n"], "paused": gate.paused,
            "queue_total": queue_total,
            "harvested_unique": harvested_unique,
            "exhausted_unique": exhausted_unique,
            "pending_unique": pending_unique,
            "deferred_cumulative": stat["deferred_cumulative"],
            "attempted_cumulative": stat["attempted_cumulative"],
            "backlog": pending_unique,
            "browser_used": browser_used["n"],
            "counts": dict(stat),
            "per_host": {h: dict(v) for h, v in per_host.items()},
            "per_method": {h: dict(v) for h, v in per_method.items()},
            "per_host_method": {h: {m: dict(c) for m, c in v.items()}
                                for h, v in per_host_method.items()},
            "error_types": dict(err_types.most_common(20)),
            "body_mb": round(bytes_in["n"] / 1e6, 1),
            "stop": stop,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # ---- closed-shard manifest: strict verify + hash at CLOSE (not end-only)
    manifest_path = auditdir / "manifest_closed.jsonl"

    def on_shard_close(path: Path):
        try:
            v = strict_scan_shard(path)
        except Exception as e:
            v = {"file": path.name, "ok": False, "error": redact(type(e).__name__)}
        v["closed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        log(f"closed-verified shard={path.name} records={v.get('records')} "
            f"strict={v.get('ok')} sha={str(v.get('sha256'))[:12]}")

    writer = ShardWriter(pagesdir, on_close=on_shard_close)
    results_fh = (auditdir / "results.jsonl").open("a", encoding="utf-8", buffering=1)
    done_fh = (statedir / "done_ids.tsv").open("a", encoding="utf-8", buffering=1)

    def journal(rec):
        done_fh.write(f"{rec['id']}\tok\n")
        done_fh.flush()
        os.fsync(done_fh.fileno())
        done_ids.add(rec["id"])
        terminal_ids.add(rec["id"])
        row_attempts[rec["id"]] = 0

    log(f"start shards={len(shards)} routes={len(proxies)} "
        f"archive={'on' if opts.archive_enabled else 'off'} "
        f"done={len(done_ids)} migrated_infra={migrated} recovered={new_recovered}")
    smoke = {"imports": True, "routes": len(proxies),
             "browser_deps": (opts.crawl_root / "tools" / "browser_deps").exists(),
             "archive_gate_open": gate.archive_open(),
             "max_mbps": opts.max_mbps, "lanes": {"http": HTTP_LANE_BPS,
                                                  "browser": BROWSER_LANE_BPS},
             "caps": {"vi_browser": VI_BROWSER_CAP,
                      "global_browser": GLOBAL_BROWSER_CAP,
                      "host_browser": HOST_BROWSER_CAP,
                      "proxy_cooldown": [PROXY_COOLDOWN_BASE,
                                         PROXY_COOLDOWN_MAX]}}
    try:
        import aiohttp, trafilatura, zstandard  # noqa: F401  (env smoke)
    except Exception as e:
        smoke["imports"] = redact(type(e).__name__)
    log(f"smoke env resolved: {json.dumps(smoke, ensure_ascii=False)}")
    lower_priority()
    limiter = BandwidthLimiter()
    pool = ThreadPoolExecutor(max_workers=opts.extract_workers + 4)
    timeout = aiohttp.ClientTimeout(total=opts.timeout, sock_connect=10)
    conn = aiohttp.TCPConnector(limit=opts.concurrency,
                                limit_per_host=opts.per_host, ssl=False,
                                ttl_dns_cache=3600)
    headers = {"User-Agent": UA, "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}
    browser_sem = asyncio.Semaphore(max(1, int(opts.browser_max))) \
        if opts.browser_max else None
    browser_pools: dict = {}     # (headless, proxy_key) -> BrowserPool
    browser_args_shared = ["--blink-settings=imagesEnabled=false",
                           "--disable-features=MediaPreloading",
                           "--disable-features=Translate",
                           "--blink-settings=fontsEnabled=false",
                           "--disable-background-video-track",
                           "--mute-audio",
                           "--disable-component-update",
                           "--disable-background-networking"]

    def get_pool(headless: bool, proxy_url: str | None) -> "BrowserPool":
        key = (bool(headless), proxy_url or "")
        pool = browser_pools.get(key)
        if pool is None:
            pool = BrowserPool(opts.crawl_root, None, bool(headless),
                               proxy_url, browser_args_shared)
            browser_pools[key] = pool
        return pool

    async def pool_janitor():
        """Close idle/overflow pools so rotated proxy routes cannot accumulate
        live Chrome processes (pool-per-route leak). Safe: only closes pools
        whose last acquire is older than 120s (fetch max ~2.5 min), so a live
        fetch is never cut."""
        while not stop["flag"]:
            try:
                await asyncio.sleep(60.0)
                now = time.monotonic()
                live = [(k, p) for k, p in list(browser_pools.items())
                        if p.browser is not None]
                for k, p in live:
                    if now - p.last_used > POOL_IDLE_CLOSE_S:
                        await p.discard()
                live = [(k, p) for k, p in list(browser_pools.items())
                        if p.browser is not None]
                if len(live) > MAX_BROWSER_POOLS:
                    live.sort(key=lambda kv: kv[1].last_used)
                    for k, p in live[:len(live) - MAX_BROWSER_POOLS]:
                        if now - p.last_used > 120.0:
                            await p.discard()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    global_browser = {"n": 0}
    archive_lock = asyncio.Lock()        # serialise archive lane (8 workers)
    archive_next_at = {"t": 0.0}         # global >=3s spacing, archive only

    async def handle_row(session, row):
        rid, url, host = row["id"], row["url"], row["host"]
        hs = hosts_state.setdefault(host, HostState(delay=opts.delay))
        hm = per_host_method[host]
        if hs.quarantined or host in exhausted_hosts:
            return ("quarantined", "blocked",
                    {"error": "host_quarantined_this_cycle"}, {}, "")

        async def host_pace():
            now = time.time()
            if hs.cooldown_until > now:
                await gate.sleep(min(15.0, hs.cooldown_until - now))
            wait = hs.next_ok_at - time.time()
            if wait > 0:
                await gate.sleep(min(wait, 30.0))
            hs.next_ok_at = time.time() + min(max(hs.delay, opts.delay), 30.0)

        host_limit = per_host_limit(opts, host)

        last = {"outcome": "transient", "err": None, "status": None,
                "text": "", "title": "", "raw": b"", "final": url}
        budget = {"skip": False}   # eligible method skipped by cap/cooldown/gate

        async def try_direct():
            if host in DIRECT_SKIP_HOSTS:
                return "continue"        # proven direct-fail: straight to browser
            per_method["direct"]["attempted"] += 1
            hm["direct"]["attempted"] += 1
            await gate.gate()                      # lock/STOP before EACH method
            async with total_sem:
                sem = host_sems.setdefault(host, asyncio.Semaphore(host_limit))
                async with sem:
                    await gate.gate()
                    await host_pace()
                    res = await fetch_direct(session, limiter, url, opts.timeout)
            bytes_in["n"] += len(res.get("body") or b"")
            text, meta, title = "", {}, ""
            raw = res.get("body") or b""
            if res["status"] == 200 and raw:
                try:
                    meta = await extract_in_pool(pool, raw, url)
                    text, title = meta["text"], meta.get("title", "")
                except Exception as e:
                    res["error"] = f"extract:{type(e).__name__}"
                    res["infra"] = True
            if res.get("infra"):
                return ("infra", res)
            outcome = classify_inline(res["status"], res.get("final_url") or "",
                                      text, title, raw, res.get("error"))
            hs.on_result(outcome, time.time())
            last.update(outcome=outcome, err=res.get("error"),
                        status=res["status"], text=text, title=title,
                        raw=raw, final=res.get("final_url") or url)
            if outcome == "ok":
                hm["direct"]["ok"] += 1
                return ("ok", "direct", res, meta, text)
            hm["direct"]["fail"] += 1
            return "continue"

        async def try_browser():
            if not opts.browser_max or browser_sem is None:
                return "continue"
            if not browser_worth_trying(last["status"], last["text"],
                                        last["err"], last["final"], host):
                per_method["browser"]["skipped_not_cf"] += 1
                return "continue"
            exempt = host in opts.browser_proxy_hosts
            if (not hs.browser_allowed(host, opts.browser_proxy_hosts)
                    or (not exempt
                        and global_browser["n"] >= GLOBAL_BROWSER_CAP)):
                per_method["browser"]["skipped_cap"] += 1
                budget["skip"] = True    # eligible method, budget-blocked
                return "continue"
            per_method["browser"]["attempted"] += 1
            hm["browser"]["attempted"] += 1
            hs.browser_attempts += 1
            browser_used["n"] += 1
            if not exempt:
                global_browser["n"] += 1
            await gate.gate()
            force_rr = False
            if host in opts.browser_proxy_hosts:
                hh = routes._h(host)
                st = hh.get("sticky")
                if st is not None and int(hh.get("uses", {}).get(st, 0)) \
                        >= BROWSER_PROXY_ROTATE_EVERY:
                    force_rr = True  # sticky rải đủ N lượt -> ép route kế tiếp
            idx, proxy_url = routes.pick(host, force_rotate=force_rr)
            if force_rr and idx is not None:
                routes.report(host, idx, ok=True)  # ép sticky sang route mới
            if host in opts.browser_proxy_hosts:
                if proxy_url is None and routes.routes:
                    # all proxy routes in cooldown: budget-skip, keep pending
                    per_method["browser"]["skipped_cap"] += 1
                    budget["skip"] = True
                    return "continue"
                _purl = proxy_url  # ALWAYS proxied (rotated on 403/429)
            else:
                _purl = proxy_url if last["status"] in CF_SHAPED_STATUSES \
                    else None
            async with browser_sem:
                await gate.gate()
                await host_pace()
                _pool = get_pool((host not in opts.headed_hosts), _purl)
                res = await asyncio.wait_for(
                    run_browser_one(row, opts.crawl_root, limiter, None,
                                    timeout_s=min(90.0, opts.timeout + 30),
                                    proxy_url=_purl,
                                    headless=(host not in opts.headed_hosts),
                                    pool=_pool),
                    timeout=min(90.0, opts.timeout + 30) + 45.0)
            bytes_in["n"] += res.get("nbytes", 0)
            if res.get("infra"):
                return ("infra", res)
            if res.get("pool_dead"):
                # dead browser already discarded: do NOT count a fail streak
                # (the row never got a real fetch); retry next cycle
                hm["browser"]["fail"] += 1
                err_types["pool-dead"] += 1
                budget["skip"] = True
                last["err"] = res.get("error")
                _pool.note_pool_dead()
                if _pool.should_discard():
                    await _pool.discard()  # Chrome hỏng liên tiếp: thay mới
                # pool-dead thường do route proxy treo tab (ConnectionClosed):
                # phải báo route fail để rotate ngay, nếu không sticky giữ route
                # chết và lặp pool-dead mãi. Khớp hành vi main-frame-absent.
                if _purl is not None and idx is not None:
                    routes.report(host, idx, ok=False, blocked=True)
                return "continue"
            if not res.get("ok"):
                hs.browser_fail_streak += 1
                hm["browser"]["fail"] += 1
                err = redact(str(res.get("error"))[:60])
                err_types[err] += 1
                last["err"] = res.get("error")
                if "main-frame-absent" in (res.get("error") or ""):
                    _pool.note_frame_absent()
                    if _pool.should_discard():
                        await _pool.discard()  # tab stall liên tiếp: thay mới
                else:
                    _pool.note_page_ok()
                if _purl is not None and idx is not None:
                    routes.report(host, idx, ok=False, blocked=True)
                return "continue"
            try:
                meta = await extract_in_pool(pool, res["raw"], res["final_url"])
                text, title = meta["text"], meta.get("title", "")
            except Exception as e:
                hs.browser_fail_streak += 1
                hm["browser"]["fail"] += 1
                return ("infra", {"error": f"extract:{type(e).__name__}"})
            outcome = classify_inline(res["status"], res["final_url"], text,
                                      title, res.get("raw") or b"")
            hs.on_result(outcome, time.time())
            last.update(outcome=outcome, err=None, status=res["status"],
                        text=text, title=title, raw=res.get("raw") or b"",
                        final=res["final_url"])
            if outcome == "ok":
                hs.browser_ok += 1
                hs.browser_fail_streak = 0
                hm["browser"]["ok"] += 1
                _pool.note_page_ok()
                if _purl is not None and idx is not None:
                    routes.report(host, idx, ok=True)
                bres = {"status": res["status"], "final_url": res["final_url"],
                        "body": res.get("raw") or b"", "error": None}
                return ("ok", "browser", bres, meta, text)
            hs.browser_fail_streak += 1
            hm["browser"]["fail"] += 1
            if _purl is not None and idx is not None:
                routes.report(host, idx, ok=False,
                              blocked=(outcome == "blocked"))
            return "continue"

        async def try_proxy():
            if not routes.routes:
                per_method["proxy"]["no_proxy"] += 1
                return "continue"        # method unavailable (not a budget skip)
            if not hs.proxy_allowed():
                per_method["proxy"]["skipped_streak"] += 1
                budget["skip"] = True    # route cooldown = deferred, not burned
                return "continue"
            if last["status"] in PERMANENT_STATUSES:
                per_method["proxy"]["skipped_permanent"] += 1
                return "continue"        # not eligible (404/410)
            await gate.gate()
            rotate = hs.proxy_fail_streak > 0 or hs.proxy_block_level > 0
            idx, proxy_url = routes.pick(host, force_rotate=rotate)
            if not proxy_url:
                per_method["proxy"]["no_route"] += 1
                budget["skip"] = True    # all routes cooled = deferred
                return "continue"
            per_method["proxy"]["attempted"] += 1
            hm["proxy"]["attempted"] += 1
            await host_pace()
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                pool, fetch_via_proxy, url, proxy_url, limiter)
            bytes_in["n"] += res.get("nbytes", 0)
            if res.get("infra"):
                return ("infra", res)
            text, meta, title = "", {}, ""
            raw = res.get("body") or b""
            if res["status"] == 200 and raw:
                try:
                    meta = await extract_in_pool(pool, raw, url)
                    text, title = meta["text"], meta.get("title", "")
                except Exception as e:
                    res["error"] = f"extract:{type(e).__name__}"
                    res["infra"] = True
                    return ("infra", res)
            outcome = classify_inline(res["status"], res.get("final_url") or "",
                                      text, title, raw, res.get("error"))
            hs.on_result(outcome, time.time())
            routes.report(host, idx, ok=(outcome == "ok"),
                          blocked=(outcome == "blocked"))
            last.update(outcome=outcome, err=res.get("error"),
                        status=res["status"], text=text, title=title,
                        raw=raw, final=res.get("final_url") or url)
            if outcome == "ok":
                hs.on_proxy_ok()
                hm["proxy"]["ok"] += 1
                return ("ok", "proxy", res, meta, text)
            cool = hs.on_proxy_fail()
            if cool:
                log(f"proxy cooldown {host} {cool:.0f}s "
                    f"level={hs.proxy_block_level}")
            hm["proxy"]["fail"] += 1
            return "continue"

        async def try_archive():
            if not opts.archive_enabled:
                return "continue"        # method unavailable by config
            async with archive_lock:     # SERIAL: 8 workers must not stampede
                if not gate.archive_open():
                    per_method["archive"]["skipped_429_backoff"] += 1
                    budget["skip"] = True    # 429 gate = DEFERRED, not burned
                    return "continue"          # zero network while gated
                await gate.gate()          # recheck STOP/lock before network
                if not gate.archive_open():
                    per_method["archive"]["skipped_429_backoff"] += 1
                    budget["skip"] = True
                    return "continue"
                # global archive politeness: >=3s between archive requests
                wait = archive_next_at["t"] - time.time()
                if wait > 0:
                    await gate.sleep(min(wait, 5.0))
                    if not gate.archive_open():
                        per_method["archive"]["skipped_429_backoff"] += 1
                        budget["skip"] = True
                        return "continue"
                archive_next_at["t"] = time.time() + 3.0
                per_method["archive"]["attempted"] += 1
                hm["archive"]["attempted"] += 1
                loop = asyncio.get_running_loop()
                snap, st1, err1 = await loop.run_in_executor(
                    pool, wayback_available, url, limiter)
                if st1 == 429:
                    gate.archive_close(ARCHIVE_429_COOLDOWN)
                    save_ckpt()          # persist gate IMMEDIATELY (crash-safe)
                    log(f"archive metadata 429 -> global gate "
                        f"{ARCHIVE_429_COOLDOWN:.0f}s (persisted)")
                    hm["archive"]["fail"] += 1
                    budget["skip"] = True
                    return "continue"
                if not snap:
                    hm["archive"]["fail"] += 1
                    return "continue"
                await gate.gate()
                archive_next_at["t"] = time.time() + 3.0
                raw, st2, err2 = await loop.run_in_executor(
                    pool, fetch_snapshot_text, snap, limiter)
                bytes_in["n"] += len(raw or b"")
                if st2 == 429:
                    gate.archive_close(ARCHIVE_429_COOLDOWN)
                    save_ckpt()          # persist gate IMMEDIATELY
                    log(f"archive content 429 -> global gate "
                        f"{ARCHIVE_429_COOLDOWN:.0f}s (persisted)")
                    hm["archive"]["fail"] += 1
                    budget["skip"] = True
                    return "continue"
                if raw is None or st2 != 200:
                    hm["archive"]["fail"] += 1
                    return "continue"
                try:
                    meta = await extract_in_pool(pool, raw, url)
                    text, title = meta["text"], meta.get("title", "")
                except Exception:
                    hm["archive"]["fail"] += 1
                    return "continue"
                if (len(text) >= MIN_TEXT
                        and not is_challenge(200, snap, text, title, raw[:200_000])):
                    hm["archive"]["ok"] += 1
                    ares = {"status": 200, "final_url": snap, "body": raw,
                            "error": None}
                    return ("archive_ok", "archive", ares, meta, text)
                hm["archive"]["fail"] += 1
                return "continue"

        runners = [("direct", try_direct), ("browser", try_browser),
                   ("proxy", try_proxy), ("archive", try_archive)]
        infra_seen = False
        for method, fn in runners:
            if stop["flag"]:
                raise StopRequested()
            try:
                res = await fn()
            except StopRequested:
                raise
            except Exception as e:
                name, infra = classify_exc(type(e).__name__)
                err_types[f"{method}:{name}"] += 1
                hm[method]["fail"] += 1
                infra_seen = infra_seen or infra
                last["err"] = f"{method}:{name}"
                res = "continue"
            if res == "continue":
                continue
            if isinstance(res, tuple) and res[0] == "infra":
                infra_seen = True
                last["err"] = (res[1] or {}).get("error")
                continue
            return res
        if infra_seen:
            return ("infra", last)
        if budget["skip"]:
            # an ELIGIBLE method was skipped by cap/cooldown/gate -> the row
            # never had a full genuine evaluation: keep it PENDING (no attempt
            # burn, no quarantine); retried automatically next cycle
            return ("deferred", last["outcome"],
                    {"error": last["err"], "status": last["status"]}, {}, "")
        hs.note_full_chain()
        return ("parked", last["outcome"],
                {"error": last["err"], "status": last["status"]}, {}, "")

    total_sem = asyncio.Semaphore(opts.concurrency)
    host_sems: dict = {}

    def record_result(row, res):
        kind = res[0]
        stat["attempted_cumulative"] += 1
        per_host[row["host"]][kind] += 1
        if kind == "quarantined":
            stat["quarantined_parked"] += 1
            stat["deferred_cumulative"] += 1
            # aggregate ONLY: per-row audit writes here would spam millions of
            # lines per cycle for a dead host (parent-review contract)
            return
        if kind == "deferred":
            # budget-skip (cap/cooldown/gate): row stays PENDING — no attempt
            # burn, no quarantine streak, never exhausted
            stat["deferred_cumulative"] += 1
            hs_d = hosts_state.get(row["host"])
            if hs_d is not None:
                hs_d.on_row_deferred()
            results_fh.write(json.dumps(
                {"id": row["id"], "host": row["host"], "group": row.get("group"),
                 "ok": False, "class": "deferred",
                 "status": (res[2] or {}).get("status"),
                 "error": redact(str((res[2] or {}).get("error"))[:80])},
                ensure_ascii=False) + "\n")
            return
        if kind == "infra":
            stat["infra"] += 1
            results_fh.write(json.dumps(
                {"id": row["id"], "host": row["host"], "group": row.get("group"),
                 "ok": False, "class": "infra",
                 "error": redact(str((res[1] or {}).get("error"))[:80])},
                ensure_ascii=False) + "\n")
            with done_fh_lock:
                done_fh.write(f"{row['id']}\tinfra\n")
                done_fh.flush()
                row_attempts[row["id"]] = 0
            return
        hs = hosts_state.get(row["host"])
        if kind in ("ok", "archive_ok"):
            _, method, r, meta, text = res
            rec = {"id": row["id"], "url": row["url"], "host": row["host"],
                   "final_url": r["final_url"], "status": r["status"],
                   "title": meta.get("title", ""), "date": meta.get("date", ""),
                   "text": text, "ts": int(time.time())}
            if tuple(rec) != SCHEMA:
                raise ValueError("schema drift")
            writer.write_publish(rec, journal)   # fsync output THEN journal
            stat["ok"] += 1
            if kind == "archive_ok":
                stat["archive_ok"] += 1
            if method == "proxy":
                stat["proxy_ok"] += 1
            elif method == "browser":
                stat["browser_ok"] += 1
            per_method[method]["ok"] += 1
            audit_row = {"id": row["id"], "host": row["host"],
                         "group": row.get("group"), "method": method, "ok": True,
                         "status": r["status"], "text_len": len(text),
                         "final_url": r["final_url"]}
            if kind == "archive_ok":
                audit_row["provenance"] = "wayback-snapshot"
            results_fh.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
            if hs is not None:
                hs.on_row_ok()
            return
        _, outcome, r, _, _ = res
        stat["parked"] += 1
        done_kind = outcome if outcome in ("permanent", "blocked", "transient") \
            else "transient"
        stat[done_kind] += 1
        results_fh.write(json.dumps(
            {"id": row["id"], "host": row["host"], "group": row.get("group"),
             "ok": False, "class": outcome, "status": (r or {}).get("status"),
             "error": redact(str((r or {}).get("error")))}, ensure_ascii=False) + "\n")
        with done_fh_lock:
            done_fh.write(f"{row['id']}\t{done_kind}\n")
            done_fh.flush()
            row_attempts[row["id"]] = row_attempts.get(row["id"], 0) + 1
            if row_attempts[row["id"]] >= MAX_ROW_ATTEMPTS:
                stat["exhausted"] += 1
                done_ids.add(row["id"])
        if hs is not None:
            hs.on_row_parked()

    done_fh_lock = threading.Lock()
    exit_code = 0

    async def heartbeat_loop():
        """Independent 30s heartbeat+status INSIDE the job: no external
        monitor; keeps lockfile mtime + status.json fresh even while archive
        waits out its gate, Chrome is slow, or the job is heavy-lock paused."""
        while not stop["flag"]:
            try:
                heartbeat()
                write_status()
            except Exception:
                pass
            await asyncio.sleep(30.0)

    hb_task = asyncio.get_running_loop().create_task(heartbeat_loop())
    janitor_task = asyncio.get_running_loop().create_task(pool_janitor())
    try:
        async with aiohttp.ClientSession(headers=headers, timeout=timeout,
                                         connector=conn) as session:
            while True:
                cycle_no["n"] += 1
                for hs in hosts_state.values():
                    hs.begin_cycle()      # per-cycle caps reset; successes kept
                global_browser["n"] = 0
                stat[f"cycle_{cycle_no['n']}"] = 1
                log(f"cycle {cycle_no['n']} begin done={len(done_ids)}")
                buffers: dict[str, deque] = defaultdict(deque)
                producers_left = {"n": len(shards)}
                order = [s.name for s in shards]
                cursor = {"i": 0}
                host_inflight = Counter()

                def fair_host_pick():
                    hosts = [h for h in list(buffers) if buffers[h]]
                    if not hosts:
                        return None
                    if opts.vi_first:
                        vi_hosts = [h for h in hosts if is_vi_host(h)]
                    else:
                        vi_hosts = []
                    if vi_hosts:
                        # VI-first: drain every buffered VI row before ANY
                        # other host gets a slot (strict priority within a
                        # pick pass; quarantine/exhaustion still respected).
                        ordered = sorted(vi_hosts)
                    else:
                        ordered = sorted(hosts)
                    for _ in range(len(ordered)):
                        h = ordered[cursor["i"] % len(ordered)]
                        cursor["i"] += 1
                        hs = hosts_state.get(h)
                        if hs is not None and hs.quarantined:
                            continue
                        if h in exhausted_hosts:
                            continue
                        if host_inflight[h] >= per_host_limit(opts, h):
                            continue
                        return h
                    # second pass: quarantined hosts' leftover rows must still
                    # drain (handle_row fast-parks them, zero network) or the
                    # workers would spin forever on a full buffer
                    for h in ordered:
                        if host_inflight[h] < per_host_limit(opts, h):
                            return h
                    return None

                # ONE PRODUCER PER SHARD (fairness): a 313k-row single-host
                # shard blocking its own bounded buffer must NOT starve the
                # other shards' hosts — all shards feed concurrently and the
                # workers round-robin across hosts.
                async def shard_producer(spath):
                    try:
                        single = shard_host.get(spath.name)
                        if single is not None:
                            if single in exhausted_hosts:
                                stat["deferred_exhausted_hosts"] += max(
                                    1, shard_rows_hint.get(spath.name, 1))
                                return
                            hs_s = hosts_state.get(single)
                            if hs_s is not None and hs_s.quarantined:
                                hint = max(1, shard_rows_hint.get(spath.name, 1))
                                stat["deferred_cumulative"] += hint
                                stat["shards_skipped_quarantined"] += 1
                                log(f"cycle {cycle_no['n']} skip shard "
                                    f"{spath.name} quarantined host={single} "
                                    f"deferred_est={hint}")
                                return
                        for row in iter_queue_shard(spath):
                            if stop["flag"]:
                                return
                            rid, host = row["id"], row["host"]
                            if rid in done_ids:
                                stat["skipped"] += 1
                                continue
                            if host in exhausted_hosts:
                                stat["deferred_exhausted_hosts"] += 1
                                continue
                            hs = hosts_state.get(host)
                            if hs is not None and hs.quarantined:
                                stat["deferred_cumulative"] += 1
                                continue
                            if opts.max_urls and \
                                    stat["attempted_cumulative"] >= opts.max_urls:
                                stop["flag"], stop["cause"] = True, "max_urls_reached"
                                return
                            while len(buffers[host]) >= 64 and not stop["flag"]:
                                await asyncio.sleep(0.05)   # bounded buffers
                            if stop["flag"]:
                                return
                            buffers[host].append(row)
                    finally:
                        producers_left["n"] -= 1

                async def worker():
                    while True:
                        if stop["flag"]:
                            return
                        h = fair_host_pick()
                        if h is None:
                            if producers_left["n"] <= 0 and not any(buffers.values()):
                                return
                            await asyncio.sleep(0.05)
                            continue
                        row = buffers[h].popleft()
                        host_inflight[h] += 1
                        try:
                            res = await handle_row(session, row)
                        except StopRequested:
                            stop["flag"], stop["cause"] = True, "stop_file"
                            return
                        except Exception as e:
                            name, infra = classify_exc(type(e).__name__)
                            err_types[f"row:{name}"] += 1
                            res = ("infra", {"error": name})
                        finally:
                            host_inflight[h] -= 1
                        record_result(row, res)
                        if stat["attempted"] % 25 == 0:
                            save_ckpt()
                            heartbeat()
                        if stat["attempted"] % 200 == 0:
                            write_status()
                        if shutil.disk_usage(wd).free / 1e9 < opts.min_free_gb:
                            stop["flag"], stop["cause"] = True, "low_disk"
                            return

                tasks = [asyncio.create_task(worker())
                         for _ in range(max(1, opts.concurrency))]
                producers = [asyncio.create_task(shard_producer(sp))
                             for sp in shards]
                try:
                    await asyncio.gather(*producers, *tasks)
                except StopRequested:
                    stop["flag"], stop["cause"] = True, "stop_file"
                for hs in hosts_state.values():
                    hs.end_cycle()
                # exhausted_hosts ONLY for hosts with a proven genuine full
                # chain: budget-skipped/archive-gated hosts stay DEFERRED
                for h, hs in hosts_state.items():
                    if (hs.counts.get("ok", 0) == 0 and hs.bad_cycles >= 6
                            and hs.full_chain_seen and hs.genuine_parks >= 6):
                        exhausted_hosts.add(h)
                save_ckpt()
                write_status()
                log(f"cycle {cycle_no['n']} end "
                    f"attempted_cum={stat['attempted_cumulative']} "
                    f"ok={stat['ok']} deferred_cum={stat['deferred_cumulative']} "
                    f"deferred_exhausted={stat['deferred_exhausted_hosts']} "
                    f"skipped={stat['skipped']} stop={stop}")
                if stop["flag"]:
                    break
                # forced next cycle (operator file NEXT_CYCLE): end this cycle
                # early WITHOUT stopping the job — the while loop spins back
                # to begin_cycle() and resets per-cycle caps (non-exempt hosts
                # get a fresh VI/HOST_BROWSER_CAP + GLOBAL_BROWSER_CAP;
                # browser_proxy_hosts are unbounded anyway). Deferred rows stay
                # pending and are retried in the new cycle.
                try:
                    nc = wd / "NEXT_CYCLE"
                    if nc.exists():
                        try:
                            nc.unlink()
                        except OSError:
                            pass
                        log(f"cycle {cycle_no['n']} forced next cycle "
                            f"attempted_cum={stat['attempted_cumulative']} "
                            f"ok={stat['ok']} "
                            f"deferred_cum={stat['deferred_cumulative']}")
                        continue
                except OSError:
                    pass
                # stop when nothing retriable remains (done budget spent)
                remaining = _remaining_retriable(shards, done_ids,
                                                 exhausted_hosts, cap=5000,
                                                 single_host_of=shard_host)
                if remaining == 0:
                    log(f"all rows harvested/exhausted OR parked under deferred "
                        f"exhausted hosts (deferred_exhausted="
                        f"{stat['deferred_exhausted_hosts']}) -> done")
                    break
                if opts.max_cycles and cycle_no["n"] >= opts.max_cycles:
                    log(f"max_cycles={opts.max_cycles} reached")
                    break
                log(f"cycle {cycle_no['n']} cooldown {opts.cycle_cooldown:.0f}s "
                    f"retriable={remaining}")
                try:
                    await gate.sleep(opts.cycle_cooldown, on_tick=heartbeat)
                except StopRequested:
                    stop["flag"], stop["cause"] = True, "stop_file"
                    break
    finally:
        try:
            stop["flag"] = stop["flag"] or hb_task.done()
        except Exception:
            pass
        try:
            hb_task.cancel()
        except Exception:
            pass
        try:
            janitor_task.cancel()
        except Exception:
            pass
        try:
            for _pool in list(browser_pools.values()):
                try:
                    await _pool.discard()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            writer.close()      # strict-verified manifest entries appended here
        except Exception:
            pass
        try:
            results_fh.close()
        except Exception:
            pass
        try:
            done_fh.close()
        except Exception:
            pass
        pool.shutdown(wait=False, cancel_futures=True)
    save_ckpt()
    write_status()
    run_exit = 0 if not stop["flag"] else 3
    if stop["cause"] == "low_disk":
        run_exit = 1
    atomic_json(auditdir / "run.json", {
        "exit": run_exit, "cycles": cycle_no["n"],
        "processed": stat["attempted"], "counts": dict(stat),
        "per_host_top": dict(sorted(((h, dict(v)) for h, v in per_host.items()),
                                    key=lambda kv: -sum(kv[1].values()))[:20]),
        "stop": stop, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    log(f"end exit={run_exit} processed={stat['attempted']} ok={stat['ok']} "
        f"stop={stop}")
    try:
        runlog.close()
    except Exception:
        pass
    try:
        # singleton released ONLY after closed-verified manifest entries exist
        lockfile.unlink()
    except OSError:
        pass
    print(json.dumps({"exit": run_exit, "cycles": cycle_no["n"],
                      "processed": stat["attempted"], "ok": stat["ok"],
                      "deferred_cumulative": stat["deferred_cumulative"], "stop": stop},
                     ensure_ascii=False))
    return run_exit


def _remaining_retriable(shards, done_ids, exhausted_hosts, cap=5000,
                         single_host_of=None):
    """Accurate retriable counter: NO scan-floor abort (a 200k done-prefix must
    not hide a pending tail) and NO first-host break on mixed shards (an
    exhausted first host must not hide other hosts). Only KNOWN single-host
    exhausted shards are skipped whole. Early return at `cap` is fine: the
    caller only needs "is anything left?"."""
    n = 0
    for spath in shards:
        single = single_host_of.get(spath.name) if single_host_of else None
        if single is not None and single in exhausted_hosts:
            continue
        for row in iter_queue_shard(spath):
            if row["id"] in done_ids:
                continue
            if row.get("host") in exhausted_hosts:
                continue
            n += 1
            if n >= cap:
                return n
    return n


def build_opts(argv=None) -> Opts:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--crawl-root", type=Path, default=DEFAULT_CRAWL_ROOT)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--proxy-file", type=Path, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--vi-per-host", type=int, default=4)
    ap.add_argument("--headed-hosts", default=",".join(sorted(HEADED_HOSTS)))
    ap.add_argument("--browser-proxy-hosts",
                    default=",".join(sorted(BROWSER_PROXY_HOSTS)))
    ap.add_argument("--no-vi-first", action="store_true")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--max-mbps", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--extract-workers", type=int, default=3)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--browser-max", type=int, default=1)
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--max-urls", type=int, default=0)
    ap.add_argument("--max-cycles", type=int, default=0)
    ap.add_argument("--cycle-cooldown", type=float, default=60.0)
    a = ap.parse_args(argv)
    headed = {h.strip().lower() for h in str(a.headed_hosts or "").split(",")
              if h.strip()}
    bprox = {h.strip().lower() for h in str(a.browser_proxy_hosts or "")
             .split(",") if h.strip()}
    return Opts(workdir=a.workdir, crawl_root=a.crawl_root, lock=a.lock,
                proxy_file=a.proxy_file, concurrency=a.concurrency,
                per_host=a.per_host, vi_per_host=a.vi_per_host,
                delay=a.delay, max_mbps=a.max_mbps,
                timeout=a.timeout, extract_workers=a.extract_workers,
                min_free_gb=a.min_free_gb, browser_max=a.browser_max,
                archive_enabled=not a.no_archive, max_urls=a.max_urls,
                max_cycles=a.max_cycles, cycle_cooldown=a.cycle_cooldown,
                headed_hosts=headed, vi_first=not a.no_vi_first,
                browser_proxy_hosts=bprox)


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
        print(f"[async] error: {exc}", flush=True)
        return 2
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    lower_priority()
    try:
        return asyncio.run(supervise(opts))
    except KeyboardInterrupt:
        print("[async] interrupted; checkpoint kept.", flush=True)
        return 3
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
