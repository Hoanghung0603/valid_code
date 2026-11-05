"""Crawler đầy đủ cho links_corpus.parquet: resume được, rate-limit thích ứng theo host, lưu text đã trích (zstd).

Thiết kế:
- Mỗi host có 1 worker riêng (mặc định 1 kết nối/host) với delay thích ứng:
  thành công -> giảm dần delay về min; bị chặn (403/429/captcha) -> nhân đôi delay, chặn liên tiếp -> nghỉ cooldown,
  chặn quá nhiều -> tạm bỏ host trong lượt chạy này (URL còn lại sẽ thử ở lượt sau). KHÔNG vượt captcha.
- Trích nội dung (trafilatura) chạy trong process pool để không nghẽn event loop.
- Output: data/crawl/pages/crawl-XXXXX.jsonl.zst  (id, url, host, final_url, status, title, date, text, ts)
  State:  data/crawl/state/log-XXXXX.tsv         (id \t outcome) -- append-only, dùng để resume.
- Dừng an toàn khi ổ đĩa còn dưới --min-free-gb.

Ví dụ:
  python scripts/crawl.py --tiers 1 --limit-per-host 50          # smoke test
  python scripts/crawl.py                                       # chạy tất cả theo thứ tự ưu tiên
  python scripts/crawl.py --hosts vinmec.com,medlatec.vn
"""
import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import classify, extract  # noqa: E402
from shards import ShardWriter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
MAX_BODY = 3 * 1024 * 1024
MAX_ATTEMPTS = 3  # transient/blocked: tối đa số lần thử qua các lượt chạy

# Tier 1: tiếng Việt (ít, quan trọng). Tier 2: Q&A / bách khoa tiếng Trung. Tier 3: phần còn lại.
TIER2_HOSTS = {
    "120ask.com", "ask.39.net", "jbk.39.net", "a-hospital.com", "familydoctor.com.cn", "cnkang.com",
    "youlai.cn", "msdmanuals.cn", "baidu.com", "ask.familydoctor.com.cn", "jb39.com",
}
# Host đã biết chặn cứng khi crawl thử: để cuối cùng, delay lớn.
HARD_HOSTS = {"zysjonline.com", "nhathuoclongchau.com.vn", "tamanhhospital.vn", "vov.vn",
              "pmc-ecm-healthblog.beta.pharmacity.io", "article.iiyi.com", "bingli.iiyi.com"}


async def read_capped(resp, cap: int) -> bytes:
    """Đọc body tới EOF hoặc tối đa `cap` byte.

    Lưu ý: StreamReader.read(n) chỉ trả phần đang có trong buffer (có thể << n), nên phải lặp.
    """
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(64 * 1024):
        buf += chunk
        if len(buf) >= cap:
            break
    return bytes(buf[:cap])


class BandwidthLimiter:
    """Giới hạn tổng tốc độ tải (megabit/s). Tính trên byte đã giải nén nên byte thật trên mạng còn thấp hơn."""

    def __init__(self, mbps: float):
        self.rate = mbps * 1e6 / 8 if mbps and mbps > 0 else 0.0  # byte/s
        self.free_at = 0.0

    def reserve(self, nbytes: int, now: float) -> float:
        """Đặt chỗ `nbytes`, trả về số giây cần chờ trước khi request kế tiếp được phép bắt đầu."""
        if not self.rate:
            return 0.0
        start = max(now, self.free_at)
        self.free_at = start + nbytes / self.rate
        return start - now

    async def throttle(self, nbytes: int):
        delay = self.reserve(nbytes, asyncio.get_running_loop().time())
        if delay > 0:
            await asyncio.sleep(delay)


def lower_priority(nice: int = 10):
    """Chạy ở ưu tiên thấp để không tranh CPU với dịch vụ khác trên máy (con của process pool kế thừa)."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(nice)
    except Exception as e:
        print(f"[WARN] Không hạ được ưu tiên: {e}", flush=True)


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def tier_of(host: str) -> int:
    if host in HARD_HOSTS:
        return 4
    if host.endswith(".vn") or host in {"hellobacsi.com", "vinmec.com", "suckhoeviet.org.vn", "benhvienvietduc.org"}:
        return 1
    if host in TIER2_HOSTS:
        return 2
    return 3


# ---------------------------------------------------------------- state
class State:
    """Log append-only: mỗi dòng 'id\toutcome'. outcome ok/permanent là xong; còn lại đếm số lần thử."""

    def __init__(self, state_dir: Path):
        self.dir = state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        self.done: set[int] = set()
        self.attempts: Counter = Counter()
        for p in sorted(state_dir.glob("log-*.tsv")):
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    sid, outcome = line.split("\t")[:2]
                    i = int(sid)
                except ValueError:
                    continue
                if outcome in ("ok", "permanent"):
                    self.done.add(i)
                else:
                    self.attempts[i] += 1
        idx = len(list(state_dir.glob("log-*.tsv")))
        self._fh = (state_dir / f"log-{idx:05d}.tsv").open("a", encoding="utf-8", buffering=1)

    def pending(self, i: int) -> bool:
        return i not in self.done and self.attempts[i] < MAX_ATTEMPTS

    def record(self, i: int, outcome: str):
        self._fh.write(f"{i}\t{outcome}\n")
        if outcome in ("ok", "permanent"):
            self.done.add(i)

    def close(self):
        self._fh.close()


# ---------------------------------------------------------------- per-host rate control
@dataclass
class HostCtl:
    host: str
    delay: float
    min_delay: float
    max_delay: float = 60.0
    consec_block: int = 0
    blocks: int = 0
    total: int = 0
    gave_up: bool = False
    outcomes: Counter = field(default_factory=Counter)
    recent: deque = field(default_factory=lambda: deque(maxlen=40))  # 40 kết quả gần nhất: True = bị chặn

    def on_result(self, outcome: str, cooldown: float, give_up_after: int) -> float:
        """Cập nhật delay, trả về thời gian cần ngủ trước request tiếp theo."""
        self.total += 1
        self.outcomes[outcome] += 1
        self.recent.append(outcome == "blocked")
        if outcome == "blocked":
            self.blocks += 1
            self.consec_block += 1
            self.delay = min(self.max_delay, self.delay * 2)
            mostly_blocked = len(self.recent) == self.recent.maxlen and sum(self.recent) > self.recent.maxlen // 2
            if self.consec_block >= give_up_after or mostly_blocked:  # chặn liên tiếp hoặc >50% trong 40 request gần nhất
                self.gave_up = True
                return 0
            if self.consec_block >= 3:
                return cooldown
        else:
            self.consec_block = 0
            if outcome == "ok":
                self.delay = max(self.min_delay, self.delay * 0.9)
        return self.delay * random.uniform(0.8, 1.2)


# ---------------------------------------------------------------- crawler
class Crawler:
    def __init__(self, args):
        self.args = args
        out = ROOT / args.out
        self.state = State(out / "state")
        self.writer = ShardWriter(out / "pages", "crawl", max_records=args.shard_size)
        self.stats_path = out / "stats.json"
        self.pool = ProcessPoolExecutor(max_workers=args.extract_workers)
        self.sem = asyncio.Semaphore(args.concurrency)
        self.ctl: dict[str, HostCtl] = {}
        self.t0 = time.time()
        self.counts = Counter()
        self.bytes_in = 0
        self.limiter = BandwidthLimiter(args.max_mbps)
        self.stop_file = out / "STOP"
        self.stop = asyncio.Event()

    def disk_ok(self) -> bool:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < self.args.min_free_gb:
            print(f"[STOP] Ổ đĩa còn {free_gb:.1f}GB < {self.args.min_free_gb}GB", flush=True)
            self.stop.set()
            return False
        return True

    async def fetch_one(self, session, row) -> None:
        rec = {"id": int(row["id"]), "url": row["url"], "host": row["host"]}
        status = final_url = error = None
        body = b""
        async with self.sem:
            try:
                async with session.get(row["url"], allow_redirects=True, max_redirects=5) as r:
                    status, final_url = r.status, str(r.url)
                    body = await read_capped(r, MAX_BODY)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # timeout, DNS, SSL, redirect loop...
                error = type(e).__name__
        await self.limiter.throttle(len(body))  # ngoài semaphore để không giữ chỗ kết nối khi chờ
        text, meta = "", {}
        if status == 200 and body:
            try:
                meta = await asyncio.get_running_loop().run_in_executor(self.pool, extract, body, row["url"])
                text = meta["text"]
            except Exception as e:
                error = f"extract:{type(e).__name__}"
        outcome = classify(status, final_url or "", text, error)
        self.bytes_in += len(body)  # byte đã giải nén (aiohttp tự giải gzip) -> cận trên băng thông
        if outcome == "ok":
            rec.update(final_url=final_url, status=status, title=meta.get("title", ""),
                       date=meta.get("date", ""), text=text, ts=int(time.time()))
            self.writer.write(rec)
        self.state.record(rec["id"], outcome if not error else f"{outcome}:{error}")
        self.counts[outcome] += 1
        return outcome

    async def host_worker(self, session, host: str, rows: list[dict]):
        a = self.args
        ctl = self.ctl.setdefault(host, HostCtl(host, delay=a.delay * (4 if tier_of(host) == 4 else 1), min_delay=a.delay))
        for row in rows:
            if self.stop.is_set() or ctl.gave_up:
                break
            outcome = await self.fetch_one(session, row)
            if ctl.total % 200 == 0 and not self.disk_ok():
                break
            wait = ctl.on_result(outcome, a.cooldown, a.give_up_after)
            if ctl.gave_up:
                print(f"[GIVE UP] {host}: {ctl.consec_block} lần bị chặn liên tiếp, bỏ qua lượt này", flush=True)
                break
            await asyncio.sleep(wait)

    async def watch_stop_file(self):
        """Tạo file <out>/STOP để dừng êm (đợi request đang chạy, lưu tiến độ)."""
        while not self.stop.is_set():
            if self.stop_file.exists():
                print(f"[STOP] Thấy {self.stop_file}, dừng êm...", flush=True)
                self.stop.set()
                return
            await asyncio.sleep(5)

    async def report_loop(self, total: int):
        while not self.stop.is_set():
            await asyncio.sleep(self.args.report_every)
            self.report(total)

    def report(self, total: int):
        n = sum(self.counts.values())
        el = time.time() - self.t0
        s = {
            "elapsed_s": int(el), "processed": n, "planned": total, "rate_per_s": round(n / el, 2) if el else 0,
            "outcomes": dict(self.counts), "free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
            "body_mb": round(self.bytes_in / 1e6, 1), "body_kb_per_page": round(self.bytes_in / 1e3 / n, 1) if n else 0,
            "hosts": {h: {"done": c.total, "delay": round(c.delay, 2), "gave_up": c.gave_up, **c.outcomes}
                      for h, c in sorted(self.ctl.items(), key=lambda kv: -kv[1].total)},
        }
        self.stats_path.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        eta = (total - n) / s["rate_per_s"] / 3600 if s["rate_per_s"] else float("inf")
        print(f"[{el/60:6.1f}m] {n}/{total} {s['rate_per_s']}/s ETA {eta:.1f}h {dict(self.counts)} "
              f"free {s['free_gb']}GB", flush=True)

    async def run(self, plan: pd.DataFrame):
        by_host = defaultdict(list)
        for row in plan.to_dict("records"):
            by_host[row["host"]].append(row)
        total = len(plan)
        print(f"Kế hoạch: {total} URL, {len(by_host)} host", flush=True)
        if not self.disk_ok():
            return
        timeout = aiohttp.ClientTimeout(total=self.args.timeout, sock_connect=10)
        a = self.args
        conn = aiohttp.TCPConnector(limit=a.concurrency, limit_per_host=max(a.per_host, a.big_per_host),
                                    ssl=False, ttl_dns_cache=3600)
        headers = {"User-Agent": UA, "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"}
        async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=conn) as session:
            reporter = asyncio.create_task(self.report_loop(total))
            watcher = asyncio.create_task(self.watch_stop_file())
            # Host lớn (>= big_threshold URL) được nhiều kết nối hơn; URL chia xen kẽ cho các worker của host
            workers = []
            for h, rows in by_host.items():
                k = a.big_per_host if len(rows) >= a.big_threshold else a.per_host
                print(f"  {h}: {len(rows)} URL, {k} kết nối", flush=True)
                workers += [self.host_worker(session, h, rows[i::k]) for i in range(k)]
            try:
                await asyncio.gather(*workers)
            finally:
                self.stop.set()
                reporter.cancel()
                watcher.cancel()
                self.report(total)

    def close(self):
        self.writer.close()
        self.state.close()
        self.pool.shutdown(wait=False, cancel_futures=True)


def build_plan(args, state: State) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data/links_corpus.parquet")
    df["host"] = df.url.map(host_of)
    df["tier"] = df.host.map(tier_of)
    if args.hosts:
        df = df[df.host.isin(set(args.hosts.split(",")))]
    if args.exclude_hosts:
        df = df[~df.host.isin(set(args.exclude_hosts.split(",")))]
    if args.tiers:
        df = df[df.tier.isin({int(t) for t in args.tiers.split(",")})]
    df = df[df.id.map(state.pending)]
    df = df.sample(frac=1, random_state=0)  # trộn để mỗi host phủ đều theo thời gian, tránh dồn 1 chuyên mục
    if args.limit_per_host:
        df = df.groupby("host").head(args.limit_per_host)
    return df.sort_values("tier", kind="stable")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/crawl")
    p.add_argument("--hosts", help="chỉ crawl các host này (phân tách bằng dấu phẩy)")
    p.add_argument("--exclude-hosts", help="bỏ qua các host này (để chia việc giữa các máy/IP)")
    p.add_argument("--tiers", help="vd 1,2 (1=VI, 2=ZH Q&A, 3=ZH khác, 4=host chặn cứng)")
    p.add_argument("--limit-per-host", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=96, help="tổng request đồng thời")
    p.add_argument("--per-host", type=int, default=1, help="số kết nối đồng thời mỗi host thường")
    p.add_argument("--big-per-host", type=int, default=4, help="số kết nối mỗi host lớn (>= --big-threshold URL)")
    p.add_argument("--big-threshold", type=int, default=50000)
    p.add_argument("--max-mbps", type=float, default=0, help="trần băng thông tải (megabit/s, byte đã giải nén); 0 = không giới hạn")
    p.add_argument("--nice", type=int, default=0, help=">0: hạ ưu tiên tiến trình (Linux: nice N; Windows: below normal)")
    p.add_argument("--log-file", help="ghi stdout/stderr vào file (đường dẫn tương đối ROOT)")
    p.add_argument("--delay", type=float, default=1.0, help="delay tối thiểu giữa 2 request cùng worker (s)")
    p.add_argument("--cooldown", type=float, default=120.0, help="nghỉ khi bị chặn 3 lần liên tiếp (s)")
    p.add_argument("--give-up-after", type=int, default=12, help="bỏ host khi bị chặn N lần liên tiếp")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--extract-workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=20000)
    p.add_argument("--min-free-gb", type=float, default=2.0)
    p.add_argument("--report-every", type=float, default=60.0)
    args = p.parse_args()

    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252 -> lỗi khi in tiếng Việt
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if args.log_file:
        log_path = ROOT / args.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(log_path, "a", buffering=1, encoding="utf-8")
    if args.nice > 0:
        lower_priority(args.nice)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    crawler = Crawler(args)
    plan = build_plan(args, crawler.state)
    print(f"Đã xong trước đó: {len(crawler.state.done)} URL", flush=True)
    try:
        asyncio.run(crawler.run(plan))
    except KeyboardInterrupt:
        print("Dừng theo yêu cầu, đã lưu tiến độ.", flush=True)
    finally:
        crawler.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
