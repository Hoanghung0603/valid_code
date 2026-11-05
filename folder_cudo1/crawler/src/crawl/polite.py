"""Crawler lịch sự cho crawl diện rộng: lưu HTML thô vào parquet, giới hạn theo nhóm site, chạy lại được.

Khác crawl.crawler (lưu text trafilatura vào SQLite):
- Lưu **HTML thô** + metadata vào `<out>/<nhóm>/part-*.parquet` (zstd), nên đổi bộ trích xuất không phải crawl lại.
- Mỗi nhóm (host, hoặc cả domain như 39.net) chỉ 1 nhịp: các request bắt đầu cách nhau >= `interval` giây,
  có tôn trọng robots.txt.
- Bị chặn (captcha / 403 / 429 / 503 / timeout liên tiếp) thì cả nhóm nghỉ, lùi theo cấp số nhân (30 s gấp đôi tới 30 phút).
  **Không** tìm cách vượt chặn: không giả lập TLS, không proxy, không trình duyệt.
- Resume: id đã có nhãn cuối (sets.FINAL) bị bỏ qua. Dừng êm bằng file `<out>/STOP` hoặc Ctrl+C.
"""
import asyncio
import json
import logging
import random
import time
from collections import Counter, deque
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from crawl.classify import HEADERS, classify, decode_html, visible_text_len
from crawl.sets import SHARED_DOMAINS

# host hay treo response: cho nhiều request đang chờ hơn, nhịp bắt đầu request vẫn như cũ
SLOW_HOSTS = {"www.familydoctor.com.cn": 1, "www.120ask.com": 2, "www.cnkang.com": 2}
# lỗi kiểu "server cắt / treo kết nối": thử lại ngay, không nghỉ dài
TRANSIENT = ("RemoteProtocolError", "ReadError", "ReadTimeout", "WriteError", "PoolTimeout")
SCHEMA = pa.schema([("id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()),
                    ("status", pa.int32()), ("label", pa.string()), ("detail", pa.string()),
                    ("ctype", pa.string()), ("ms", pa.int32()), ("fetched_at", pa.int64()),
                    ("html", pa.binary())])
log = logging.getLogger("crawl.polite")


class Shards:
    """Gom bản ghi theo nhóm, ghi ra parquet mỗi `every` dòng hoặc `secs` giây (ghi .tmp rồi đổi tên: mất điện
    không để lại shard dở). Mất tối đa ~`secs` giây fetch mỗi nhóm khi tắt ngang, lần chạy sau fetch lại."""

    def __init__(self, out: Path, every: int = 500, secs: int = 600):
        self.out, self.every, self.secs = out, every, secs
        self.buf, self.last = {}, {}

    def add(self, group: str, rec: dict) -> None:
        self.buf.setdefault(group, []).append(rec)
        if len(self.buf[group]) >= self.every or time.time() - self.last.get(group, time.time()) > self.secs:
            self.flush(group)
        self.last.setdefault(group, time.time())

    def flush(self, group: str) -> None:
        rows = self.buf.pop(group, [])
        self.last[group] = time.time()
        if not rows:
            return
        d = self.out / group.replace(":", "_")
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"part-{int(time.time() * 1000)}.parquet"
        tmp = f.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression="zstd", compression_level=6)
        tmp.replace(f)

    def flush_all(self) -> None:
        for g in list(self.buf):
            self.flush(g)


def group_interval(group: str, interval: float) -> float:
    return max(interval, SHARED_DOMAINS.get(group, interval))


def backoff_pause(strikes: int, base: float, cap: float) -> float:
    """Lần bị chặn thứ `strikes` (từ 1) -> số giây nghỉ cả nhóm."""
    return min(base * 2 ** (strikes - 1), cap)


def keep_awake() -> None:
    """Windows: xin hệ điều hành không tự ngủ khi process đang chạy (tự hết khi thoát, không đổi cài đặt).
    Hệ khác: không làm gì."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


async def _robots(client, url: str, cache: dict):
    s = urlsplit(url)
    if s.netloc not in cache:
        rp = robotparser.RobotFileParser()
        try:
            r = await client.get(f"{s.scheme}://{s.netloc}/robots.txt", timeout=20)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except Exception:
            rp.parse([])
        cache[s.netloc] = rp
    return cache[s.netloc]


async def _nap(secs: float, out: Path) -> None:
    """Ngủ, nhưng dậy trong 2 s nếu có file STOP (dừng không phải đợi hết một lần backoff dài)."""
    end = time.monotonic() + secs
    while (left := end - time.monotonic()) > 0:
        if (out / "STOP").exists():
            return
        await asyncio.sleep(min(2.0, left))


async def _run_group(group, rows, interval, inflight, client, shards, stats, opt, out, timeout):
    """`inflight` worker dùng chung 1 hàng đợi và 1 nhịp, nên các request vẫn bắt đầu cách nhau >= interval."""
    q, tries, rcache = deque(rows), Counter(), {}
    st = {"next_t": 0.0, "strikes": 0, "paused_until": 0.0, "neterr_run": 0}
    pace = asyncio.Lock()

    async def worker():
        while q:
            if (out / "STOP").exists():
                return
            r = q.popleft()
            rp = await _robots(client, r["url"], rcache)
            if not rp.can_fetch(HEADERS["User-Agent"], r["url"]):
                shards.add(group, dict(id=r["id"], url=r["url"], status=0, label="robots_disallowed",
                                       fetched_at=int(time.time())))
                stats[group]["robots"] += 1
                continue
            async with pace:
                now = time.monotonic()
                await _nap(max(0.0, st["next_t"] - now, st["paused_until"] - now), out)
                if (out / "STOP").exists():
                    q.appendleft(r)
                    return
                st["next_t"] = time.monotonic() + interval * random.uniform(0.9, 1.2)
            t0 = time.perf_counter()
            rec = dict(id=r["id"], url=r["url"], fetched_at=int(time.time()))
            # host đã redirect http -> đúng url https thì gọi thẳng https, bớt một vòng redirect
            url = "https://" + r["url"][7:] if st.get("https") and r["url"].startswith("http://") else r["url"]
            try:
                resp = await client.get(url, timeout=timeout)
                if not st.get("https") and r["url"].startswith("http://") and str(resp.url) == "https://" + r["url"][7:]:
                    st["https"] = True
                raw = resp.content
                html, _ = decode_html(raw, resp.headers.get("content-type", ""))
                label, detail = classify(resp.status_code, dict(resp.headers), html, visible_text_len(html),
                                         str(resp.url), r["url"])
                rec.update(final_url=str(resp.url), status=resp.status_code, label=label, detail=detail[:200],
                           ctype=resp.headers.get("content-type", ""), ms=round((time.perf_counter() - t0) * 1000),
                           html=raw if label == "ok" else None)
            except Exception as e:
                rec.update(status=-1, label=f"net_error:{type(e).__name__}", detail=str(e)[:200],
                           ms=round((time.perf_counter() - t0) * 1000))
            lab = rec["label"]
            # lỗi mạng liên tiếp = host quá tải hoặc đang âm thầm bóp băng thông: coi như bị chặn
            st["neterr_run"] = st["neterr_run"] + 1 if lab.startswith("net_error:") else 0
            blocked = lab.startswith("blocked:") or rec.get("status") in (429, 503) or st["neterr_run"] >= opt.stall_errors
            if blocked or lab.startswith("net_error:"):
                tries[r["id"]] += 1
                stats[group]["blocked" if blocked else "neterr"] += 1
                if blocked:
                    if st["paused_until"] <= time.monotonic():  # nhiều lỗi song song trong 1 đợt chỉ tính 1 lần
                        st["strikes"] += 1
                    pause = backoff_pause(st["strikes"], opt.backoff, opt.max_backoff)
                    st["paused_until"] = time.monotonic() + pause
                    stats[group]["paused_until"] = time.time() + pause
                elif lab.split(":")[1] in TRANSIENT:
                    pause = 0
                else:
                    pause = opt.neterr_pause
                if tries[r["id"]] < opt.retries:
                    q.append(r)  # xuống cuối hàng, một url lỗi không chặn cả nhóm
                else:
                    shards.add(group, rec)  # lưu nhưng không phải nhãn cuối -> lần chạy sau thử lại
                log.warning(f"{group}: {lab} on {r['url']} -> pause {pause:.0f}s (strike {st['strikes']})")
                await _nap(pause, out)
                continue
            st["strikes"] = 0
            shards.add(group, rec)
            stats[group]["ok" if lab == "ok" else "other"] += 1

    await asyncio.gather(*(worker() for _ in range(inflight)))
    shards.flush(group)
    stats[group]["finished"] = True


async def _progress(stats, total, out, t0):
    while True:
        await asyncio.sleep(60)
        keep_awake()
        done = sum(s["ok"] + s["other"] + s["robots"] for s in stats.values())
        ok = sum(s["ok"] for s in stats.values())
        rate = done / max(1, time.time() - t0) * 60
        paused = [g for g, s in stats.items() if s.get("paused_until", 0) > time.time()]
        log.info(f"done {done:,}/{total:,} (ok {ok:,}) {rate:,.0f}/min, paused {paused[:8]}")
        (out / "_progress.json").write_text(json.dumps(
            {"t": time.time(), "t0": t0, "done": done, "ok": ok, "todo": total, "per_min": rate, "groups": stats},
            default=str, indent=1))


async def crawl_groups(groups: dict[str, list[dict]], out: Path, opt) -> None:
    """groups: nhóm -> [{id, url}, ...] (chưa xong). opt: tham số của scripts/crawl_set.py."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "STOP").unlink(missing_ok=True)
    keep_awake()
    total = sum(len(r) for r in groups.values())
    log.info(f"todo {total:,} urls in {len(groups)} rate groups")
    stats = {g: Counter() for g in groups}
    shards = Shards(out)
    timeout = httpx.Timeout(connect=opt.connect_timeout, read=opt.read_timeout, write=15, pool=60)
    limits = httpx.Limits(max_connections=len(groups) * 2 + 10, max_keepalive_connections=len(groups) + 10)
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, http2=True, limits=limits,
                                 verify=not opt.insecure) as client:
        mon = asyncio.create_task(_progress(stats, total, out, time.time()))
        try:
            await asyncio.gather(*(_run_group(g, r, group_interval(g, opt.interval), SLOW_HOSTS.get(g, 1),
                                              client, shards, stats, opt, out, timeout)
                                   for g, r in groups.items()))
        finally:
            shards.flush_all()
            mon.cancel()
    log.info("stopped by STOP file" if (out / "STOP").exists() else "finished")
