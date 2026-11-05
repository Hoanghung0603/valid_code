"""Thu các domain chặn client script bằng Chrome thật (không headless), qua CDP.

Vì sao cần: `nhathuoclongchau.com.vn` (và `tamanhhospital.vn`) trả 403 + trang challenge cho
httpx, cho httpx giả header Chrome, và cho cả Chrome headless (kiểm 20/10/2026 — xem
`reviews/2026-10-20_longchau_access.md`). Cùng IP, cùng máy, Chrome có cửa sổ thì trang thật
hiện bình thường. Script này điều khiển đúng cửa sổ đó, mở **một tab**, mỗi lần một trang.

Một trang mỗi 2,5 giây mặc định (`--rate 0.4`): 79.598 trang ≈ 55 giờ, nên chạy qua đêm.
Chrome giữ cookie phiên nên không phải qua lại bước xác minh thường xuyên; nếu có challenge,
script dừng lại chờ người bấm xác minh rồi tự chạy tiếp, và nghỉ 10/30/60 phút nếu bị chặn.

Định dạng đầu ra giống `crawl/crawl_home.py` để hợp nhất cùng một cách:
    <out>/html/browser_shard_NNNNN.jsonl.zst
        id, final_url, http_code, content_type, encoding, body_b64
    <out>/status.csv
        cột của crawl_status + shard_file; một id có thể có nhiều dòng, lấy dòng cuối
    <out>/crawl.log

Chạy tiếp được: status chỉ ghi sau khi shard đóng; mất điện thì mất tối đa shard đang mở và
các URL đó được thu lại ở lần sau. Tạo <out>/STOP để dừng sạch.

Dùng:
    python crawl/crawl_browser.py --corpus D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet \
        --status-db status.duckdb --out D:/r2ai_data/crawl_browser \
        --rate nhathuoclongchau.com.vn=0.4 --limit 5
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
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import websocket
import zstandard as zstd
from charset_normalizer import from_bytes

STATUS_COLS = ["id", "url", "final_url", "lang_expected", "status", "http_code", "content_type", "n_bytes",
               "attempts", "fetched_at", "shard", "sha1_raw", "error_msg", "shard_file"]
CHALLENGE_MARKERS = ("challenge-platform", "cf-chl-", "just a moment", "checking your browser",
                     "verify you are human", "xác minh", "captcha")
PAUSES = [600, 1800, 3600]
SHARD_RECORDS = 2000
CHARSET_RE = re.compile(r"charset=[\"']?([\w-]+)", re.I)

log = logging.getLogger("crawl_browser")


class CdpTimeout(RuntimeError):
    """Kênh CDP treo/đứt: bỏ kết nối và thử lại trang đó."""


# --------------------------------------------------------------------------- output

def detect_encoding(raw: bytes, content_type: str) -> str:
    for src in (content_type or "", raw[:4096].decode("ascii", errors="replace")):
        m = CHARSET_RE.search(src)
        if m:
            enc = m.group(1).lower()
            return "gb18030" if enc in ("gb2312", "gbk", "x-gbk") else enc
    best = from_bytes(raw[:32768]).best()
    return best.encoding if best else "utf-8"


class Output:
    """Ghi shard .jsonl.zst; status của shard chỉ ghi khi shard đã đóng."""

    def __init__(self, out: Path):
        self.dir = out / "html"
        self.dir.mkdir(parents=True, exist_ok=True)
        for tmp in self.dir.glob("*.tmp"):
            tmp.unlink()  # shard của lần chạy bị crash: URL của nó chưa có status nên sẽ chạy lại
        done = [int(p.name.split("_")[2].split(".")[0]) for p in self.dir.glob("browser_shard_*.jsonl.zst")]
        self.idx = max(done, default=-1) + 1
        self.status_path = out / "status.csv"
        new = not self.status_path.exists()
        cut = not new and self.status_path.stat().st_size and not self.status_path.read_bytes().endswith(b"\n")
        self.status_f = open(self.status_path, "a", newline="", encoding="utf-8")
        if cut:
            self.status_f.write("\n")
        self.status = csv.DictWriter(self.status_f, fieldnames=STATUS_COLS)
        if new:
            self.status.writeheader()
        self.cctx = zstd.ZstdCompressor(level=3)
        self._open()

    def _open(self):
        self.name = f"browser_shard_{self.idx:05d}.jsonl.zst"
        self.fh = open(self.dir / (self.name + ".tmp"), "wb")
        self.writer = self.cctx.stream_writer(self.fh)
        self.pending: list[dict] = []

    def _close(self):
        self.writer.close()
        if self.pending:
            os.replace(self.dir / (self.name + ".tmp"), self.dir / self.name)
            for row in self.pending:
                self.status.writerow(row)
            self.status_f.flush()
            log.info(f"shard {self.name}: {len(self.pending)} trang")
            self.idx += 1
        else:
            (self.dir / (self.name + ".tmp")).unlink()

    def page(self, rec: dict, row: dict):
        self.writer.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        row["shard_file"] = self.name
        self.pending.append(row)
        if len(self.pending) >= SHARD_RECORDS:
            self._close()
            self._open()

    def row(self, row: dict):
        self.status.writerow(row)
        self.status_f.flush()

    def close(self):
        self._close()
        self.status_f.close()


# --------------------------------------------------------------------------- chrome

class Chrome:
    """Chrome thật + một tab dùng lại; đọc body gốc qua Network.getResponseBody."""

    def __init__(self, chrome: str, port: int, profile: Path, challenge_wait: float):
        self.port = port
        self.profile = profile
        self.challenge_wait = challenge_wait
        self.proc = None
        self.ws = None
        self.tab = None
        self._mid = 0
        self.doc: dict = {}
        self._launch(chrome)

    # -- lifecycle
    def _launch(self, chrome: str):
        if self._alive():
            log.info("dùng Chrome đang mở ở cổng %s", self.port)
            return
        exe = chrome or self._find_chrome()
        if not exe or not Path(exe).exists():
            raise SystemExit("Không tìm thấy chrome.exe; truyền --chrome")
        self.profile.mkdir(parents=True, exist_ok=True)
        log.info(f"mở Chrome: {exe} (profile {self.profile})")
        self.proc = __import__("subprocess").Popen([
            exe, f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile}",
            "--remote-allow-origins=*", "--no-first-run", "--no-default-browser-check",
            "--window-size=1280,900", "about:blank",
        ])
        for _ in range(80):
            if self._alive():
                break
            time.sleep(0.25)
        else:
            raise SystemExit("Chrome không mở cổng debug")

    @staticmethod
    def _find_chrome() -> str:
        for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                  "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
            if Path(p).exists():
                return p
        return ""

    def _alive(self) -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/version", timeout=1).read()
            return True
        except Exception:
            return False

    def connect(self):
        if self.ws is not None:
            return
        if self.tab:
            try:  # nối lại đúng tab cũ để giữ cookie/trạng thái
                self.ws = websocket.create_connection(self.tab["webSocketDebuggerUrl"], timeout=120,
                                                      suppress_origin=True)
            except Exception:
                self.ws = None
        if self.ws is None:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/json/new?about:blank", method="PUT")
            self.tab = json.load(urllib.request.urlopen(req))
            self.ws = websocket.create_connection(self.tab["webSocketDebuggerUrl"], timeout=120,
                                                  suppress_origin=True)
        self.cmd("Page.enable")
        # Bắt đúng response của document để lấy body gốc + status + header.
        # Network.getResponseBody không giữ body của document chính trên Chrome 155.
        self.cmd("Fetch.enable", patterns=[{"urlPattern": "*", "resourceType": "Document",
                                            "requestStage": "Response"}])
        # Chỉ giữ document của frame chính: iframe quảng cáo/reCAPTCHA cũng là "Document"
        # và nếu lẫn vào thì body 2KB của iframe sẽ thay mất body bài viết.
        self.frame_id = self.cmd("Page.getFrameTree")["frameTree"]["frame"]["id"]
        # Không dùng cache: response 304 không có body, getResponseBody sẽ treo.
        self.cmd("Network.setCacheDisabled", cacheDisabled=True)
        self.cmd("Network.enable")

    def close(self):
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()

    # -- CDP
    def cmd(self, method: str, **params):
        if self.ws is None:
            self.connect()
        self._mid += 1
        mid = self._mid
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            try:
                msg = json.loads(self.ws.recv())
            except Exception as e:
                # Kênh CDP đứt/treo: bỏ kết nối để lần sau mở lại sạch sẽ.
                self._reset_ws()
                raise CdpTimeout(f"{method}: {e}") from e
            self._on_event(msg)
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def _reset_ws(self):
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        self.ws = None
        self._mid = 0

    def _on_event(self, msg: dict):
        """Fetch.requestPaused của Document: lấy body+status rồi cho trang chạy tiếp."""
        if msg.get("method") != "Fetch.requestPaused":
            return
        p = msg["params"]
        if p.get("resourceType") != "Document" or p.get("frameId") != self.frame_id:
            self.cmd("Fetch.continueRequest", requestId=p["requestId"])
            return
        headers = {str(h.get("name", "")).lower(): str(h.get("value", "")) for h in p.get("responseHeaders", [])}
        raw = None
        try:
            r = self.cmd("Fetch.getResponseBody", requestId=p["requestId"])
            raw = base64.b64decode(r["body"]) if r.get("base64Encoded") else r.get("body", "").encode()
        except Exception as e:
            log.warning(f"getResponseBody lỗi ({e})")
        self.doc = {"status": p.get("responseStatusCode"), "headers": headers, "body": raw}
        self.cmd("Fetch.continueRequest", requestId=p["requestId"])

    def _recv(self, timeout: float):
        self.ws.settimeout(timeout)
        try:
            return json.loads(self.ws.recv())
        except Exception:
            return None
        finally:
            self.ws.settimeout(120)

    def evaluate(self, expr: str):
        r = self.cmd("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")

    # -- one page
    def fetch(self, url: str, stop: threading.Event) -> dict:
        """Mở URL, chờ load (và chờ người xác minh nếu gặp challenge). Trả dict để ghi status."""
        self.doc = {}
        self.loads = 0
        # Dọn request còn treo từ trang trước: mỗi request bị Fetch giữ phải được continue,
        # nếu không tab sẽ đứng ở lần điều hướng sau.
        while True:
            stale = self._recv(0.05)
            if stale is None:
                break
            try:
                self._on_event(stale)
            except CdpTimeout:
                break
        self.cmd("Page.navigate", url=url)
        deadline = time.time() + 60
        challenge_deadline = None
        last_load = 0.0
        while not stop.is_set():
            now = time.time()
            msg = self._recv(0.4)
            if msg:
                if msg.get("method") == "Page.loadEventFired":
                    self.loads += 1
                    last_load = now
                self._on_event(msg)
            body = self.doc.get("body")
            if body is not None and self.doc.get("status") == 200 and not self._is_challenge(body):
                # Đã có HTML gốc của server: không cần chờ hydrate/ảnh, sang URL kế tiếp luôn.
                break
            elif body is not None and challenge_deadline is None and self.loads:
                challenge_deadline = now + self.challenge_wait
                log.warning(f"challenge tại {url} — chờ người xác minh trong {int(self.challenge_wait)}s")
            if now > (challenge_deadline or deadline):
                break
        if not self.doc:
            self.doc = {"status": None, "headers": {}, "body": None}
        final_url = self.evaluate("location.href") or url
        body = self.doc.get("body")
        if body is None:
            try:
                html = self.evaluate("document.documentElement.outerHTML")
                body = ("<!DOCTYPE html>\n" + (html or "")).encode("utf-8")
            except Exception:
                body = None
        self.doc["final_url"] = final_url
        self.doc["challenge"] = self._is_challenge(body or b"")
        return self.doc

    @staticmethod
    def _is_challenge(body: bytes) -> bool:
        """Trang challenge của Cloudflare ~6KB; bài viết thật của Long Châu ~200-300KB."""
        if not body or len(body) > 30_000:
            return False
        head = body[:30_000].decode("utf-8", errors="ignore").lower()
        return any(m in head for m in CHALLENGE_MARKERS)


# --------------------------------------------------------------------------- domain pacing

class Domain:
    def __init__(self, name: str, rate: float):
        self.name, self.rate = name, max(0.05, rate)
        self.lock = threading.Lock()
        self.next_at = time.monotonic()
        self.pauses = 0

    def wait_turn(self, stop: threading.Event) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = now + wait + random.uniform(0.8, 1.2) / self.rate
        if wait > 0:
            stop.wait(wait)

    def block(self, stop: threading.Event) -> None:
        pause = PAUSES[min(self.pauses, len(PAUSES) - 1)]
        self.pauses += 1
        log.warning(f"{self.name}: bị chặn, nghỉ {pause // 60} phút rồi thử lại")
        stop.wait(pause)


# --------------------------------------------------------------------------- queue

def load_queue(corpus: str, domains: list[str], status_db: str | None, out: Path) -> dict[str, list]:
    import pyarrow.parquet as pq
    rows = pq.read_table(corpus, columns=["id", "url"]).to_pylist()
    host = lambda u: (urllib.parse.urlparse(u).hostname or "").lower().removeprefix("www.")  # noqa: E731
    skip: set[int] = set()
    if status_db:
        import duckdb
        con = duckdb.connect(status_db, read_only=True)
        try:
            skip |= {r[0] for r in con.execute(
                "SELECT id FROM crawl_status WHERE status IN ('ok', 'non_html')").fetchall()}
        finally:
            con.close()
    status_path = out / "status.csv"
    if status_path.exists():
        last: dict[int, dict] = {}
        with open(status_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("id") or "").isdigit() and r.get("status"):
                    last[int(r["id"])] = r
        skip |= {i for i, r in last.items()
                 if r["status"] in ("ok", "non_html", "empty", "http_error", "blocked")}
    queues = {d: [] for d in domains}
    for r in rows:
        h = host(r["url"])
        if h in queues and r["id"] not in skip:
            queues[h].append({"id": r["id"], "url": r["url"]})
    return queues


# --------------------------------------------------------------------------- main

def crawl(dom: Domain, queue: list, chrome: Chrome, out: Output, stop: threading.Event) -> None:
    while queue and not stop.is_set():
        item = queue.pop(0)
        attempts = 0
        cdp_retries = 0
        while True:
            attempts += 1
            dom.wait_turn(stop)
            if stop.is_set():
                queue.insert(0, item)
                return
            try:
                state = chrome.fetch(item["url"], stop)
            except CdpTimeout as e:
                cdp_retries += 1
                log.warning(f"CDP treo ({e}); nối lại và thử lại {item['url']}")
                attempts -= 1  # chưa gửi được request nào cho URL này
                if stop.is_set():
                    queue.insert(0, item)
                    return
                if cdp_retries > 3:
                    log.error("CDP treo 4 lần liên tiếp, dừng để kiểm tra Chrome")
                    stop.set()
                    queue.insert(0, item)
                    return
                time.sleep(5)
                continue
            cdp_retries = 0
            if state["challenge"]:
                if attempts > len(PAUSES):
                    out.row({"id": item["id"], "url": item["url"], "final_url": state["final_url"],
                             "lang_expected": "vi", "status": "blocked", "http_code": state["status"],
                             "content_type": state["headers"].get("content-type", ""),
                             "n_bytes": len(state["body"] or b""), "attempts": attempts,
                             "fetched_at": datetime.now(timezone.utc).isoformat(), "shard": None,
                             "sha1_raw": None, "error_msg": "challenge sau khi đã nghỉ đủ", "shard_file": None})
                    log.error(f"{dom.name}: vẫn challenge sau {attempts} lần thử, dừng domain")
                    stop.set()
                    return
                dom.block(stop)
                continue
            break
        raw = state["body"]
        row = {"id": item["id"], "url": item["url"], "final_url": state["final_url"] or item["url"],
               "lang_expected": "vi", "attempts": attempts, "fetched_at": datetime.now(timezone.utc).isoformat(),
               "shard": None, "sha1_raw": None, "error_msg": None, "http_code": state["status"],
               "content_type": state["headers"].get("content-type", ""),
               "n_bytes": len(raw or b""), "status": "http_error"}
        if raw is None:
            row["error_msg"] = "không lấy được body"
            out.row(row)
            continue
        ct = (row["content_type"] or "").lower()
        html = "text/html" in ct or "application/xhtml" in ct or ct == ""
        if row["http_code"] == 200 and html and len(raw) >= 100:
            row.update(status="ok", sha1_raw=hashlib.sha1(raw).hexdigest())
            rec = {"id": item["id"], "final_url": row["final_url"],
                   "http_code": 200, "content_type": row["content_type"],
                   "encoding": detect_encoding(raw, row["content_type"]),
                   "body_b64": base64.b64encode(raw).decode("ascii")}
            out.page(rec, row)
            dom.pauses = 0
        elif row["http_code"] == 200 and not html:
            row.update(status="non_html", sha1_raw=hashlib.sha1(raw).hexdigest())
            out.row(row)
        elif row["http_code"] == 200:
            row.update(status="empty", sha1_raw=hashlib.sha1(raw).hexdigest())
            out.row(row)
        else:
            row["error_msg"] = f"HTTP {row['http_code']}"
            out.row(row)


def main():
    ap = argparse.ArgumentParser(description="Thu domain chặn script bằng Chrome thật (CDP)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--status-db")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rate", nargs="+", default=["nhathuoclongchau.com.vn=0.4", "tamanhhospital.vn=0.4"],
                    help="domain=trang mỗi giây")
    ap.add_argument("--limit", type=int, default=0, help="số URL mỗi domain, để thử")
    ap.add_argument("--chrome", default="")
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--profile", type=Path,
                    default=Path(os.environ.get("LOCALAPPDATA", ".")) / "r2ai_chrome_profile")
    ap.add_argument("--challenge-wait", type=float, default=180.0,
                    help="số giây chờ người bấm xác minh khi gặp challenge")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "STOP").unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.out / "crawl.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    logging.getLogger("websocket").setLevel(logging.WARNING)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # máy không ngủ khi chạy đêm

    rates = {d: float(r) for d, r in (x.split("=") for x in args.rate)}
    if any(d.lower().removeprefix('www.')=='nhathuoclongchau.com.vn' for d in rates):
        ap.error('Long Chau is odd-ID-only. Use scripts/run_local.ps1 longchau-odd; whole-domain browser/home crawl is disabled.')
    queues = load_queue(args.corpus, list(rates), args.status_db, args.out)
    if args.limit:
        queues = {d: q[:args.limit] for d, q in queues.items()}
    log.info("hàng đợi: " + ", ".join(f"{d} {len(q)}" for d, q in queues.items()))
    if not any(queues.values()):
        log.info("không còn URL cần thu")
        return

    chrome = Chrome(args.chrome, args.port, args.profile, args.challenge_wait)
    out = Output(args.out)
    stop = threading.Event()
    stop_file = args.out / "STOP"

    def watch_stop():
        while not stop.is_set():
            if stop_file.exists():
                log.warning("tìm thấy file STOP: dừng mềm, shard đang mở vẫn được ghi")
                stop.set()
                return
            time.sleep(2)

    threading.Thread(target=watch_stop, daemon=True).start()
    total = sum(len(q) for q in queues.values())
    t0 = time.time()
    try:
        chrome.connect()
        for d, queue in queues.items():
            crawl(Domain(d, rates[d]), queue, chrome, out, stop)
            if stop.is_set():
                break
            log.info(f"{d}: còn {len(queue)} URL trong hàng đợi")
    except KeyboardInterrupt:
        log.info("Ctrl-C: đóng shard rồi dừng")
        stop.set()
    finally:
        out.close()
        chrome.close()
        log.info(f"đã dừng sau {(time.time() - t0) / 60:.1f} phút; tổng URL trong lượt = {total}")


if __name__ == "__main__":
    main()
