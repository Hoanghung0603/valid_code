"""Bounded browser fallback probe (LOCAL PREP ONLY - NOT DEPLOYED).

User review corrections applied:
- text = extracted article text via scripts/extract.py (never raw HTML).
- status/final_url = actual CDP Network.responseReceived main-document
  response, correlated excluding subresources (never hardcoded 200).
- classify() via scripts/extract (block text markers + real HTTP);
  only `ok` records are written; blocked -> stop host immediately.
- 'xb' exclusive create (no overwrite); STOP/lock startup checks;
  periodic timeout, low priority, browser.stop() in finally.
- Archive provenance stays in separate audit; source id/url stay original;
  archive candidate URL preserved as evidence, never merged into id.

Limits: sequential, delay>=3s, max 9 pages, body cap 3MB, HEADLESS with
images/media disabled for bandwidth. Browser network path has no maxMbps
enforcement -> REFUSE production run without external traffic cap report.
nodriver pinned (see COVERAGE_RESEARCH.md); no install now (trung Chrome
absent, host affinity stays trung; b660 browser NOT deployed - IP change
would break machine affinity, user approved local prep only).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Pinned runtime dep for the future browser stage (NOT installed now).
NODRIVER_PIN = "nodriver==0.50.6"

MAX_PAGES_HARD = 9
BODY_CAP = 3 * 1024 * 1024

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract import classify, extract  # noqa: E402


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def parse_main_html(html: bytes, url: str) -> dict:
    """Extract article text; never persist raw HTML as corpus text."""
    return extract(html, url)


def pick_main_response(events: list, url: str):
    """Correlate main-document response; exclude subresources.

    events: list of dicts {url, status, type, frame_main(bool)}.
    Returns (status, final_url) of last main-frame document response,
    or (None, url) if none.
    """
    cur = (None, url)
    for e in events:
        if e.get("frame_main") and e.get("type") == "Document":
            cur = (e.get("status"), e.get("url") or url)
    return cur


async def _wire_cdp(page, store: list):
    """Best-effort CDP Network listener; returns detach callable or None."""
    try:
        sess = await page.get_cdp_session()  # nodriver API (pinned)
        await sess.send("Network.enable", {})
        async def _cb(evt: dict):
            p = (evt.get("params") or {})
            resp = p.get("response") or {}
            if p.get("type") == "Document" or p.get("frameId"):
                store.append({
                    "url": resp.get("url", ""),
                    "status": resp.get("status"),
                    "type": p.get("type", "Document"),
                    "frame_main": True,
                })
        await sess.on("Network.responseReceived", _cb)
        return sess
    except Exception:
        return None


async def run(urls: list, max_pages: int, delay: float, out: Path,
              crawl_root: Path | None = None, timeout_s: float = 120.0,
              browser_factory=None):
    if max_pages > MAX_PAGES_HARD:
        raise ValueError(f"max-pages {max_pages} exceeds hard bound {MAX_PAGES_HARD}")
    if delay < 3.0:
        raise ValueError("delay must be >= 3.0s")
    if crawl_root is None:
        crawl_root = Path(os.environ.get("R2AI_CRAWL_ROOT", r"D:\r2ai\data\crawl"))
    if (crawl_root / "STOP").exists():
        print("STOP present: refusing run", flush=True)
        return 3
    if (crawl_root / "heavy_job.lock").exists():
        print("heavy_job.lock present: refusing run", flush=True)
        return 3
    if browser_factory is None:
        try:
            import nodriver as nd  # type: ignore
            _ = nd
        except Exception as e:
            print(f"nodriver unavailable (pinned {NODRIVER_PIN}): {e}", flush=True)
            return 2
    rec_path = out.with_suffix(".records.jsonl")
    audit_path = out.with_suffix(".audit.jsonl")
    if rec_path.exists() or audit_path.exists():
        print("output collision: refusing overwrite", flush=True)
        return 4
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
        except Exception:
            pass
    else:
        try:
            os.nice(10)
        except Exception:
            pass
    browser = None
    n = 0
    try:
        if browser_factory is not None:
            browser = browser_factory()
        else:
            import nodriver as nd2
            browser = await nd2.start(
                headless=True,
                browser_args=["--blink-settings=imagesEnabled=false",
                              "--disable-features=MediaPreloading"],
            )
        rf = rec_path.open("x", encoding="utf-8")
        af = audit_path.open("x", encoding="utf-8")
        with rf, af:
            for item in urls[:max_pages]:
                t_start = time.time()
                url = item.get("url", "")
                _id = item.get("id")
                if not isinstance(_id, int) or not url:
                    af.write(json.dumps({"id": _id, "url": url, "ok": False,
                                         "error": "invalid-id-url"}) + "\n")
                    continue
                events: list = []
                detach = None
                try:
                    page = await asyncio.wait_for(browser.get(url), timeout=timeout_s)
                    detach = await _wire_cdp(page, events)
                    await asyncio.wait_for(page.sleep(2), timeout=timeout_s)
                    html = await asyncio.wait_for(page.get_content(), timeout=timeout_s)
                    raw = (html or "").encode("utf-8", "ignore")[:BODY_CAP]
                    status, final_url = pick_main_response(events, url)
                    if status is None:
                        # No documented main-response log: do NOT fake 200.
                        status, final_url = None, url
                    ex = parse_main_html(raw, final_url or url)
                    cls = classify(status, final_url or "", ex.get("text", ""))
                    nbytes = len(raw)
                    audit = {"id": _id, "url": url, "final_url": final_url,
                             "status": status, "class": cls, "nbytes": nbytes,
                             "title": (ex.get("title") or "")[:80],
                             "text_len": len(ex.get("text", "")),
                             "elapsed": round(time.time() - t_start, 2),
                             "archive_provenance": item.get("archive_provenance")}
                    if cls == "blocked":
                        audit["ok"] = False
                        af.write(json.dumps(audit, ensure_ascii=False) + "\n")
                        af.flush()
                        print(f"blocked -> stop host {host_of(url)}", flush=True)
                        break
                    if cls != "ok" or status != 200:
                        audit["ok"] = False
                        af.write(json.dumps(audit, ensure_ascii=False) + "\n")
                        af.flush()
                    else:
                        rf.write(json.dumps({
                            "id": _id, "url": url, "host": host_of(url),
                            "final_url": final_url, "status": status,
                            "title": ex.get("title", ""), "date": ex.get("date", ""),
                            "text": ex.get("text", ""), "ts": int(time.time()),
                        }, ensure_ascii=False) + "\n")
                        audit["ok"] = True
                        af.write(json.dumps(audit, ensure_ascii=False) + "\n")
                        af.flush()
                        n += 1
                except Exception as e:
                    af.write(json.dumps({"id": _id, "url": url, "ok": False,
                                         "error": repr(e)[:200]}) + "\n")
                    af.flush()
                    break
                if time.time() - t_start > timeout_s:
                    af.write(json.dumps({"id": _id, "url": url, "ok": False,
                                         "error": "page-timeout"}) + "\n")
                    break
                await asyncio.sleep(max(0.0, delay))
    finally:
        try:
            if browser is not None:
                res = browser.stop()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass
    print(f"pages={n} records={rec_path} audit={audit_path}", flush=True)
    return 0


def refuse_production_without_traffic_cap():
    raise SystemExit("REFUSE production: browser network path has no maxMbps "
                     "enforcement; needs external traffic-cap approved report.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--max-pages", type=int, default=9)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--production", action="store_true")
    a = ap.parse_args(argv)
    if a.production:
        refuse_production_without_traffic_cap()
    items = [json.loads(x) for x in Path(a.urls).read_text(encoding="utf-8").splitlines() if x.strip()]
    return asyncio.run(run(items, a.max_pages, a.delay, Path(a.out)))


if __name__ == "__main__":
    sys.exit(main())
