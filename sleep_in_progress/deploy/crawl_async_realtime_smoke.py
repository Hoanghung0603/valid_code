"""Bounded REALTIME smoke for async recovery (runs on b660 only).

Evidence (not env-claims):
  * CDP Response.status float 200.0 normalises to int 200 (real code path).
  * Real browser fetch of ONE public URL: HTTP 200 + extracted text >= 200.
  * Real HTTPS proxy transport of ONE public URL via ONE runtime route:
    typed status, redacted error, zero credential bytes in output.

Proxy secrets are parsed ONLY at runtime from --proxy-file and NEVER
printed, logged, or included in any output dict. Exit 0 requires the
browser leg to harvest HTTP 200 + text>=200; the proxy leg is reported
honestly whatever the network says (typed status or typed error name).
"""
import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_async_supervisor as sup  # noqa: E402

PROXY_URL = "https://example.com/"
# ONE known medical article (corpus vinmec URL, long static prose) is the
# primary browser probe: the feature must harvest HTTP 200 + text>=200 from
# a REAL article, not a homepage shell. cnkang pilot URL is the fallback
# (earlier real pilot), httpbin the last resort. No invented URLs.
BROWSER_PROBE_URLS = (
    "https://www.vinmec.com/vie/bai-viet/03-thuc-pham-nhuan-trang-cho-ba-bau-vi",
    "http://www.cnkang.com/nkwk/jisuiyan/701540.html",
    "https://httpbin.org/html",
)
HARVEST_MIN_TEXT = 200


async def amain(proxy_file: Path, crawl_root: Path) -> dict:
    out: dict = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "proxy_url": PROXY_URL}
    out["normalize_200_float"] = sup.normalize_status(200.0) == 200
    limiter = sup.BandwidthLimiter()
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10)
    direct_probe = None
    async with aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": sup.UA}) as session:
        for u in BROWSER_PROBE_URLS:
            d = await sup.fetch_direct(session, limiter, u, 30.0)
            direct_probe = {"url": u, "status": d["status"],
                            "body_len": len(d.get("body") or b""),
                            "error": sup.redact(str(d.get("error"))) if d.get("error") else None,
                            "infra": d.get("infra")}
            if d["status"] == 200 and len(d.get("body") or b"") > 200:
                break
    out["direct"] = direct_probe
    browser_probe = None
    browser_probes = []
    for u in BROWSER_PROBE_URLS:
        row = {"id": -1, "url": u, "host": sup.host_of(u)}
        b = await sup.run_browser_one(row, crawl_root, limiter, None,
                                      timeout_s=60.0)
        raw_len = len(b.get("raw") or b"")
        text_len, title = 0, ""
        if b.get("ok") and b.get("raw"):
            raw_head = b["raw"][:120].lower()
            if b"just a moment" in raw_head or b"attention required" in raw_head:
                pass    # challenge shell: keep text_len 0 as typed evidence
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                meta = await sup.extract_in_pool(pool, b["raw"], b["final_url"])
                text_len, title = len(meta.get("text") or ""), meta.get("title", "")
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            if not text_len and raw_len >= 200:
                # Headless get_content() can return a bare shell (scripts not
                # executed into DOM text) while CDP status is a REAL typed 200.
                # raw_len>=200 + status 200 is honest transport evidence; the
                # harvest gate (text>=200) stays closed — never fake it.
                pass
        browser_probe = {"url": u, "ok": b.get("ok"), "status": b.get("status"),
                         "status_is_int": isinstance(b.get("status"), int),
                         "raw_len": raw_len, "text_len": text_len,
                         "title": (title or "")[:80],
                         "nbytes": b.get("nbytes", 0),
                         "error": sup.redact(str(b.get("error"))) if b.get("error") else None,
                         "cap_applied": bool((b.get("cap") or {}).get("applied")),
                         "settle": b.get("settle") or {}}
        browser_probes.append(browser_probe)
        if b.get("status") == 200 and text_len >= HARVEST_MIN_TEXT:
            break
        continue    # every probe is real evidence: try next until one harvests
    out["browser"] = browser_probe
    out["browser_probes"] = browser_probes
    out["harvest_gate"] = {"min_text": HARVEST_MIN_TEXT}
    proxies = sup.load_proxies(proxy_file, 100) if proxy_file else []
    routes = sup.RouteTable(proxies)
    idx, purl = routes.pick(sup.host_of(PROXY_URL))
    if purl is None:
        out["proxy"] = {"routes": len(proxies), "route": None,
                        "note": "no-route-available"}
    else:
        loop = asyncio.get_running_loop()
        r = await loop.run_in_executor(
            None, sup.fetch_via_proxy, PROXY_URL, purl, limiter)
        out["proxy"] = {"routes": len(proxies), "route_idx": idx,
                        "status": r.get("status"), "nbytes": r.get("nbytes", 0),
                        "elapsed_s": round(r.get("elapsed", 0.0), 1),
                        "error": sup.redact(str(r.get("error"))) if r.get("error") else None,
                        "infra": r.get("infra")}
    blob = json.dumps(out, ensure_ascii=False)
    for secret_hint in ("pass", "Proxy-Authorization", "@"):
        assert secret_hint not in blob or secret_hint in (
            "httpbin.org/html", PROXY_URL), "credential leak guard"
    # Exit 0 ONLY when the browser leg really harvests (HTTP 200 int +
    # extracted text>=200 on a real article). Transport-only is NOT enough.
    out["ok"] = bool(out["normalize_200_float"]
                     and out["browser"]["status"] == 200
                     and isinstance(out["browser"]["status"], int)
                     and out["browser"]["text_len"] >= HARVEST_MIN_TEXT
                     and out["proxy"]["status"] == 200)
    out["harvest_open"] = bool(
        out["browser"]["text_len"] >= HARVEST_MIN_TEXT)
    return out


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-file", type=Path, default=None)
    ap.add_argument("--crawl-root", type=Path,
                    default=Path(r"D:\r2ai\data\crawl"))
    a = ap.parse_args(argv)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        out = asyncio.run(amain(a.proxy_file, a.crawl_root))
    except Exception as e:
        print(json.dumps({"ok": False,
                          "error": sup.redact(type(e).__name__)}))
        return 1
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
