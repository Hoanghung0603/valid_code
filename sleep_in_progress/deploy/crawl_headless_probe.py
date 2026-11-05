"""Bounded nodriver headless vs headed probe. No HTML saved. No crawl writes.

One isolated Chrome per URL. Stops on a captcha page (records it, no solve).
Does not touch the user's Chrome or the running supervisor.
"""
import argparse
import asyncio
import io
import json
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
    except Exception:
        pass
from pathlib import Path
from urllib.parse import urlparse

CHALLENGE_URL = (
    "verify.html", "/captcha", "/challenge", "cdn-cgi", "waf",
    "security-check", "chk_js", "geetest", "verifycode",
)
CHALLENGE_TEXT = (
    "just a moment", "attention required", "checking your browser",
    "enable javascript", "cf-browser-verification", "cf_chl",
    "challenge-platform", "turnstile", "hcaptcha", "g-recaptcha",
    "recaptcha/api", "px-captcha", "geetest", "verify you are human",
    "are you a robot", "robot check", "ddos protection",
    "滑动拼图验证", "安全验证", "访问验证", "人机验证", "请开启javascript",
    "正在验证您的浏览器", "请输入验证码", "验证码", "captcha",
)


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def challenge_hit(final_url: str, title: str, text: str) -> str:
    blob = f"{final_url}\n{title}\n{text}".lower()
    for m in CHALLENGE_URL:
        if m in (final_url or "").lower():
            return m
    for m in CHALLENGE_TEXT:
        if m in blob:
            return m
    return ""


async def probe_one(url: str, headless: bool, crawl_root: Path,
                    chrome_exe: str, timeout_s: float) -> dict:
    t0 = time.time()
    browser = None
    page = None
    out = {"url": url, "host": host_of(url), "headless": headless,
           "http": None, "final_host": "", "title": "", "text_len": 0,
           "challenge": "", "error": "", "seconds": 0.0}
    try:
        sys.path.insert(0, str(crawl_root / "tools" / "browser_deps"))
        import nodriver as nd
        kw = {"headless": headless,
              "browser_args": ["--blink-settings=imagesEnabled=false",
                               "--disable-features=Translate"]}
        if chrome_exe:
            kw["browser_executable_path"] = chrome_exe
        browser = await asyncio.wait_for(nd.start(**kw), timeout=timeout_s)
        page = await asyncio.wait_for(browser.get("about:blank"), timeout=timeout_s)
        await page.send(nd.cdp.network.enable())
        docs = []

        def _resp(event, _tab=None):
            try:
                resp = getattr(event, "response", event)
                rurl = getattr(resp, "url", "") or ""
                st = getattr(resp, "status", None)
                t = getattr(event, "type_", None)
                rtype = str(getattr(t, "value", t) or "")
                if rtype == "Document":
                    docs.append((int(st) if st is not None else None, rurl))
            except Exception:
                pass

        page.add_handler(nd.cdp.network.ResponseReceived, _resp)
        await asyncio.sleep(0.3)
        await asyncio.wait_for(page.get(url), timeout=timeout_s)
        deadline = time.monotonic() + min(25.0, timeout_s)
        title = ""
        text = ""
        href = ""
        while time.monotonic() < deadline:
            try:
                href = str(await asyncio.wait_for(
                    page.evaluate("document.location.href"), timeout=8) or "")
            except Exception:
                href = ""
            try:
                title = str(await asyncio.wait_for(
                    page.evaluate("document.title"), timeout=8) or "")
            except Exception:
                title = ""
            try:
                text = str(await asyncio.wait_for(
                    page.evaluate(
                        "(document.body?document.body.innerText:'').slice(0,4000)"),
                    timeout=8) or "")
            except Exception:
                text = ""
            if challenge_hit(href, title, text):
                break
            if len(text) >= 200:
                break
            await asyncio.sleep(1.0)
        status = None
        for st, rurl in reversed(docs):
            if href and rurl and href.split("?")[0] == rurl.split("?")[0]:
                status = st
                break
        if status is None and docs:
            status = docs[-1][0]
        hit = challenge_hit(href, title, text)
        out.update({
            "http": status,
            "final_host": host_of(href) if href.startswith("http") else "",
            "title": title[:120],
            "text_len": len(text),
            "challenge": hit,
            "docs": len(docs),
        })
        if hit:
            out["error"] = "captcha-stop"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:180]
    finally:
        out["seconds"] = round(time.time() - t0, 1)
        try:
            if browser is not None:
                res = browser.stop()
                if asyncio.iscoroutine(res):
                    await asyncio.wait_for(res, timeout=8)
        except Exception:
            pass
    return out


async def amain(urls, crawl_root, chrome_exe, timeout_s, delay, out_path,
                modes=(True, False)):
    results = []
    for url in urls:
        for headless in modes:
            row = await probe_one(url, headless, crawl_root, chrome_exe, timeout_s)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            await asyncio.sleep(delay)
        await asyncio.sleep(delay)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8")
    print("PROBE_DONE " + json.dumps(
        {"urls": len(urls), "rows": len(results)}, ensure_ascii=True),
        flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crawl-root", default=r"D:\r2ai\data\crawl")
    ap.add_argument("--chrome-exe",
                    default=r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--headed-only", action="store_true")
    a = ap.parse_args(argv)
    urls = []
    for line in Path(a.urls).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            urls.append(json.loads(line)["url"])
        else:
            urls.append(line)
    if len(urls) > 40:
        raise SystemExit("cap 40 urls")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    modes = (False,) if a.headed_only else (True, False)
    asyncio.run(amain(urls, Path(a.crawl_root), a.chrome_exe,
                       a.timeout, a.delay, Path(a.out), modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
