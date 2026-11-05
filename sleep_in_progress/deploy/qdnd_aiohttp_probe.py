import asyncio
import sys

import aiohttp

URL = ("https://www.qdnd.vn/chinh-tri/tin-tuc/"
       "quy-dinh-chat-che-tranh-lam-dung-day-nguoi-benh-di-mua-thuoc-750509")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


async def run(trust_env):
    trace = aiohttp.TraceConfig()
    seen = []

    async def on_redirect(session, ctx, params):
        st = getattr(getattr(params, "response", None), "status", None)
        seen.append((st, str(params.url), str(params.headers.get("Location"))))

    trace.on_request_redirect.append(on_redirect)
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(
            headers={"User-Agent": UA, "Accept-Language": "vi,zh-CN;q=0.9,en;q=0.8"},
            timeout=timeout, trace_configs=[trace], trust_env=trust_env) as s:
        try:
            async with s.get(URL, allow_redirects=True, max_redirects=5) as r:
                body = await r.read()
                print("trust_env=%s status=%s final=%s len=%d" % (
                    trust_env, r.status, str(r.url), len(body)))
        except Exception as e:
            print("trust_env=%s EXC %s: %s" % (trust_env, type(e).__name__, str(e)[:200]))
        print("redirects=%d" % len(seen))
        for st, u, loc in seen[:8]:
            print("  %s %s -> %s" % (st, u, loc))


async def main():
    await run(False)
    await run(True)


asyncio.run(main())
