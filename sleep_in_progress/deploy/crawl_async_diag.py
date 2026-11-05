"""One-shot import/proxy diagnostic for b660 async supervisor (read-only net to local proxy port 9)."""
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("py", sys.version.replace("\n", " "))
print("exe", sys.executable)
print("path0", sys.path[0])
for mod in ("urllib.parse", "urllib.request", "re", "zstandard",
            "trafilatura", "aiohttp", "requests", "nodriver"):
    try:
        __import__(mod)
        print("OK", mod)
    except BaseException:
        print("FAIL", mod)
        traceback.print_exc()
try:
    import urllib.parse
    import urllib.request
    p = urllib.parse.urlparse("http://user:pass@localhost:9")
    print("parsed", p.hostname, p.port, bool(p.username))
    base = "http://localhost:9"
    handlers = [urllib.request.ProxyHandler({"http": base, "https": base})]
    pm = urllib.request.ProxyBasicAuthHandler()
    pm.add_password("proxy", base, "user", "pass")
    handlers.append(pm)
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request("http://example.com/", headers={"User-Agent": "diag"})
    with opener.open(req, timeout=2) as r:
        print("unexpected_status", getattr(r, "status", None))
except Exception:
    print("proxy_sim_err")
    traceback.print_exc()
print("done")
