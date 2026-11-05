"""Read-only probe for the proxy lane: auth/connectivity vs target block.

Never prints credentials or full URLs with auth. Results go to stdout and the
log file appended by the .bat wrapper.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_async_supervisor as sup

PROXY_FILE = Path(r"D:\r2ai\data\crawl\tools\proxy_secrets.txt")
TARGETS = [
    ("ipify", "https://api.ipify.org/?format=json"),
    ("longchau", "https://nhathuoclongchau.com.vn/"),
]
ROUTE_IDX = [0, 17, 34, 51, 68, 85]


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def main():
    routes = sup.load_proxies(PROXY_FILE, 100)
    log(f"probe start routes={len(routes)}")
    limiter = sup.BandwidthLimiter()
    for idx in ROUTE_IDX:
        if idx >= len(routes):
            continue
        route = routes[idx]
        for name, url in TARGETS:
            t0 = time.monotonic()
            try:
                res = sup.fetch_via_proxy(url, route, limiter, timeout=20)
                dt = time.monotonic() - t0
                log(f"route#{idx} {name} status={res.get('status')} "
                    f"bytes={res.get('nbytes', 0)} err={res.get('error')} "
                    f"t={dt:.1f}s")
            except Exception as e:
                log(f"route#{idx} {name} EXC={type(e).__name__}")
    log("probe done")


if __name__ == "__main__":
    main()
