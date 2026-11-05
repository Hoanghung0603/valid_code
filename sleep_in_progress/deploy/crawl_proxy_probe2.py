"""Proxy lane probe v2: sequential vs concurrent load against Long Chau.

Used to tell a dead lane from per-host rate/WAF behaviour. No secrets logged.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_async_supervisor as sup

PROXY_FILE = Path(r"D:\r2ai\data\crawl\tools\proxy_secrets.txt")
URL = "https://nhathuoclongchau.com.vn/"
LIMITER = sup.BandwidthLimiter()


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def probe(route):
    t0 = time.monotonic()
    try:
        res = sup.fetch_via_proxy(URL, route, LIMITER, timeout=20)
        return res.get("status"), res.get("error"), time.monotonic() - t0
    except Exception as e:
        return None, type(e).__name__, time.monotonic() - t0


def main():
    routes = sup.load_proxies(PROXY_FILE, 100)
    log(f"probe2 start routes={len(routes)}")
    counts = {}
    for idx in range(20):
        st, err, dt = probe(routes[idx])
        counts[st] = counts.get(st, 0) + 1
        log(f"seq route#{idx} status={st} err={err} t={dt:.1f}s")
    log(f"seq counts={counts}")
    counts2 = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(probe, routes[20 + i]): i for i in range(12)}
        for fut in as_completed(futs):
            st, err, dt = fut.result()
            counts2[st] = counts2.get(st, 0) + 1
            log(f"par route#{20 + futs[fut]} status={st} err={err} t={dt:.1f}s")
    log(f"par counts={counts2}")
    log("probe2 done")


if __name__ == "__main__":
    main()
