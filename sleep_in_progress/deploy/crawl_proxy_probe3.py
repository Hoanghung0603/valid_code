"""Probe real queue URLs (Long Chau) through distinct proxy routes."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crawl_async_supervisor as sup

PROXY_FILE = Path(r"D:\r2ai\data\crawl\tools\proxy_secrets.txt")
QUEUE = Path(r"D:\r2ai\data\crawl\async_recovery_20261009\queue\host-nhathuoclongchau_com_vn.jsonl.zst")
LIMITER = sup.BandwidthLimiter()


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def urls(n=10):
    import zstandard as zstd
    out = []
    dctx = zstd.ZstdDecompressor()
    with QUEUE.open("rb") as fh:
        with dctx.stream_reader(fh) as rd:
            buf = b""
            while len(out) < n:
                chunk = rd.read(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf and len(out) < n:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        out.append(json.loads(line)["url"])
    return out


def main():
    routes = sup.load_proxies(PROXY_FILE, 100)
    us = urls(10)
    log(f"probe3 start routes={len(routes)} urls={len(us)}")
    for i, url in enumerate(us):
        t0 = time.monotonic()
        res = sup.fetch_via_proxy(url, routes[i], LIMITER, timeout=25)
        raw = res.get("body") or b""
        status = res.get("status")
        text, title = "", ""
        if status == 200 and raw:
            try:
                import trafilatura
                text = trafilatura.extract(raw.decode("utf-8", "ignore")) or ""
                if text:
                    title = ""
            except Exception:
                pass
        outcome = sup.classify_inline(status, res.get("final_url") or "", text,
                                      title, raw, res.get("error"))
        log(f"url#{i} route#{i} status={status} bytes={len(raw)} "
            f"text={len(text)} outcome={outcome} err={res.get('error')} "
            f"t={time.monotonic() - t0:.1f}s")
    log("probe3 done")


if __name__ == "__main__":
    main()
