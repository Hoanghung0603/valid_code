"""Query public Common Crawl CDX blocks directly when the CDX API is down.

Uses byte ranges on the published cluster.idx and compressed CDX blocks. No
origin restrictions are bypassed; this reads Common Crawl's public archives.
"""

import argparse
import gzip
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from curl_cffi import requests
import pandas as pd


class Cluster:
    def __init__(self, collection, output):
        self.base = f"https://data.commoncrawl.org/cc-index/collections/{collection}/indexes/"
        self.output = output
        self.last = 0.0
        head = self.get(self.base + "cluster.idx", head=True)
        self.total = int(head.headers["content-length"])

    def get(self, url, start=None, length=None, head=False):
        pause = max(0, .35 - (time.monotonic() - self.last))
        if pause:
            time.sleep(pause)
        self.last = time.monotonic()
        headers = {"Range": f"bytes={start}-{start + length - 1}"} if start is not None else {}
        response = requests.head(url, timeout=25) if head else requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        if start is not None:
            if response.status_code != 206 or not response.headers.get("content-range", "").startswith(f"bytes {start}-"):
                raise ValueError("Range response mismatch")
        return response

    def rows(self, start, length):
        response = self.get(self.base + "cluster.idx", start, min(length, self.total - start))
        data = response.content
        if start:
            data = data.partition(b"\n")[2]
        lines = data.splitlines()[:-1]
        return [line.decode().split("\t") for line in lines if line.count(b"\t") >= 4]

    def blocks(self, prefix):
        low, high = 0, self.total
        while high - low > 32768:
            middle = (high + low) // 2
            rows = self.rows(middle, 8192)
            if not rows:
                raise ValueError("Empty cluster range")
            if rows[0][0] < prefix:
                low = middle
            else:
                high = middle
        rows = self.rows(max(0, low - 32768), min(131072, self.total - max(0, low - 32768)))
        before = [row for row in rows if row[0] < prefix]
        within = [row for row in rows if row[0].startswith(prefix)]
        return ([before[-1]] if before else []) + within


def key(url):
    parsed = urlparse(url)
    return (parsed.hostname or "").removeprefix("www.") + parsed.path.rstrip("/") + ("?" + parsed.query if parsed.query else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--collection", default="CC-MAIN-2026-39")
    parser.add_argument("--hosts", nargs="+", default=["zysjonline.com", "article.iiyi.com", "bingli.iiyi.com"])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input, columns=["id", "url"])
    targets = {key(row.url): int(row.id) for row in frame.itertuples(index=False)}
    cluster = Cluster(args.collection, args.out_dir)
    hits = []
    results = []
    for requested_host in args.hosts:
        host = requested_host.removeprefix("www.")
        prefix = ",".join(reversed(host.split("."))) + ")/"
        try:
            blocks = cluster.blocks(prefix)
            observed = 0
            matches = 0
            for row in blocks:
                filename, offset, length = row[1], int(row[2]), int(row[3])
                content = cluster.get(cluster.base + filename, offset, length).content
                path = args.out_dir / f"{filename}_{offset}_{length}.gz"
                path.write_bytes(content)
                for line in gzip.decompress(content).splitlines():
                    parts = line.split(b" ", 2)
                    if len(parts) != 3:
                        continue
                    data = json.loads(parts[2])
                    url = data.get("url", "")
                    if (urlparse(url).hostname or "").removeprefix("www.") != host:
                        continue
                    observed += 1
                    if key(url) not in targets:
                        continue
                    data["official_id"] = targets[key(url)]
                    data["collection"] = args.collection
                    data["index_block"] = str(path)
                    hits.append(data)
                    matches += 1
            result = {"host": host, "cluster_blocks": len(blocks), "observed_urls": observed,
                      "target_matches": matches, "http200_matches": sum(str(x.get("status")) == "200" for x in hits if (urlparse(x.get("url", "")).hostname or "").removeprefix("www.") == host)}
        except Exception as error:
            result = {"host": host, "error": f"{type(error).__name__}: {error}"[:300]}
        results.append(result)
        print(json.dumps(result), flush=True)
    (args.out_dir / "probe_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out_dir / "cdx_hits.json").write_text(json.dumps(hits, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
