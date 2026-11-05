"""Recover official staging URLs from Pharmacity's public frontend API.

Endpoint discovered in publisher asset 782-b41fed6b2ffd4318.js (blogGetPost,
GET blog/posts, isAuth=false). Accepts only exact official WordPress links.
Uses batches and <=1 request/second, stores original JSON and per-URL evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from curl_cffi import requests
import pandas as pd

from recover_publisher_frontends import text_from_html


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api-gateway.pharmacity.vn/blog/posts"
HOST = "pmc-ecm-healthblog.beta.pharmacity.io"


def matched_records(payload: list, official_by_url: dict) -> dict:
    result = {}
    for post in payload:
        if not isinstance(post, dict) or post.get("status") != "publish":
            continue
        link = post.get("link", "")
        if link not in official_by_url or urlparse(link).path.strip("/") != post.get("slug"):
            continue
        content = post.get("content", {}).get("rendered", "")
        title = post.get("title", {}).get("rendered", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if not text_from_html(content).strip():
            continue
        text = text_from_html(title + "\n" + content)
        if len(text) < 30:
            continue
        result[link] = {"text": text, "wordpress_id": post.get("id"),
                        "source_content_sha256": hashlib.sha256(content.encode()).hexdigest()}
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    sources = args.source_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    targets = pd.read_parquet(args.input, columns=["id", "url"])
    targets = targets[targets["url"].map(lambda u: urlparse(u).hostname == HOST)]
    official = pd.read_parquet(ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet", columns=["id", "url"])
    mapping = official.set_index("id")["url"]
    assert targets["id"].map(mapping).eq(targets["url"]).all()
    by_url = dict(zip(targets["url"], targets["id"].astype(int)))
    done = set()
    for path in out.glob("api_part_*.parquet"):
        old = pd.read_parquet(path)
        done.update(old.loc[old["status"].eq("success"), "id"])
    targets = targets[~targets["id"].isin(done)].reset_index(drop=True)
    start = time.monotonic()
    last_request = 0.0
    succeeded = 0
    missing = 0
    print(json.dumps({"target_urls": len(targets), "already_saved_successes": len(done)}), flush=True)
    for offset in range(0, len(targets), args.batch_size):
        batch = targets.iloc[offset:offset + args.batch_size]
        slugs = [urlparse(url).path.strip("/") for url in batch["url"]]
        params = {"slug": ",".join(slugs), "per_page": 100}
        pause = max(0.0, 1.0 - (time.monotonic() - last_request))
        if pause:
            time.sleep(pause)
        last_request = time.monotonic()
        response = requests.get(ENDPOINT, params=params, timeout=35)
        if response.status_code in (403, 429):
            raise RuntimeError(f"Publisher API blocked/rate limited: {response.status_code}; stopped without hammering")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Expected a list of WordPress posts")
        sha = hashlib.sha256(response.content).hexdigest()
        raw_path = sources / f"batch_{offset:05d}_{sha[:12]}.json.gz"
        with gzip.open(raw_path, "wb") as handle:
            handle.write(response.content)
        matches = matched_records(payload, by_url)
        rows = []
        for row in batch.itertuples(index=False):
            found = matches.get(row.url)
            record = {"id": int(row.id), "url": row.url, "status": "success" if found else "error",
                      "text": found["text"] if found else None,
                      "reason": "Exact-link public WordPress API payload" if found else "No exact source record in API batch",
                      "source_url": response.url, "source_file": str(raw_path.relative_to(ROOT)),
                      "response_sha256": sha, "identity_evidence": "wordpress_exact_official_link" if found else "unverified",
                      "fetched_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(), "http_status": response.status_code}
            if found:
                record.update(wordpress_id=found["wordpress_id"], source_content_sha256=found["source_content_sha256"])
                succeeded += 1
            else:
                missing += 1
            rows.append(record)
        path = out / f"api_part_{offset:05d}_{time.time_ns()}.parquet"
        temp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(temp, index=False)
        temp.replace(path)
        if offset % (args.batch_size * 4) == 0 or offset + args.batch_size >= len(targets):
            print(json.dumps({"processed": min(offset + len(batch), len(targets)), "total": len(targets),
                              "recovered": succeeded, "not_found_in_api": missing,
                              "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)


if __name__ == "__main__":
    main()
