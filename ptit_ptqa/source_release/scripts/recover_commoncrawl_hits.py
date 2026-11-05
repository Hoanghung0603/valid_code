"""Restore original-source HTML from verified Common Crawl WARC records."""

import argparse
from datetime import datetime
import gzip
import hashlib
import http.client
import io
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

from curl_cffi import requests
import pandas as pd
import trafilatura

from probe_commoncrawl_shards import key


ROOT = Path(__file__).resolve().parents[1]


class Socket:
    def __init__(self, data):
        self.data = data

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.data)


def decode_warc(raw, expected_url, charset):
    record = gzip.decompress(raw)
    headers, separator, payload = record.partition(b"\r\n\r\n")
    if not separator or not headers.startswith(b"WARC/"):
        raise ValueError("Not a WARC response record")
    fields = dict(line.decode("utf-8").split(": ", 1) for line in headers.split(b"\r\n")[1:] if b": " in line)
    if fields.get("WARC-Type") != "response" or key(fields.get("WARC-Target-URI", "")) != key(expected_url):
        raise ValueError("WARC identity mismatch")
    response = http.client.HTTPResponse(Socket(payload))
    response.begin()
    if response.status != 200:
        raise ValueError("Archived source is not HTTP 200")
    body = response.read()
    encoding = response.getheader("Content-Encoding", "").lower()
    if encoding == "gzip":
        body = gzip.decompress(body)
    elif encoding not in ("", "identity"):
        raise ValueError(f"Unsupported archived content encoding: {encoding}")
    return body.decode(charset or "utf-8", errors="replace"), fields


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    out, sources = args.out_dir.resolve(), args.source_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    official = pd.read_parquet(ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet", columns=["id", "url"])
    by_id = official.set_index("id")["url"].to_dict()
    hits = []
    for path in args.hits_root.rglob("cdx_hits.json"):
        hits.extend(json.loads(path.read_text(encoding="utf-8")))
    successful = set()
    for path in out.glob("part_*.parquet"):
        old = pd.read_parquet(path)
        successful.update(old.loc[old["status"].eq("success"), "id"])
    for hit in sorted(hits, key=lambda x: x["collection"], reverse=True):
        row_id = int(hit["official_id"])
        if row_id in successful or str(hit.get("status")) != "200":
            continue
        if key(by_id[row_id]) != key(hit["url"]):
            raise ValueError("Index URL is not the official source")
        endpoint = "https://data.commoncrawl.org/" + hit["filename"]
        offset, length = int(hit["offset"]), int(hit["length"])
        result = {"id": row_id, "url": by_id[row_id], "status": "error", "reason": "",
                  "text": None, "source_url": endpoint, "source_file": "",
                  "archive_original_url": hit["url"], "archive_collection": hit["collection"],
                  "archive_offset": offset, "archive_length": length,
                  "fetched_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
                  "identity_evidence": "warc_exact_original_source"}
        try:
            response = requests.get(endpoint, headers={"Range": f"bytes={offset}-{offset + length - 1}"}, timeout=25)
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != length:
                raise ValueError("Archive range mismatch")
            digest = hashlib.sha256(response.content).hexdigest()
            path = sources / f"{row_id}_{hit['collection']}_{digest[:12]}.warc.gz"
            path.write_bytes(response.content)
            body, fields = decode_warc(response.content, hit["url"], hit.get("charset"))
            low = body.lower()
            if any(s in low for s in ("<title>just a moment", "cookies must be enabled", "cf-browser-verification")):
                raise ValueError("Archived anti-bot page")
            text = trafilatura.extract(body, include_links=False, include_images=False)
            if not text or len(text.strip()) < 50:
                raise ValueError("No article body in archive")
            result.update(status="success", reason="Original source recovered from Common Crawl WARC",
                          text=text.strip(), source_file=str(path.relative_to(ROOT)),
                          response_sha256=digest, archive_timestamp=fields.get("WARC-Date"),
                          archive_target_uri=fields.get("WARC-Target-URI"))
            successful.add(row_id)
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"[:200]
        path = out / f"part_{row_id}_{time.time_ns()}.parquet"
        pd.DataFrame([result]).to_parquet(path, index=False)
        print(json.dumps({"id": row_id, "status": result["status"], "reason": result["reason"],
                          "text_chars": len(result["text"] or "")}), flush=True)
        time.sleep(.35)
    print(json.dumps({"unique_successes_saved": len(successful)}), flush=True)


if __name__ == "__main__":
    main()
