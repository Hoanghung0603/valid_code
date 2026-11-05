"""Recover Pharmacity source articles through its public publishing frontend.

Stores source responses and provenance. A WordPress post with an exact official
link is strongest evidence; the public article itself requires exact publisher,
slug, canonical and headline identity. Never uses search snippets as source.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import html as html_module
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
from urllib.parse import urlparse
from uuid import uuid4
from threading import Lock
from urllib.robotparser import RobotFileParser
from zoneinfo import ZoneInfo

from curl_cffi import requests
from lxml import html
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGIN_HOST = "pmc-ecm-healthblog.beta.pharmacity.io"


class PoliteRequests:
    def __init__(self, delay: float):
        self.delay = delay
        self.lock = Lock()
        self.next_request = 0.0

    def wait(self):
        with self.lock:
            delay = max(0.0, self.next_request - time.monotonic())
            if delay:
                time.sleep(delay)
            self.next_request = time.monotonic() + self.delay


def slugify(value: str) -> str:
    value = value.lower().replace("đ", "d")
    value = "".join(c for c in unicodedata.normalize("NFKD", value) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def text_from_html(value: str) -> str:
    # Malformed publisher image attributes (e.g. alt=" height="800") can
    # otherwise swallow all following paragraphs in lxml's recovery parser.
    # Images contribute no textual content; preserve the surrounding wording.
    value = re.sub(r"<img\b[^>]*>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"</?span\b[^>]*>", "", value, flags=re.IGNORECASE)
    if not value.strip():
        return ""
    tree = html.fromstring(value)
    for node in tree.xpath("//script|//style|//noscript|//svg"):
        node.drop_tree()
    for node in tree.xpath("//p|//div|//h1|//h2|//h3|//h4|//li|//br|//tr"):
        node.tail = "\n" + (node.tail or "")
    lines = [re.sub(r"\s+", " ", line).strip() for line in tree.text_content().splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_flight(tree) -> dict:
    """Parse JSON records and byte-length-prefixed UTF-8 text in React Flight."""
    streams = []
    for script in tree.xpath("//script/text()"):
        marker = "self.__next_f.push("
        position = script.find(marker)
        if position < 0:
            continue
        try:
            payload, _ = json.JSONDecoder().raw_decode(script[position + len(marker):])
            if len(payload) > 1 and isinstance(payload[1], str):
                streams.append(payload[1])
        except (ValueError, TypeError):
            continue
    raw = "".join(streams).encode("utf-8")
    position = 0
    records = {}
    while position < len(raw):
        header = re.match(rb"([0-9a-f]+):", raw[position:])
        if not header:
            position += 1
            continue
        key = header[1].decode()
        position += header.end()
        size = re.match(rb"T([0-9a-f]+),", raw[position:])
        if size:
            position += size.end()
            length = int(size[1], 16)
            records[key] = raw[position:position + length].decode("utf-8")
            position += length
            continue
        end = raw.find(b"\n", position)
        if end < 0:
            end = len(raw)
        payload = raw[position:end]
        position = end + 1
        try:
            records[key] = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            pass
    return records


def wordpress_posts(records: dict) -> list[dict]:
    result = {}

    def walk(value, depth=0):
        if depth > 30:
            return
        if isinstance(value, dict):
            if {"link", "slug", "content"}.issubset(value):
                result[value["link"]] = value
            for child in value.values():
                walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                walk(child, depth + 1)
        elif isinstance(value, str):
            if re.fullmatch(r"\$[0-9a-f]+", value) and value[1:] in records:
                walk(records[value[1:]], depth + 1)
            elif value.lstrip().startswith(("{", "[")):
                try:
                    walk(json.loads(value), depth + 1)
                except (ValueError, RecursionError):
                    pass

    for value in records.values():
        walk(value)
    return list(result.values())


def resolve_text(value, records: dict) -> str:
    if isinstance(value, str) and re.fullmatch(r"\$[0-9a-f]+", value):
        value = records.get(value[1:], "")
    return value if isinstance(value, str) else ""


def public_article(tree, official_url: str, final_url: str) -> tuple[str, str]:
    slug = urlparse(official_url).path.strip("/")
    expected_path = f"/{slug}.htm"
    final = urlparse(final_url)
    if final.hostname not in ("www.pharmacity.vn", "pharmacity.vn") or final.path != expected_path:
        return "", "unexpected_redirect"
    canonical = tree.xpath("//link[@rel='canonical']/@href")
    if len(canonical) != 1 or urlparse(canonical[0]).path != expected_path:
        return "", "canonical_mismatch"
    if urlparse(canonical[0]).hostname not in ("www.pharmacity.vn", "pharmacity.vn"):
        return "", "canonical_publisher_mismatch"
    heading = " ".join(tree.xpath("//h1//text()")).strip()
    # WordPress can add an ordinal suffix to otherwise duplicate slugs.
    normalized = re.sub(r"-\d+$", "", slug)
    title_slug = slugify(heading)
    if title_slug not in (slug, normalized):
        return "", "headline_slug_mismatch"
    nodes = tree.xpath("//*[@id='pmc-content-blog-detail']")
    if len(nodes) != 1:
        return "", "missing_article_body"
    body = text_from_html(html.tostring(nodes[0], encoding="unicode"))
    if len(body) < 80 or any(marker in body.lower() for marker in ("just a moment", "403 forbidden")):
        return "", "empty_or_blocked_body"
    # Heading plus article body, preserving source wording. No author/footer.
    return heading + "\n" + body, "same_publisher_exact_slug_canonical_headline"


def extract_records(record: dict, raw: bytes, official_by_url: dict) -> tuple[dict, list[dict]]:
    extras = []
    tree = html.fromstring(raw)
    text, evidence = public_article(tree, record["url"], record["source_url"])
    record["identity_evidence"] = evidence
    if text:
        record.update(text=text, status="review", reason="Publisher frontend candidate; exact source link required")
    else:
        record["reason"] = evidence
    flight = parse_flight(tree)
    for post in wordpress_posts(flight):
        link = html_module.unescape(str(post.get("link", "")))
        if link not in official_by_url:
            continue
        body = resolve_text(post.get("content", {}).get("rendered", ""), flight)
        title = resolve_text(post.get("title", {}).get("rendered", ""), flight)
        if len(body) < 80 or post.get("status") != "publish":
            continue
        content = text_from_html(title + "\n" + body)
        if len(content) < 80:
            continue
        evidence_row = record.copy()
        evidence_row.update(id=official_by_url[link], url=link, text=content,
                            status="success", reason="Exact-link WordPress source payload",
                            identity_evidence="wordpress_exact_official_link", wordpress_id=post.get("id"))
        extras.append(evidence_row)
    return record, extras


def fetch_one(row: dict, official_by_url: dict, source_dir: Path,
              limiter: PoliteRequests) -> tuple[dict, list[dict]]:
    start = time.monotonic()
    origin = str(row["url"])
    target = "https://www.pharmacity.vn/" + urlparse(origin).path.strip("/") + ".htm"
    record = dict(id=int(row["id"]), url=origin, text=None, status="error", reason="",
                  source_url=target, fetched_at=datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
                  http_status=0, identity_evidence="", response_sha256="", source_file="")
    extras = []
    try:
        limiter.wait()
        response = requests.get(target, timeout=22, impersonate="chrome124")
        record["source_url"] = response.url
        record["http_status"] = response.status_code
        raw = response.content
        digest = hashlib.sha256(raw).hexdigest()
        record["response_sha256"] = digest
        path = source_dir / f"{record['id']}_{digest[:12]}.html.gz"
        with gzip.open(path, "wb") as handle:
            handle.write(raw)
        record["source_file"] = str(path.relative_to(ROOT))
        if response.status_code != 200:
            record["reason"] = f"Frontend HTTP {response.status_code}"
        else:
            record, extras = extract_records(record, raw, official_by_url)
    except Exception as error:
        # Never include proxy or other credentials in errors.
        record["reason"] = type(error).__name__
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return record, extras


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=int, default=50)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Re-extract saved responses without network")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input, columns=["id", "url"])
    official = pd.read_parquet(ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet", columns=["id", "url"])
    official_map = official.set_index("id")["url"]
    if not frame["id"].map(official_map).eq(frame["url"]).all():
        raise ValueError("Input contains unofficial identities")
    frame = frame[frame["url"].map(lambda u: urlparse(u).hostname == ORIGIN_HOST)]
    by_url = dict(zip(frame["url"], frame["id"].astype(int)))
    if args.offline:
        by_id = dict(zip(frame["id"].astype(int), frame["url"]))
        outputs = []
        for path in sorted(args.source_dir.glob("*.html.gz")):
            row_id = int(path.name.split("_")[0])
            if row_id not in by_id:
                continue
            raw = gzip.decompress(path.read_bytes())
            tree = html.fromstring(raw)
            canonical = tree.xpath("//link[@rel='canonical']/@href")
            source_url = canonical[0] if canonical else "https://www.pharmacity.vn/" + urlparse(by_id[row_id]).path.strip("/") + ".htm"
            record = dict(id=row_id, url=by_id[row_id], text=None, status="error", reason="",
                          source_url=source_url,
                          fetched_at=datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Asia/Bangkok")).isoformat(),
                          http_status=200 if canonical else 0, identity_evidence="",
                          response_sha256=hashlib.sha256(raw).hexdigest(), source_file=str(path.relative_to(ROOT)))
            try:
                main_record, extras = extract_records(record, raw, by_url)
                main_record["record_kind"] = "offline_requested"
                outputs.append(main_record)
                for extra in extras:
                    extra["record_kind"] = "wordpress_payload"
                    outputs.append(extra)
            except Exception as error:
                record.update(reason=type(error).__name__, record_kind="offline_error")
                outputs.append(record)
        output = args.out_dir / f"part_offline_{time.time_ns()}.parquet"
        pd.DataFrame(outputs).to_parquet(output, index=False)
        successful = pd.DataFrame(outputs).query("status == 'success'")
        print(json.dumps({"saved_responses": len(list(args.source_dir.glob('*.html.gz'))),
                          "unique_success_ids": int(successful['id'].nunique()), "output": str(output)}), flush=True)
        return
    robots_url = "https://www.pharmacity.vn/robots.txt"
    robots_response = requests.get(robots_url, timeout=20)
    robots_response.raise_for_status()
    robot = RobotFileParser()
    robot.parse(robots_response.text.splitlines())
    if not robot.can_fetch("*", "https://www.pharmacity.vn/example.htm"):
        raise RuntimeError("Publisher robots.txt disallows this crawl")
    delay = max(1.0, float(robot.crawl_delay("*") or 1.0))
    limiter = PoliteRequests(delay)
    limiter.next_request = time.monotonic() + delay
    print(json.dumps({"crawl_delay_seconds": delay}), flush=True)
    processed = set()
    existing_success = set()
    for path in args.out_dir.glob("part_*.parquet"):
        part = pd.read_parquet(path)
        requested = part["record_kind"].eq("requested")
        if args.retry_errors:
            requested &= part["status"].eq("success")
        processed.update(part.loc[requested, "id"])
        existing_success.update(part.loc[part["status"].eq("success"), "id"])
    frame = frame[~frame["id"].isin(processed | existing_success)]
    if args.limit:
        frame = frame.head(args.limit)
    print(json.dumps({"pending_pages": len(frame), "previous_success_ids": len(existing_success)}), flush=True)
    staged = []
    start = time.monotonic()
    done = 0
    counts = {}
    successes = set(existing_success)

    def checkpoint():
        if staged:
            output = args.out_dir / f"part_{time.time_ns()}_{uuid4().hex[:8]}.parquet"
            temporary = output.with_suffix(".tmp")
            pd.DataFrame(staged).to_parquet(temporary, index=False)
            temporary.replace(output)
            staged.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_one, row, by_url, args.source_dir, limiter) for row in frame.to_dict("records")]
        for future in as_completed(futures):
            record, extras = future.result()
            record["record_kind"] = "requested"
            staged.append(record)
            for extra in extras:
                extra["record_kind"] = "wordpress_payload"
                staged.append(extra)
            for value in [record] + extras:
                if value["status"] == "success":
                    successes.add(int(value["id"]))
            counts[record["reason"]] = counts.get(record["reason"], 0) + 1
            done += 1
            if done % args.checkpoint == 0:
                checkpoint()
                print(json.dumps({"pages_done": done, "total_pages": len(frame),
                                  "unique_success_ids": len(successes), "counts": counts,
                                  "elapsed_seconds": round(time.monotonic() - start, 1)}, ensure_ascii=False), flush=True)
    checkpoint()
    print(json.dumps({"pages_done": done, "unique_success_ids": len(successes), "counts": counts,
                      "elapsed_seconds": round(time.monotonic() - start, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
