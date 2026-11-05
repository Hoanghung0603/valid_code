"""Export saved Long Chau pages in the reviewed TV corpus for Git sharing.

The source crawl directories are read only. Each ID appears once in the export,
preferring the fast crawler, then the older odd and raw crawlers, then home TV.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import zstandard as zstd


SOURCES = ("longchau_odd_fast", "longchau_odd", "longchau_odd_raw", "crawl_home_tv")
STATUS_FIELDS = (
    "id", "url", "final_url", "lang_expected", "status", "http_code",
    "content_type", "n_bytes", "attempts", "fetched_at", "shard", "sha1_raw",
    "error_msg", "shard_file",
)
MANIFEST_FIELDS = (
    "id", "url", "source", "source_shard", "export_shard", "final_url",
    "http_code", "content_type", "encoding", "body_bytes", "body_sha256",
)


def corpus_rows(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rid = int(row["id"])
            url = row["url"]
            if rid % 2 != 1 or "nhathuoclongchau.com.vn" not in url or rid in rows:
                raise ValueError(f"Invalid or duplicate corpus ID: {rid}")
            rows[rid] = url
    return rows


def raw_shards(source: Path) -> list[Path]:
    return sorted((source / "html").glob("home_shard_*.jsonl.zst")) + sorted(
        (source / "html").glob("tv_shard_*.jsonl.zst")
    )


def export(corpus: Path, data: Path, out: Path, part_size: int) -> None:
    wanted = corpus_rows(corpus)
    out.mkdir(parents=True, exist_ok=False)
    raw_dir = out / "raw"
    raw_dir.mkdir()
    started = datetime.now(timezone.utc)

    seen: set[int] = set()
    manifest: list[dict[str, object]] = []
    part_number = 0
    part_count = 0
    part_file = None
    part_writer = None
    part_name = ""

    def close_part() -> None:
        nonlocal part_file, part_writer
        if part_writer is not None:
            part_writer.close()
            part_file.close()
            part_writer = None
            part_file = None

    try:
        for source_name in SOURCES:
            for shard in raw_shards(data / source_name):
                with shard.open("rb") as compressed:
                    with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                        for line in io.TextIOWrapper(reader, encoding="utf-8"):
                            record = json.loads(line)
                            rid = int(record["id"])
                            if rid not in wanted or rid in seen:
                                continue
                            body = base64.b64decode(record["body_b64"], validate=True)
                            if not body or int(record["http_code"]) != 200:
                                raise ValueError(f"Invalid successful raw record: {rid} in {shard}")
                            if part_writer is None or part_count >= part_size:
                                close_part()
                                part_name = f"part-{part_number:04d}.jsonl.zst"
                                part_file = (raw_dir / part_name).open("wb")
                                part_writer = zstd.ZstdCompressor(level=5).stream_writer(
                                    part_file, closefd=False
                                )
                                part_number += 1
                                part_count = 0
                            part_writer.write(line.encode("utf-8"))
                            part_count += 1
                            seen.add(rid)
                            manifest.append({
                                "id": rid,
                                "url": wanted[rid],
                                "source": source_name,
                                "source_shard": shard.name,
                                "export_shard": part_name,
                                "final_url": record.get("final_url", ""),
                                "http_code": record["http_code"],
                                "content_type": record.get("content_type", ""),
                                "encoding": record.get("encoding", ""),
                                "body_bytes": len(body),
                                "body_sha256": hashlib.sha256(body).hexdigest(),
                            })
    finally:
        close_part()

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda row: int(row["id"])))

    status_count = 0
    with (out / "status_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", *STATUS_FIELDS))
        writer.writeheader()
        for source_name in SOURCES:
            path = data / source_name / "status.csv"
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as source:
                for row in csv.DictReader(source):
                    value = row.get("id", "")
                    if value.isdigit() and int(value) in wanted:
                        writer.writerow({"source": source_name, **{
                            field: row.get(field, "") for field in STATUS_FIELDS
                        }})
                        status_count += 1

    expected = {int(row["id"]): str(row["body_sha256"]) for row in manifest}
    verified: set[int] = set()
    for part in sorted(raw_dir.glob("*.jsonl.zst")):
        if part.stat().st_size >= 95 * 1024 * 1024:
            raise ValueError(f"GitHub file size margin exceeded: {part}")
        with part.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    record = json.loads(line)
                    rid = int(record["id"])
                    body_hash = hashlib.sha256(base64.b64decode(record["body_b64"], validate=True)).hexdigest()
                    if rid in verified or expected.get(rid) != body_hash:
                        raise ValueError(f"Export verification failed for ID {rid}")
                    verified.add(rid)
    if verified != seen:
        raise ValueError(f"Export has {len(seen - verified)} missing records")

    total_bytes = sum(path.stat().st_size for path in raw_dir.glob("*.zst"))
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    (out / "README.md").write_text(
        f"# Long Châu TV crawl snapshot\n\n"
        f"Snapshot started: {started.isoformat()}\n\n"
        f"Corpus: `crawl/longchau_tv_crawl_odd.csv` ({len(wanted):,} odd IDs, "
        f"SHA-256 `{corpus_hash}`).\n\n"
        f"Saved raw HTML: **{len(seen):,}/{len(wanted):,} IDs** in {part_number} "
        f"Zstandard JSONL parts ({total_bytes:,} compressed bytes). "
        f"Remaining IDs had no saved raw page at snapshot time.\n\n"
        f"`raw/` contains the original crawler records (including base64 HTML bytes), "
        f"filtered to the corpus and deduplicated by ID. "
        f"Source priority: {', '.join(SOURCES)}. "
        f"`manifest.csv` maps each exported ID to its source and part and gives "
        f"the SHA-256 of the decoded HTML bytes. "
        f"`status_history.csv` contains {status_count:,} attempt/status rows for corpus IDs "
        f"from the same source directories; it is not a count of distinct pages.\n\n"
        f"Rebuild with `python scripts/export_longchau_tv_snapshot.py "
        f"--corpus crawl/longchau_tv_crawl_odd.csv "
        f"--data D:/r2ai_data/crawl --out NEW_OUTPUT_DIR`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "corpus_ids": len(wanted), "saved_ids": len(seen),
        "missing_ids": len(wanted) - len(seen), "parts": part_number,
        "compressed_bytes": total_bytes, "status_rows": status_count,
        "out": str(out),
    }, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("D:/r2ai_data/crawl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part-size", type=int, default=100)
    args = parser.parse_args()
    if args.part_size < 1:
        parser.error("--part-size must be positive")
    export(args.corpus, args.data, args.out, args.part_size)


if __name__ == "__main__":
    main()
