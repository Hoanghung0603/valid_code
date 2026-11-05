"""Consolidate run-1 micro-shards into one closed shard (recovery-safe merge).

Run-1 of the async supervisor rotated a new pages shard on every durability
flush (rotator bug, now fixed), leaving ~175 micro-shards with 181 valid
records. This tool merges them into a single closed shard WITHOUT losing a
record and WITHOUT deleting evidence: originals move to pages_run1_micro/ by
explicit name (no wildcard deletes). Never touches original crawl state/pages.

Usage: python crawl_merge_micro_shards.py --pages <dir> --archive-dir <dir>
Output: <pages>/recov-00000.jsonl.zst (only when no non-micro shard exists),
        stdout JSON {merged, records, ids_unique, schema_ok}.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
MIN_TEXT = 200
MERGED_NAME = "recov-00000.jsonl.zst"


def read_records(path: Path):
    import zstandard as zstd
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    if raw.strip():
                        yield json.loads(raw.decode("utf-8", "ignore"))
            if buf.strip():
                yield json.loads(buf.decode("utf-8", "ignore"))


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", required=True, type=Path)
    ap.add_argument("--archive-dir", required=True, type=Path)
    a = ap.parse_args(argv)
    pages, arch = a.pages, a.archive_dir
    files = sorted(pages.glob("recov-*.jsonl.zst"))
    if not files:
        print(json.dumps({"merged": 0, "reason": "no shards"}))
        return 0
    big = [p for p in files if p.stat().st_size > 100_000]
    if big:
        print(json.dumps({"merged": 0,
                          "reason": f"refuse: {len(big)} non-micro shard(s) present"}))
        return 2
    if (pages / MERGED_NAME).exists() and (pages / MERGED_NAME).stat().st_size > 100_000:
        print(json.dumps({"merged": 0, "reason": "target exists"}))
        return 2
    arch.mkdir(parents=True, exist_ok=True)
    recs, ids, schema_ok, content_ok = [], set(), True, True
    for p in files:
        for rec in read_records(p):
            if tuple(rec) != SCHEMA:
                schema_ok = False
            if rec.get("status") != 200 or len(rec.get("text") or "") < MIN_TEXT:
                content_ok = False
            if rec["id"] in ids:
                continue  # exact output-id dedup
            ids.add(rec["id"])
            recs.append(rec)
    recs.sort(key=lambda r: r["id"])
    tmp = pages / "recov-merge.tmp"
    if tmp.exists():
        print(json.dumps({"merged": 0, "reason": "tmp exists"}))
        return 2
    import zstandard as zstd
    with tmp.open("xb") as fh:
        with zstd.ZstdCompressor(level=9).stream_writer(fh) as zw:
            for rec in recs:
                zw.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
    # move originals first (explicit list captured above, never wildcard
    # delete), then promote the merged file to the canonical shard name
    moved = []
    for p in files:
        dest = arch / p.name
        p.replace(dest)
        moved.append(p.name)
    tmp.replace(pages / MERGED_NAME)
    out = pages / MERGED_NAME
    h = hashlib.sha256()
    with out.open("rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    print(json.dumps({"merged": len(moved), "records": len(recs),
                      "ids_unique": len(ids), "schema_ok": schema_ok,
                      "content_ok": content_ok,
                      "out": out.name, "sha256": h.hexdigest(),
                      "archived_to": str(arch)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
