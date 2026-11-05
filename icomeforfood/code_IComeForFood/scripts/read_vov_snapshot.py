"""Verify the shared VOV raw snapshot or extract one original HTML document."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import zstandard as zstd


def records(snapshot, name):
    if Path(name).name != name:
        raise ValueError(f'Invalid shard filename: {name}')
    with (snapshot / 'html' / name).open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as reader, io.TextIOWrapper(reader, encoding='utf-8') as lines:
        for line in lines:
            yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).resolve().parents[1] / 'crawl/vov_snapshot_2026-10-30')
    parser.add_argument('--verify', action='store_true', help='Verify file SHA-256 and every decoded HTML SHA-1')
    parser.add_argument('--id', type=int, help='Original corpus ID to extract')
    parser.add_argument('--out', type=Path, help='Write the original HTML bytes for --id here')
    args = parser.parse_args(argv)
    if (args.id is None) != (args.out is None):
        parser.error('Use --id and --out together.')
    snapshot = args.snapshot.resolve()
    with (snapshot / 'manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        manifest = {int(row['id']): row for row in csv.DictReader(handle)}
    summary = json.loads((snapshot / 'summary.json').read_text(encoding='utf-8'))
    if len(manifest) != summary['saved_unique_ids']:
        raise ValueError('Manifest count differs from summary')
    if args.verify:
        checksums = json.loads((snapshot / 'checksums.json').read_text(encoding='utf-8'))
        for name, expected in checksums.items():
            path = (snapshot / name).resolve()
            if not path.is_relative_to(snapshot):
                raise ValueError(f'Invalid checksum path: {name}')
            data = path.read_bytes()
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError(f'File checksum mismatch: {name}')
        seen = set()
        for name in sorted({row['shard'] for row in manifest.values()}):
            for rec in records(snapshot, name):
                rid = int(rec['id'])
                raw = base64.b64decode(rec['body_b64'], validate=True)
                expected = manifest[rid]
                if rid in seen or expected['shard'] != name or rec['http_code'] != 200:
                    raise ValueError(f'Duplicate ID or record mismatch: {rid}')
                if len(raw) != int(expected['n_bytes']) or hashlib.sha1(raw).hexdigest() != expected['sha1_raw']:
                    raise ValueError(f'HTML checksum mismatch: {rid}')
                seen.add(rid)
        if seen != set(manifest):
            raise ValueError('Missing raw IDs')
        print(f'Verified {len(checksums)} files and {len(seen):,} unique HTML records.')
    if args.id is not None:
        expected = manifest[args.id]
        for rec in records(snapshot, expected['shard']):
            if int(rec['id']) == args.id:
                raw = base64.b64decode(rec['body_b64'], validate=True)
                if hashlib.sha1(raw).hexdigest() != expected['sha1_raw']:
                    raise ValueError('HTML checksum mismatch')
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_bytes(raw)
                print(f'Saved {args.id}: {expected["title"]}\nHTML: {args.out.resolve()}')
                break
        else:
            raise ValueError(f'ID missing from its shard: {args.id}')
    if not args.verify and args.id is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    sys.exit(main())
