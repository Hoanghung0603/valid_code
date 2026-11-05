"""Convert delivered clean VI JSONL.zst to shard_docs schema for a new index.

The source ``text`` is vi55's title+clean body. Strip its title prefix exactly
once so bigindex does not prepend the title a second time. This cannot restore
the original pre-cleaned crawl or reproduce the original index byte-for-byte.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import zstandard as zstd


def split_body(title, body):
    title = title or ''
    if not title:
        return '', body
    prefix = (title + '\n').lstrip()
    if body.startswith(prefix):
        return title, body[len(prefix):]
    if body == title.strip():
        return title, ''
    # Whitespace at the start of the source title is stripped by make_body.
    raise ValueError('Cannot safely separate title from body')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-dir', type=Path, required=True)
    p.add_argument('--links', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--max-shards', type=int, default=None, help='Only for smoke tests')
    args = p.parse_args()
    assert not args.out_dir.exists(), 'Refusing to overwrite output'
    files = sorted(args.corpus_dir.glob('corpus.part*.jsonl.zst'))
    if args.max_shards is not None:
        if args.max_shards <= 0:
            p.error('--max-shards must be positive')
        files = files[:args.max_shards]
    if not files:
        p.error('No corpus shards found')
    rows = pq.read_table(args.links, columns=['id', 'url']).to_pylist()
    urls = {int(r['id']): r['url'] for r in rows}
    if len(urls) != len(rows):
        raise ValueError('Duplicate IDs in links')
    del rows
    args.out_dir.mkdir(parents=True)
    seen = set()
    for src in files:
        count = 0
        target = args.out_dir / src.name
        with zstd.open(src, 'rt', encoding='utf-8') as inp, zstd.open(target, 'wt', encoding='utf-8') as out:
            for line in inp:
                r = json.loads(line)
                did = int(r['id'])
                if did in seen or did not in urls or urls[did] != r['url']:
                    raise ValueError(f'duplicate ID or unmatched URL: {did}')
                seen.add(did)
                title, body = split_body(r.get('title'), r['text'])
                host = urlsplit(urls[did]).hostname
                if not host:
                    raise ValueError(f'missing host: {did}')
                out.write(json.dumps({'id': did, 'url': urls[did], 'host': host, 'status': 200,
                                      'title': title, 'text': body}, ensure_ascii=False) + '\n')
                count += 1
        print(f'{src.name}: {count}', flush=True)


if __name__ == '__main__':
    main()
