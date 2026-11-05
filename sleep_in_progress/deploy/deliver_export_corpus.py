"""One-shot VI clean snapshot and small input packages; never modifies sources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime

ROOT = Path('D:/r2ai')
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pyarrow.parquet as pq
import zstandard as zstd
from bigindex import BigIndex
from boilerplate import BoilerplateFilter
from export_docs import iter_all_texts
from shards import read_shard
from vi55 import vi_corpus

OUT = ROOT / 'deliver/data'
LOCK = ROOT / 'locks/heavy_job.lock'


def fingerprint(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def manifest(folder, description, extra=None):
    files = {p.name: fingerprint(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'MANIFEST.json'}
    (folder / 'MANIFEST.json').write_text(json.dumps({'description': description, 'files': files, **(extra or {})}, ensure_ascii=False, indent=2), encoding='utf-8')


def small_packages():
    packages = {
        'r2ai_queries_links': [('data/query.parquet', 'query.parquet'), ('data/sub/diag19/links_corpus_vi_en_zh.parquet', 'links_corpus_vi_en_zh.parquet')],
        'r2ai_llm_outputs': [('data/llm/hyde.jsonl', 'hyde.jsonl'), ('work/hyde/q_hyde.jsonl', 'q_hyde.jsonl'), ('data/sub/vi41/rewrites.jsonl', 'vi41_rewrites.jsonl')],
    }
    for name, sources in packages.items():
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=False)
        missing = []
        for src, dest in sources:
            p = ROOT / src
            if p.is_file():
                shutil.copyfile(p, folder / dest)
            else:
                missing.append(src)
        manifest(folder, 'Query và links chuẩn' if name.endswith('links') else 'HyDE Qwen3.5-9B; rewrite vi41 Qwen3-14B', {'source_files': dict(sources), 'missing_sources': missing, 'complete': not missing})
        print(name, 'missing', missing, flush=True)


def corpus():
    folder = OUT / 'r2ai_corpus_vi_clean'
    folder.mkdir(parents=True, exist_ok=False)
    links = ROOT / 'data/sub/diag19/links_corpus_vi_en_zh.parquet'
    rows = pq.read_table(links).to_pylist()
    allowed = vi_corpus(rows)
    urls = {int(r['id']): r['url'] for r in rows[:740598]}
    del rows
    idx = BigIndex(ROOT / 'data/index/vi36')
    wanted = {int(d): i for i, d in enumerate(idx.ids) if int(d) in allowed}
    by_shard = defaultdict(dict)
    for d, i in wanted.items():
        by_shard[idx.shard_of(i)][d] = i
    bp = BoilerplateFilter.from_json((ROOT / 'data/index/vi36/boilerplate.json').read_text(encoding='utf-8'))
    seen, nonempty = set(), set()
    writer = None
    size = part = 0
    def close_writer():
        nonlocal writer
        if writer is not None:
            writer.close()
            assert (folder / f'corpus.part{part-1:03d}.jsonl.zst').stat().st_size <= 500_000_000
            writer = None
    try:
        for n, (shard, group) in enumerate(sorted(by_shard.items()), 1):
            if datetime.now().hour >= 19:
                raise TimeoutError('Hard stop 19:00')
            # First status-200 record is exactly the choice in iter_all_texts.
            left, titles = set(group), {}
            for r in read_shard(Path(shard)):
                d = r.get('id')
                if d in left and r.get('status') == 200:
                    left.remove(d)
                    titles[d] = r.get('title') or ''
                    if not left:
                        break
            for d, _, text in iter_all_texts(idx, group, bp, 2**31-1):
                assert d not in seen
                seen.add(d)
                if not text.strip():
                    continue
                nonempty.add(d)
                line = (json.dumps({'id': d, 'url': urls[d], 'title': titles[d], 'text': text}, ensure_ascii=False) + '\n').encode('utf-8')
                assert len(line) < 64_000_000
                if writer is None or size + len(line) > 64_000_000:
                    close_writer()
                    writer = zstd.ZstdCompressor(level=6).stream_writer((folder / f'corpus.part{part:03d}.jsonl.zst').open('xb'))
                    part += 1
                    size = 0
                writer.write(line)
                size += len(line)
            print('SHARD', n, '/', len(by_shard), 'nonempty', len(nonempty), flush=True)
    finally:
        close_writer()
    missing = sorted(allowed - nonempty)
    expected = set(map(int, np.load(ROOT / 'data/sub/vi55/all_ids.npy', allow_pickle=False)))
    assert len(nonempty) == 729704 and len(missing) == 10894
    assert expected == nonempty, 'ID set differs from vi55'
    (folder / 'ids_without_text.json').write_text(json.dumps(missing), encoding='utf-8')
    manifest(folder, 'VI clean: text = make_body(title, BoilerplateFilter.clean(host, text), 2**31-1); first status 200 per indexed ID', {'documents': len(nonempty), 'without_text': len(missing), 'vi55_exact_ids': True, 'unreadable_index_ids': len(set(wanted)-seen), 'source_links_sha256': fingerprint(links)['sha256']})


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if datetime.now().hour >= 19:
            raise TimeoutError('Hard stop 19:00')
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(10)
    label = f'deliver_export_corpus pid={os.getpid()}'
    with os.fdopen(fd, 'w') as f:
        f.write(label)
    try:
        small_packages()
        corpus()
    finally:
        if LOCK.exists() and LOCK.read_text(encoding='utf-8') == label:
            LOCK.unlink()


if __name__ == '__main__':
    main()
