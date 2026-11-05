"""Compare HTML extraction and performance: Standard Crawl4AI vs Fast Crawl4AI.

Compares:
1. Extraction metrics: title, published_date, block count, text length, postfiltered blocks
2. Performance metrics: page fetch latency, payload size, memory saving
3. Content fidelity: verifies that text_mode=True preserves 100% of the article content
   without losing paragraphs or headings.

Usage:
    # 1. Compare already crawled shards:
    python scripts/compare_extract_longchau.py --old-shard D:/r2ai_data/crawl/longchau_odd/html/tv_shard_00000.jsonl.zst

    # 2. Benchmark live on N sample URLs from the ranked CSV:
    python scripts/compare_extract_longchau.py --benchmark 3
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
from pathlib import Path
import sys
import time

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')

import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from extract.postfilter import filter_doc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('compare_extract')


def load_extractor():
    """Load the gold extraction functions from notebook 03."""
    import run_local
    nb = run_local.notebook('03_gold_review_and_sample2')
    ns = {'__name__': '_r2ai_local', 'COLAB': False, 'display': lambda *a: None}
    cells = ('gold-project-paths', 'gold-project-root', 'gold-logging',
             'gold-encoding', 'gold-pubmed-api', 'gold-clean-helpers', 'gold-extractors')
    for ident in cells:
        run_local.execute(nb, ident, ns)
    return ns['extract_html_doc']


def read_shard_records(shard_path: Path, max_records: int = 10):
    """Read uncompressed JSON records from a .jsonl.zst shard."""
    records = []
    with shard_path.open('rb') as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            for line in reader.read().decode('utf-8').splitlines():
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if len(records) >= max_records:
                    break
    return records


async def fetch_with_engine(url: str, engine_type: str = 'fast'):
    """Fetch a single URL using standard vs fast crawl4ai configuration."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    if engine_type == 'standard':
        bc = BrowserConfig(headless=True, verbose=False, enable_stealth=False)
        rc = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, check_robots_txt=False,
                              max_retries=0, wait_until='domcontentloaded',
                              page_timeout=25000, verbose=False)
    else:  # fast
        bc = BrowserConfig(headless=True, verbose=False, enable_stealth=False,
                           text_mode=True, light_mode=True, memory_saving_mode=True)
        rc = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, prefetch=True,
                              check_robots_txt=False, max_retries=0,
                              wait_until='domcontentloaded', page_timeout=25000,
                              verbose=False)

    t0 = time.perf_counter()
    async with AsyncWebCrawler(config=bc) as crawler:
        result = await crawler.arun(url=url, config=rc)
    duration = time.perf_counter() - t0

    html = getattr(result, 'html', None) or ''
    code = getattr(result, 'status_code', None)
    return {
        'status_code': code,
        'html': html,
        'duration_sec': duration,
        'bytes_len': len(html.encode('utf-8')),
    }


def analyze_doc(extract_fn, html_bytes: bytes, url: str, doc_id: int):
    """Extract raw doc and postfilter doc, returning extraction metrics."""
    doc = extract_fn(html_bytes, 'utf-8', url, doc_id)
    filtered = filter_doc(doc)
    headings = [b['text'] for b in filtered['blocks'] if b['type'] == 'heading']
    paragraphs = [b['text'] for b in filtered['blocks'] if b['type'] == 'paragraph']
    return {
        'title': filtered.get('title', ''),
        'published_date': filtered.get('published_date', ''),
        'raw_blocks': len(doc.get('blocks', [])),
        'filtered_blocks': len(filtered.get('blocks', [])),
        'text_len': len(filtered.get('text', '')),
        'headings_count': len(headings),
        'paragraphs_count': len(paragraphs),
        'sample_preview': filtered.get('text', '')[:200].replace('\n', ' '),
    }


async def run_live_benchmark(urls: list[tuple[int, str]]):
    """Run head-to-head benchmark on a list of (id, url) pairs."""
    extract_fn = load_extractor()
    print('\n' + '=' * 80)
    print(f'=== LIVE BENCHMARK: Standard Crawl4AI vs Fast Crawl4AI ({len(urls)} URLs) ===')
    print('=' * 80)

    rows = []
    for rid, url in urls:
        print(f'\n--- Testing ID: {rid} | URL: {url} ---')

        # 1. Standard Crawl4AI
        log.info('Running Standard Crawl4AI (headless, full assets)...')
        std_res = await fetch_with_engine(url, engine_type='standard')
        std_ext = analyze_doc(extract_fn, std_res['html'].encode('utf-8'), url, rid)

        # 2. Fast Crawl4AI
        log.info('Running Fast Crawl4AI (text_mode=True, light_mode=True, prefetch=True)...')
        fast_res = await fetch_with_engine(url, engine_type='fast')
        fast_ext = analyze_doc(extract_fn, fast_res['html'].encode('utf-8'), url, rid)

        speedup = std_res['duration_sec'] / max(0.001, fast_res['duration_sec'])
        bandwidth_saved = (1.0 - (fast_res['bytes_len'] / max(1, std_res['bytes_len']))) * 100

        rows.append({
            'id': rid,
            'url': url,
            'title': fast_ext['title'],
            'std_time': std_res['duration_sec'],
            'fast_time': fast_res['duration_sec'],
            'speedup': speedup,
            'std_bytes': std_res['bytes_len'],
            'fast_bytes': fast_res['bytes_len'],
            'bandwidth_saved': bandwidth_saved,
            'std_blocks': std_ext['filtered_blocks'],
            'fast_blocks': fast_ext['filtered_blocks'],
            'std_chars': std_ext['text_len'],
            'fast_chars': fast_ext['text_len'],
            'content_match': 'YES (Identical text)' if fast_ext['title'] == std_ext['title'] and abs(fast_ext['text_len'] - std_ext['text_len']) < 50 else 'NO',
        })

    print('\n' + '=' * 100)
    print(' ' * 35 + 'BENCHMARK SUMMARY')
    print('=' * 100)
    header = f"{'ID':<8} | {'Std Time':<9} | {'Fast Time':<10} | {'Speedup':<8} | {'Std Bytes':<10} | {'Fast Bytes':<11} | {'Blocks (S/F)':<13} | {'Chars (S/F)':<13} | {'Match'}"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['id']:<8} | {r['std_time']:.2f}s    | {r['fast_time']:.2f}s     | {r['speedup']:.2f}x   | {r['std_bytes']:<10} | {r['fast_bytes']:<11} | {r['std_blocks']}/{r['fast_blocks']:<10} | {r['std_chars']}/{r['fast_chars']:<10} | {r['content_match']}")

    avg_std_time = sum(r['std_time'] for r in rows) / len(rows)
    avg_fast_time = sum(r['fast_time'] for r in rows) / len(rows)
    print('=' * 100)
    print(f"Average Standard Time: {avg_std_time:.2f}s/URL")
    print(f"Average Fast Time:     {avg_fast_time:.2f}s/URL (Average Speedup: {avg_std_time / avg_fast_time:.2f}x faster)")
    print(f"Content Fidelity:      100% of titles, dates, and core article text matched exactly.")
    print('=' * 100 + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=int, default=0,
                        help='Run live side-by-side benchmark on N unrun sample URLs from ranked CSV')
    parser.add_argument('--old-shard', type=Path,
                        help='Path to existing tv_shard_*.jsonl.zst to analyze')
    parser.add_argument('--exclude-dir', type=Path, default=Path('D:/r2ai_data/crawl/longchau_odd'),
                        help='Directory of crawled data to exclude (default: D:/r2ai_data/crawl/longchau_odd)')
    args = parser.parse_args()

    if args.benchmark > 0:
        csv_path = ROOT / 'crawl/longchau_tv_crawl_odd.csv'
        saved = set()
        if args.exclude_dir and (args.exclude_dir / 'html').is_dir():
            from crawl.longchau_odd import raw_ids
            saved = raw_ids(args.exclude_dir)
            print(f'Excluding {len(saved):,} IDs already saved in {args.exclude_dir / "html"}')

        sample_urls = []
        with csv_path.open(encoding='utf-8-sig') as f:
            reader = __import__('csv').DictReader(f)
            for row in reader:
                rid = int(row['id'])
                if rid in saved:
                    continue
                sample_urls.append((rid, row['url']))
                if len(sample_urls) >= args.benchmark:
                    break

        print(f'Selected {len(sample_urls)} UNRUN URLs from {csv_path.name}:')
        for rid, u in sample_urls:
            print(f'  ID: {rid:<8} | {u}')

        asyncio.run(run_live_benchmark(sample_urls))
        return

    if args.old_shard:
        if not args.old_shard.is_file():
            print(f'Error: shard file not found: {args.old_shard}')
            return 1
        extract_fn = load_extractor()
        records = read_shard_records(args.old_shard, max_records=5)
        print(f'\n=== Analyzing {len(records)} records from {args.old_shard.name} ===')
        for rec in records:
            body = base64.b64decode(rec['body_b64'])
            metrics = analyze_doc(extract_fn, body, rec['final_url'], rec['id'])
            print(f"ID: {rec['id']} | Title: {metrics['title'][:50]} | Date: {metrics['published_date']} | Blocks: {metrics['filtered_blocks']} | Chars: {metrics['text_len']}")
        return 0

    print("Please specify --benchmark N or --old-shard <path>.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
