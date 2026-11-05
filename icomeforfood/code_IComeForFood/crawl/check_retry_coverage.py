"""Audit local crawl coverage outside Long Chau and zysjonline."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import duckdb

try:
    from crawl.db_file_lock import db_file_lock
except ModuleNotFoundError:
    from db_file_lock import db_file_lock


ROOT = Path('D:/r2ai_data/crawl')
DATABASE = ROOT / 'status.duckdb'
HTML = ROOT / 'html'
STAGED = ROOT / 'retry_crawl4ai'
SCOPE = """regexp_replace(lower(split_part(url, '/', 3)), '^www[.]', '')
           NOT IN ('nhathuoclongchau.com.vn', 'zysjonline.com')"""


def latest_stage() -> dict[int, dict]:
    rows = {}
    with (STAGED / 'status.csv').open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if (row.get('id') or '').isdigit() and row.get('status'):
                rows[int(row['id'])] = row
    return rows


def main() -> None:
    stage = latest_stage()
    durable = {rid: row for rid, row in stage.items()
               if row['status'] in {'ok', 'non_html'}
               and row.get('shard_file')
               and (STAGED / 'html' / row['shard_file']).is_file()}
    with db_file_lock(DATABASE, timeout=30):
        con = duckdb.connect(str(DATABASE), read_only=True)
        try:
            total = con.execute(f'SELECT count(*) FROM crawl_status WHERE {SCOPE}').fetchone()[0]
            statuses = con.execute(f'''SELECT coalesce(status, 'NULL'), count(*)
                FROM crawl_status WHERE {SCOPE} GROUP BY 1 ORDER BY 2 DESC''').fetchall()
            shards = con.execute(f'''SELECT shard, count(*) FROM crawl_status
                WHERE {SCOPE} AND status IN ('ok', 'non_html') AND shard IS NOT NULL
                GROUP BY shard''').fetchall()
            pending = con.execute(f'''SELECT id,url,status FROM crawl_status
                WHERE {SCOPE} AND (status NOT IN ('ok','non_html') OR status IS NULL
                    OR shard IS NULL)''').fetchall()
            indexed_pubmed = (con.execute('''SELECT idx.id,idx.batch_file
                FROM pubmed_cache_index AS idx
                JOIN crawl_status AS cs ON cs.id=idx.id
                WHERE cs.status='ok' ''').fetchall()
                if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='pubmed_cache_index'").fetchone()[0]
                else [])
        finally:
            con.close()

    present = sum(count for shard, count in shards
                  if (HTML / f'shard_{int(shard):05d}.jsonl.zst').is_file())
    missing_files = [(int(shard), int(count)) for shard, count in shards
                     if not (HTML / f'shard_{int(shard):05d}.jsonl.zst').is_file()]
    staged_pending = set()
    attempted_in_stage = set()
    by_host = Counter()
    pubmed_available = {int(rid) for rid, batch_file in indexed_pubmed
                        if (ROOT / 'pubmed' / batch_file).is_file()}
    for rid, url, _ in pending:
        rid = int(rid)
        if rid in pubmed_available:
            continue  # PubMed API records live in batches, not HTML shards.
        prior = stage.get(rid)
        if prior and prior['url'] == url:
            attempted_in_stage.add(rid)
        row = durable.get(rid)
        if row and row['url'] == url:
            staged_pending.add(rid)
        else:
            by_host[(urlparse(url).hostname or '').lower().removeprefix('www.')] += 1

    print(f'Tổng URL ngoài Long Châu và zysjonline: {total:,}')
    print('Trạng thái DB: ' + ', '.join(f'{status}={count:,}' for status, count in statuses))
    print(f'Đã có shard chính tại máy: {present:,}')
    print(f'PubMed API đã đồng bộ và có batch tại máy: {len(pubmed_available):,}')
    print(f'Đã lưu ở retry staging, chưa vào shard chính: {len(staged_pending):,}')
    print(f'Còn thiếu dữ liệu crawl: {sum(by_host.values()):,}')
    print(f'Trong số còn thiếu, đã thử ở retry staging: {len(attempted_in_stage):,}')
    print(f'Shard chính được DB tham chiếu nhưng thiếu file: '
          f'{len(missing_files):,} shard / {sum(count for _, count in missing_files):,} ID')
    print('Host còn thiếu nhiều nhất:')
    for host, count in by_host.most_common(12):
        print(f'  {host}: {count:,}')


if __name__ == '__main__':
    main()
