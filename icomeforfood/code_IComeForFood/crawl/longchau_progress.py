"""Read-only progress monitor for the Long Chau odd-ID crawler.

Run once or use --watch SECONDS to refresh while the crawler is running.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import sys
import time

import duckdb

try:
    from crawl.db_file_lock import db_file_lock
except ModuleNotFoundError:  # python crawl/longchau_progress.py
    from db_file_lock import db_file_lock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path('D:/r2ai_data/crawl')
LOG_LINE = re.compile(
    r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO '
    r'id=(\d+) status=(\w+) .* completed=(\d+)\s*$'
)


def ranked_ids(path: Path) -> set[int]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return {int(row['id']) for row in csv.DictReader(handle)
                if (row.get('id') or '').isdigit() and int(row['id']) % 2 == 1}


def saved_ids(path: Path, wanted: set[int]) -> tuple[set[int], int]:
    latest: dict[int, tuple[str, str]] = {}
    if path.is_file():
        with path.open(newline='', encoding='utf-8-sig', errors='replace') as handle:
            for row in csv.DictReader(handle):
                value = row.get('id') or ''
                if value.isdigit() and int(value) in wanted:
                    latest[int(value)] = (row.get('status') or '', row.get('shard_file') or '')
    ok = {rid for rid, (status, shard) in latest.items() if status == 'ok' and shard}
    failed = sum(status not in {'ok', 'non_html'} for status, _ in latest.values())
    return ok, failed


def db_main_ids(database: Path, wanted: set[int]) -> set[int]:
    with db_file_lock(database, timeout=3):
        con = duckdb.connect(str(database), read_only=True)
        try:
            rows = con.execute("""SELECT id FROM crawl_status
                WHERE id % 2 = 1 AND status = 'ok' AND shard IS NOT NULL
                AND regexp_replace(lower(split_part(url,'/',3)), '^www[.]', '') =
                    'nhathuoclongchau.com.vn'""").fetchall()
        finally:
            con.close()
    return {int(row[0]) for row in rows if int(row[0]) in wanted}


def current_run(log_path: Path) -> tuple[int, int, float | None, float | None]:
    entries: list[tuple[datetime, str, int]] = []
    if log_path.is_file():
        with log_path.open(encoding='utf-8', errors='replace') as handle:
            for line in handle:
                match = LOG_LINE.match(line)
                if match:
                    entries.append((datetime.strptime(match[1], '%Y-%m-%d %H:%M:%S'),
                                    match[3], int(match[4])))
    if not entries:
        return 0, 0, None, None
    start = max((index for index, entry in enumerate(entries) if entry[2] == 1), default=0)
    entries = entries[start:]
    first, last = entries[0][0], entries[-1][0]
    attempted = entries[-1][2]
    ok = sum(status == 'ok' for _, status, _ in entries)
    elapsed_hours = max((last - first).total_seconds() / 3600, 1 / 3600)
    rate = ok / elapsed_hours if len(entries) > 1 else None
    age = max((datetime.now() - last).total_seconds(), 0)
    return attempted, ok, rate, age


def latest_log(data: Path) -> Path:
    candidates = [data / 'longchau_odd_fast/crawl.log', data / 'crawl_home_tv/crawl.log']
    available = [path for path in candidates if path.is_file()]
    return max(available, key=lambda path: path.stat().st_mtime) if available else candidates[0]


def session_report(data: Path) -> None:
    log_path = latest_log(data)
    attempted, ok, rate, age = current_run(log_path)
    rate_text = f'{rate:,.0f}' if rate is not None else 'chưa đủ dữ liệu'
    age_text = f'{age:.0f}s' if age is not None else 'chưa có log'
    print(f'[{datetime.now():%H:%M:%S}] Phiên Long Châu ({log_path.parent.name}): {ok:,} trang OK '
          f'({attempted:,} lượt) | TB {rate_text} trang/giờ | Log cuối: {age_text}',
          flush=True)


def report(data: Path, database: Path, wanted: set[int],
           db_cache: set[int] | None) -> set[int] | None:
    fast_ok, fast_failed = saved_ids(data / 'longchau_odd_fast/status.csv', wanted)
    tv_ok, tv_failed = saved_ids(data / 'crawl_home_tv/status.csv', wanted)
    staged = set(fast_ok)
    for folder in ('longchau_odd', 'longchau_odd_raw', 'crawl_home_tv'):
        other, _ = saved_ids(data / folder / 'status.csv', wanted)
        staged.update(other)
    try:
        db_cache = db_main_ids(database, wanted)
        db_text = f'{len(db_cache):,}'
    except (OSError, RuntimeError, TimeoutError, duckdb.Error):
        db_text = f'{len(db_cache):,} (lần đọc trước)' if db_cache is not None else 'đang bận'
    complete = staged | (db_cache or set())
    attempted, run_ok, rate, age = current_run(latest_log(data))
    progress = 100 * len(complete) / len(wanted) if wanted else 0
    rate_text = f'{rate:,.0f} trang/giờ' if rate is not None else 'chưa đủ dữ liệu'
    age_text = f'{age:.0f}s' if age is not None else 'chưa có log'
    print(f'[{datetime.now():%H:%M:%S}] Long Châu odd: {len(complete):,}/{len(wanted):,} '
          f'({progress:.1f}%), còn {len(wanted) - len(complete):,}')
    print(f'  DB có shard chính: {db_text} | Fast: {len(fast_ok):,} OK, {fast_failed:,} lỗi '
          f'| TV: {len(tv_ok):,} OK, {tv_failed:,} lỗi '
          f'| Phiên log mới nhất: {attempted:,} lượt, {run_ok:,} OK')
    print(f'  Tốc độ TB phiên: {rate_text} | Log cuối: cách {age_text}', flush=True)
    return db_cache


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--db', type=Path, default=DEFAULT_DATA / 'status.duckdb')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'crawl/longchau_tv_crawl_odd.csv')
    parser.add_argument('--watch', type=float, default=0, metavar='SECONDS',
                        help='Refresh continuously every N seconds (default: print once)')
    parser.add_argument('--session-only', action='store_true',
                        help='Show only OK pages and average pages/hour in the latest crawl session')
    args = parser.parse_args()
    if args.watch < 0:
        parser.error('--watch must be nonnegative')
    wanted = ranked_ids(args.corpus) if not args.session_only else set()
    cached = None
    try:
        while True:
            if args.session_only:
                session_report(args.data)
            else:
                cached = report(args.data, args.db, wanted, cached)
            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
