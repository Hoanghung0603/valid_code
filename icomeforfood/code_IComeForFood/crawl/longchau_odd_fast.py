"""Collect raw Long Chau HTML for odd IDs using the fast Crawl4AI engine.

Standalone high-performance crawler:
- Disables image and font downloads in Chromium (text_mode=True)
- Disables background Chromium overhead (light_mode=True, memory_saving_mode=True)
- Bypasses Crawl4AI internal markdown/table parsing (scraping_strategy=None)
- Preserves 100% of the raw HTML snapshot needed by downstream extractors
- Retains safety rules: DB read-only snapshot, odd-only IDs, robots.txt check,
  and instant stop on block / challenge with 10-minute cooldown.
- Default output directory: D:/r2ai_data/crawl/longchau_odd_fast (isolated from the standard run).
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from urllib.parse import urlparse

import duckdb
import zstandard as zstd
from charset_normalizer import from_bytes

try:
    from crawl.db_file_lock import db_file_lock
except ModuleNotFoundError:  # python crawl/longchau_odd_fast.py
    from db_file_lock import db_file_lock

DOMAIN = 'nhathuoclongchau.com.vn'
DEFAULT_BLOCK_COOLDOWN_MINUTES = 10
COOLDOWN_MINUTES = 10
MIN_DELAY = 1.0 / 0.30
DEFAULT_DELAY = MIN_DELAY
MAX_BATCH = 100
STATUS_COLUMNS = ['id', 'url', 'final_url', 'lang_expected', 'status', 'http_code', 'content_type',
                  'n_bytes', 'attempts', 'fetched_at', 'shard', 'sha1_raw', 'error_msg', 'shard_file']
CHALLENGE_MARKERS = ('cf-chl-', 'challenge-platform', 'just a moment', 'verify you are human',
                     'checking your browser', 'captcha', 'verify.html',
                     'sorry, you have been blocked', 'access denied')
log = logging.getLogger('longchau_odd_fast')


def now():
    return datetime.now(timezone.utc)


def timestamp(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def is_longchau(url):
    return (urlparse(str(url)).hostname or '').lower().removeprefix('www.') == DOMAIN


def encoding(raw, content_type):
    match = __import__('re').search(r'charset=["\s]*([\w-]+)', content_type or '', __import__('re').I)
    if match:
        return match.group(1)
    best = from_bytes(raw[:32768]).best()
    return best.encoding if best else 'utf-8'


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def last_status(path: Path):
    rows = {}
    if path.is_file():
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if (row.get('id', '').isdigit() and row.get('status') and None not in row
                        and all(value is not None for value in row.values())):
                    rows[int(row['id'])] = row
    return rows


def source_rows(corpus=None, database=None):
    """Read DB-backed odd IDs, optionally in the order of a ranked CSV."""
    if database is None:
        if corpus is None:
            raise ValueError('A source --status-db is required')
        database = corpus
    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f'Source status DB not found: {database}')
    rows = []
    statuses = {}
    source_urls = {}
    with tempfile.TemporaryDirectory(prefix='longchau_odd_status_') as directory:
        copy = Path(directory) / 'status.duckdb'
        try:
            with db_file_lock(database):
                shutil.copy2(database, copy)
                wal = Path(str(database) + '.wal')
                if wal.is_file():
                    shutil.copy2(wal, Path(str(copy) + '.wal'))
        except PermissionError as exc:
            raise RuntimeError('Close the writer using status.duckdb before reading a snapshot.') from exc
        con = duckdb.connect(str(copy), read_only=True)
        try:
            for rid, url, state, code, fetched in con.execute("""SELECT id,url,status,http_code,fetched_at
                FROM crawl_status WHERE
                regexp_replace(lower(split_part(url,'/',3)), '^www[.]', '')=?
                ORDER BY id""", [DOMAIN]).fetchall():
                statuses[int(rid)] = dict(status=state, http_code=code, fetched_at=str(fetched) if fetched else None)
                if int(rid) % 2 == 1 and is_longchau(url):
                    rows.append(dict(id=int(rid), url=str(url)))
                    source_urls[int(rid)] = str(url)
        finally:
            con.close()
    if corpus is not None and Path(corpus).suffix.lower() == '.csv':
        ranked = []
        seen = set()
        with Path(corpus).open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if not {'id', 'url'} <= set(reader.fieldnames or ()):
                raise ValueError('Ranked CSV must contain id,url columns')
            for entry in reader:
                rid = int(entry['id'])
                if rid % 2 == 0:
                    raise ValueError(f'Ranked CSV contains an even ID: {rid}')
                url = entry['url']
                if rid in seen:
                    raise ValueError(f'Duplicate odd ID in ranked CSV: {rid}')
                if rid not in source_urls or source_urls[rid] != url:
                    raise ValueError(f'Ranked CSV ID/URL differs from Long Chau source DB: {rid}')
                ranked.append(dict(id=rid, url=url))
                seen.add(rid)
        rows = ranked
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('crawl_status contains duplicate Long Chau IDs')
    return rows, statuses


def raw_ids(out: Path):
    found = set()
    paths = list((out / 'html').glob('home_shard_*.jsonl.zst'))
    paths += list((out / 'html').glob('tv_shard_*.jsonl.zst'))
    for path in sorted(paths):
        with path.open('rb') as compressed, zstd.ZstdDecompressor().stream_reader(compressed) as reader, io.TextIOWrapper(reader, encoding='utf-8') as lines:
            for line in lines:
                record = json.loads(line)
                rid = int(record['id'])
                if rid % 2 != 1:
                    raise ValueError(f'Even ID in odd-only raw output: {rid}')
                base64.b64decode(record['body_b64'], validate=True)
                found.add(rid)
    return found


def queue_rows(rows, database_rows, local_rows, saved_raw_ids, retry_blocked=False):
    skip_states = {'ok', 'non_html', 'empty'}
    skip_codes = {'404', '410'}
    if not retry_blocked:
        skip_states.add('blocked')
        skip_codes |= {'401', '403', '429'}
    selected = []
    for row in rows:
        rid = row['id']
        if rid not in database_rows or rid % 2 != 1 or not is_longchau(row['url']):
            continue
        if database_rows[rid].get('status') in {'ok', 'non_html', 'empty'}:
            continue
        status = local_rows.get(rid) or database_rows.get(rid) or {}
        code = str(status.get('http_code') or '')
        if rid in saved_raw_ids or status.get('status') in skip_states or code in skip_codes:
            continue
        selected.append(row)
    return selected


def initial_state(rows, database_rows, local_rows, cooldown_minutes=DEFAULT_BLOCK_COOLDOWN_MINUTES):
    by_id = {r['id']: r for r in rows}
    blocks = []
    for rid, row in {**database_rows, **local_rows}.items():
        if row.get('status') == 'blocked' or str(row.get('http_code') or '') in {'401', '403', '429'}:
            stamp = timestamp(row.get('fetched_at'))
            if stamp:
                blocks.append((stamp, rid))
    if not blocks:
        return dict(kind='unprobed', next_probe_at=None)
    stamp, rid = max(blocks)
    candidate = by_id.get(rid) or (rows[0] if rows else None)
    if candidate is None:
        return dict(kind='unprobed', next_probe_at=None)
    return dict(kind='blocked', blocked_at=stamp.isoformat(),
                next_probe_at=(stamp + timedelta(minutes=cooldown_minutes)).isoformat(),
                blocked_id=candidate['id'], blocked_url=candidate['url'],
                reason='Existing blocked status; do not reset or retry continuously')


def gate(state, probe, current=None, scan=False):
    current = current or now()
    deadline = timestamp(state.get('next_probe_at'))
    if deadline and current < deadline:
        return f'Cooldown active until {deadline.astimezone().isoformat()}; zero requests allowed.'
    if state.get('kind') in {'blocked', 'unprobed'} and not (probe or scan):
        return 'Use --scan, or a single --probe before --run. Blocked IDs are never reset to pending.'
    return None


def home_cooldown_state(state, out: Path, cooldown_minutes: int):
    """Honor a recent block from the home crawler on the same Long Chau host."""
    home_status = out.parent / 'crawl_home_tv' / 'status.csv'
    blocks = []
    for rid, row in last_status(home_status).items():
        if not is_longchau(row.get('url')) or row.get('status') != 'blocked':
            continue
        try:
            blocked_at = timestamp(row.get('fetched_at'))
        except ValueError:
            continue
        if blocked_at:
            blocks.append((blocked_at, rid, row['url']))
    if not blocks:
        return state
    blocked_at, rid, url = max(blocks)
    deadline = blocked_at + timedelta(minutes=cooldown_minutes)
    if deadline <= now() or deadline <= (timestamp(state.get('next_probe_at')) or datetime.min.replace(tzinfo=timezone.utc)):
        return state
    return dict(kind='blocked', engine='crawl_home_tv', blocked_at=blocked_at.isoformat(),
                next_probe_at=deadline.isoformat(), blocked_id=rid, blocked_url=url,
                reason='Recent block from crawl_home_tv; wait before trying Crawl4AI')


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another Long Chau job is holding {path}.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def keep_system_awake():
    """Keep a long local crawl running while the Windows machine is on."""
    if os.name != 'nt':
        yield
        return
    import ctypes
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    try:
        yield
    finally:
        set_state(0x80000000)  # ES_CONTINUOUS


class RawOutput:
    def __init__(self, out: Path):
        self.out = out
        self.html = out / 'html'
        self.html.mkdir(parents=True, exist_ok=True)
        for tmp in self.html.glob('*.tmp'):
            try:
                tmp.unlink()
            except OSError:
                pass
        self.pending = []
        self.idx = max((int(p.name.split('_')[2].split('.')[0]) for p in self.html.glob('tv_shard_*.jsonl.zst')), default=-1) + 1
        path = out / 'status.csv'
        new = not path.exists() or path.stat().st_size == 0
        cut = not new and not path.read_bytes().endswith(b'\n')
        self.handle = path.open('a', newline='', encoding='utf-8')
        if cut:
            self.handle.write('\n')
        self.writer = csv.DictWriter(self.handle, fieldnames=STATUS_COLUMNS)
        if new:
            self.writer.writeheader()
            self.sync()

    def sync(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def add_status(self, row):
        self.writer.writerow(row)
        self.sync()

    def add_raw(self, record, row):
        if int(record['id']) % 2 != 1 or int(row['id']) % 2 != 1:
            raise ValueError('Only odd source IDs may be saved')
        self.pending.append((record, row))

    def flush(self):
        if not self.pending:
            return
        path = self.html / f'tv_shard_{self.idx:05d}.jsonl.zst'
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_name(path.name + '.tmp')
        raw = ''.join(json.dumps(rec, ensure_ascii=False) + '\n' for rec, _ in self.pending).encode('utf-8')
        with temp.open('wb') as handle:
            handle.write(zstd.ZstdCompressor(level=3).compress(raw))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        for _, row in self.pending:
            row['shard_file'] = path.name
            self.writer.writerow(row)
        self.sync()
        self.pending.clear()
        self.idx += 1

    def close(self):
        try:
            self.flush()
        finally:
            self.handle.close()


class DbSync:
    """Write crawl results back to the shared DuckDB crawl_status table.

    Opens a write connection only for each batch, then closes it.
    Each row is an UPDATE on the existing ``id`` — we never INSERT new rows
    because the source of truth for which IDs exist is the DB itself.

    Safe rules:
    - Only updates rows whose current status is NOT 'ok'/'non_html'/'empty'.
    - A shared file lock serializes DB access with the retry crawler.
    - On DB error, pending rows remain queued and staged output stays on disk.
    """

    def __init__(self, db_path: Path, batch_size: int = 50):
        self.db_path = db_path
        self.batch_size = batch_size
        self._pending: list[dict] = []

    def push(self, row: dict):
        """Queue a result until its raw shard has been durably flushed."""
        self._pending.append(row)

    def flush(self):
        """Write all pending rows to DB in a single transaction."""
        if not self._pending:
            return
        try:
            with db_file_lock(self.db_path):
                con = duckdb.connect(str(self.db_path))
                try:
                    con.begin()
                    for row in self._pending:
                        fetched = row.get('fetched_at')
                        con.execute(
                            """UPDATE crawl_status
                               SET status      = ?,
                                   final_url   = ?,
                                   http_code   = ?,
                                   content_type= ?,
                                   n_bytes     = ?,
                                   attempts    = COALESCE(attempts, 0) + 1,
                                   fetched_at  = ?::TIMESTAMP,
                                   sha1_raw    = ?,
                                   error_msg   = ?
                               WHERE id = ?
                                 AND COALESCE(status, 'pending') NOT IN ('ok', 'non_html', 'empty')
                            """,
                            [
                                row.get('status'),
                                row.get('final_url'),
                                row.get('http_code'),
                                row.get('content_type'),
                                row.get('n_bytes'),
                                fetched[:26] if fetched else None,
                                row.get('sha1_raw'),
                                row.get('error_msg'),
                                int(row['id']),
                            ],
                        )
                    con.commit()
                except Exception:
                    con.rollback()
                    raise
                finally:
                    con.close()
            log.debug('DB sync: flushed %d rows', len(self._pending))
            self._pending.clear()
        except Exception as exc:
            log.warning('DB sync flush failed (%s); %d staged rows await retry',
                        exc, len(self._pending))

    def close(self):
        self.flush()


def wait_for_rest(stop_file: Path, seconds: float):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_file.exists():
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return not stop_file.exists()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument('--corpus', type=Path, default=repo / 'crawl/longchau_tv_crawl_odd.csv',
                        help='TV priority id,url CSV (default: crawl/longchau_tv_crawl_odd.csv)')
    default_db = Path('D:/r2ai_data/crawl/status.duckdb')
    parser.add_argument('--status-db', type=Path, default=Path(os.environ.get('R2AI_STATUS_DB', str(default_db))),
                        help='Existing DuckDB file containing crawl_status (default: D:/r2ai_data/crawl/status.duckdb)')
    parser.add_argument('--retry-blocked', action='store_true',
                        help='Include DB/local blocked odd IDs after cooldown; stop again on the first block')
    parser.add_argument('--sample-first', type=int, default=0,
                        help='Fetch the first N ranked CSV rows, even if already saved, as one tv shard (1-50)')
    parser.add_argument('--trial-urls', type=int, default=0,
                        help='Limit a ranked Crawl4AI scan to the next N eligible URLs (1-150)')
    parser.add_argument('--exclude-dir', type=Path, default=Path('D:/r2ai_data/crawl/longchau_odd'),
                        help='Directory of previous runs to exclude saved raw IDs from (default: D:/r2ai_data/crawl/longchau_odd)')
    parser.add_argument('--out', type=Path, default=Path('D:/r2ai_data/crawl/longchau_odd_fast'),
                        help='Output directory (default: D:/r2ai_data/crawl/longchau_odd_fast)')
    parser.add_argument('--delay', type=float, default=DEFAULT_DELAY,
                        help='Extra minimum post-response delay; omit to reach the 0.30 requests/s cap')
    parser.add_argument('--block-cooldown-minutes', type=int, default=DEFAULT_BLOCK_COOLDOWN_MINUTES,
                        help='Minimum wait after a block before one probe (default: 10 minutes)')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--continuous', action='store_true',
                        help='Save each batch without a fixed 3-hour rest')
    parser.add_argument('--no-jitter', dest='jitter', action='store_false',
                        help='Disable the additional 0-8%% wait after each response')
    parser.set_defaults(jitter=True)

    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--probe', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--scan', action='store_true',
                        help='Scan remaining odd IDs without a separate probe; retain cooldown and stop-on-block')
    args = parser.parse_args(argv)

    if args.delay < MIN_DELAY:
        parser.error(f'--delay must be at least {MIN_DELAY:.3f}s for safe crawling.')

    try:
        import crawl4ai  # noqa: F401
    except ImportError as exc:
        parser.error(f'Crawl4AI is missing from this Python environment: {exc}')

    args.status_db = args.status_db.expanduser().resolve()
    all_rows, database = source_rows(corpus=args.corpus, database=args.status_db)
    rows = all_rows
    local = last_status(args.out / 'status.csv')
    if args.exclude_dir and (args.exclude_dir / 'status.csv').is_file():
        prev_local = last_status(args.exclude_dir / 'status.csv')
        local = {**prev_local, **local}

    saved = raw_ids(args.out)
    for prev_name in ('longchau_odd', 'longchau_odd_raw'):
        prev_dir = args.out.parent / prev_name
        if prev_dir.is_dir():
            saved |= raw_ids(prev_dir)
            if (prev_dir / 'status.csv').is_file():
                local = {**last_status(prev_dir / 'status.csv'), **local}
    selected = (rows[:args.sample_first] if args.sample_first else
                queue_rows(rows, database, local, saved, retry_blocked=args.retry_blocked))
    if args.trial_urls:
        selected = selected[:args.trial_urls]

    state_path = args.out / 'cooldown.json'
    state = json.loads(state_path.read_text(encoding='utf-8-sig')) if state_path.is_file() else initial_state(rows, database, local, args.block_cooldown_minutes)
    state = home_cooldown_state(state, args.out, args.block_cooldown_minutes)

    print(f'=== Long Chau Fast Crawl4AI ===')
    print(f'Source DB: {args.status_db}')
    print(f'Corpus CSV: {args.corpus.resolve() if args.corpus else "None"}')
    print(f'CSV odd Long Chau IDs: {len(all_rows):,}')
    print(f'Output directory: {args.out.resolve()}')
    print(f'Rate ramp: 0.05 -> 0.30 requests/s, +0.025 every 20 minutes; '
          f'minimum delay floor: {args.delay:.3f}s; positive jitter: {args.jitter}')
    print(f'Batch size: {args.batch_size}')
    print(f'Eligible URLs: {len(selected):,} (Already saved here: {len(saved):,})')

    if not args.probe and not args.run and not args.scan:
        print('Offline plan only: zero HTTP requests.')
        return 0

    restriction = gate(state, args.probe, scan=args.scan)
    if restriction:
        print(f'Restriction: {restriction}')
        return 2

    if not selected:
        print('No eligible odd IDs left; zero requests.')
        return 0

    import longchau_crawl4ai_fast
    import asyncio

    lock_file = args.out.parent / 'locks/nhathuoclongchau_fast.lock'
    with exclusive_lock(lock_file), keep_system_awake():
        args.out.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout),
                                      logging.FileHandler(args.out / 'crawl.log', encoding='utf-8')])
        db_sync = DbSync(args.status_db, batch_size=50)
        try:
            summary = asyncio.run(longchau_crawl4ai_fast.collect(
                selected, args.out, state, args, base=sys.modules[__name__], db_sync=db_sync,
            ))
        finally:
            # collect() closes its raw shard first; commit staged statuses even
            # when a later browser error interrupts the run.
            db_sync.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary['cooldown'].get('kind') == 'blocked':
            return 3
        return 4 if args.probe and summary['ok'] != 1 else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Stopped; completed raw shards and status rows were saved.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
