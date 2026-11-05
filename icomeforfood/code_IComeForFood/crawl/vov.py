"""DB-backed VOV Chrome crawler: offline plan, probe, batch, scan and resume.

Uses the Long Chau checkpoint/status format, but collects all VOV source IDs
and keeps the original HTTP response bytes. The source DB is never written.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
import csv
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from urllib.parse import urlsplit

import duckdb
from bs4 import BeautifulSoup
import zstandard as zstd

try:
    from crawl.longchau_odd import atomic_json, now, timestamp, last_status, keep_system_awake, encoding, STATUS_COLUMNS
except ModuleNotFoundError:  # python crawl/vov.py
    from longchau_odd import atomic_json, now, timestamp, last_status, keep_system_awake, encoding, STATUS_COLUMNS

DOMAIN = 'vov.vn'
MIN_DELAY = 2.0
COOLDOWN_MINUTES = 10
log = logging.getLogger('vov')


def is_vov(url):
    try:
        parts = urlsplit(str(url))
        return (parts.scheme in {'http', 'https'} and parts.hostname in {DOMAIN, 'www.' + DOMAIN}
                and parts.port in {None, 80, 443} and parts.username is None and parts.password is None)
    except ValueError:
        return False


def source_rows(database):
    """Read a snapshot of the selected existing DB, without parity filtering."""
    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f'Source status DB not found: {database}')
    with tempfile.TemporaryDirectory(prefix='vov_status_') as directory:
        copy = Path(directory) / 'status.duckdb'
        try:
            shutil.copy2(database, copy)
            wal = Path(str(database) + '.wal')
            if wal.exists():
                shutil.copy2(wal, Path(str(copy) + '.wal'))
        except PermissionError as exc:
            raise RuntimeError('Close the DB writer before taking a status snapshot.') from exc
        with duckdb.connect(str(copy), read_only=True) as con:
            records = con.execute("""SELECT id,url,status,http_code,attempts,fetched_at
                FROM crawl_status WHERE
                regexp_replace(lower(split_part(url,'/',3)), '^www[.]', '')=? ORDER BY id""",
                [DOMAIN]).fetchall()
    rows = []
    for rid, url, status, code, attempts, fetched in records:
        if not is_vov(url):
            raise ValueError(f'Invalid VOV source URL for id={rid}: {url}')
        rows.append(dict(id=int(rid), url=url, status=status, http_code=code,
                         attempts=int(attempts or 0), fetched_at=str(fetched) if fetched else None))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    return rows


def classify(raw, code, headers, final_url):
    """Collect successful HTML as in Long Chau; extraction happens downstream.

    No article selectors, text-length thresholds or media requirements belong
    in the raw collector. Only refuse actual access challenges/failed responses.
    """
    headers = {str(k).lower(): str(v).lower() for k, v in headers.items()}
    if not is_vov(final_url):
        return 'http_error', 'Cross-domain redirect; stopped'
    if (code in {401, 403, 429, 503} or headers.get('cf-mitigated') == 'challenge'
            or 'verify.html' in urlsplit(final_url).path.lower()):
        return 'blocked', f'Access refused or verification response (HTTP {code})'
    if code != 200:
        return 'http_error', f'HTTP {code}; no retry'
    if 'html' not in headers.get('content-type', 'text/html'):
        return 'non_html', 'Non-HTML response'
    if not raw.strip():
        return 'http_error', 'HTTP 200 with an empty HTML response; stopped for inspection'
    soup = BeautifulSoup(raw, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'template']):
        tag.decompose()
    visible = soup.get_text(' ', strip=True).lower()
    title = soup.title.get_text(' ', strip=True).lower().strip(' .!') if soup.title else ''
    article = soup.find('article') or soup.select_one('meta[property="og:type"][content="article"]')
    prompts = ('just a moment', 'checking your browser', 'verify you are human',
               'access denied', 'sorry, you have been blocked', 'xác minh bạn là con người',
               'verify captcha', 'complete the captcha')
    if (title in prompts or soup.select_one('#challenge-form, #cf-challenge-running')
            or (not article and (any(marker in visible for marker in prompts)
                                 or soup.select_one('form[action*="captcha"]')))):
        return 'blocked', 'Verification page; domain stopped'
    return 'ok', None


def saved_records(out, by_id):
    """Validate closed shards, including orphan rows from an interrupted CSV write."""
    saved = {}
    for path in sorted((out / 'html').glob('home_shard_*.jsonl.zst')):
        with path.open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as reader, io.TextIOWrapper(reader, encoding='utf-8') as lines:
            for line in lines:
                rec = json.loads(line)
                rid = int(rec['id'])
                if rid not in by_id or not is_vov(rec['final_url']) or rid in saved:
                    raise ValueError(f'Unknown, duplicate or non-VOV raw ID: {rid}')
                raw = base64.b64decode(rec['body_b64'], validate=True)
                if classify(raw, rec['http_code'], {'content-type': rec['content_type']}, rec['final_url'])[0] != 'ok':
                    raise ValueError(f'Invalid article in saved shard: {rid}')
                saved[rid] = dict(id=rid, url=by_id[rid]['url'], final_url=rec['final_url'],
                                  lang_expected='vi', status='ok', http_code=200,
                                  content_type=rec['content_type'], n_bytes=len(raw),
                                  attempts=by_id[rid]['attempts'] + 1,
                                  fetched_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                                  shard=None, sha1_raw=hashlib.sha1(raw).hexdigest(), error_msg=None,
                                  shard_file=path.name)
    return saved


def queue_rows(rows, local, saved):
    selected = []
    for item in rows:
        rid = item['id']
        state = local.get(rid, item)
        if rid in saved or item['status'] in {'ok', 'non_html', 'empty'}:
            continue
        if str(item['http_code']) in {'404', '410'} or str(state.get('http_code')) in {'404', '410'}:
            continue
        # A local `ok` without a durable shard is not proof of completion.
        selected.append(dict(item, attempts=max(item['attempts'], int(state.get('attempts') or 0))))
    return selected


class RawOutput:
    """Same six-field shards and durable-then-status checkpoint as Long Chau."""
    def __init__(self, out):
        self.html = out / 'html'
        self.html.mkdir(parents=True, exist_ok=True)
        self.pending = []
        self.idx = max((int(p.name.split('_')[2].split('.')[0]) for p in self.html.glob('home_shard_*.jsonl.zst')), default=-1) + 1
        path = out / 'status.csv'
        new = not path.exists() or not path.stat().st_size
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
        self.pending.append((record, row))

    def flush(self):
        if not self.pending:
            return
        path = self.html / f'home_shard_{self.idx:05d}.jsonl.zst'
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


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if not handle.tell():
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
            raise RuntimeError('Another VOV job is already running.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def gate(state):
    deadline = timestamp(state.get('next_probe_at'))
    if deadline and now() < deadline:
        return f'VOV cooldown active until {deadline.isoformat()}; zero requests.'
    return None


def blocked_state(state, item, code, headers, reason):
    deadline = now() + timedelta(minutes=COOLDOWN_MINUTES)
    retry = next((str(v) for k, v in headers.items() if k.lower() == 'retry-after'), '')
    if retry:
        try:
            allowed = now() + timedelta(seconds=int(retry)) if retry.isdigit() else parsedate_to_datetime(retry)
            if allowed.tzinfo is None:
                allowed = allowed.replace(tzinfo=timezone.utc)
            deadline = max(deadline, allowed)
        except (ValueError, TypeError, OverflowError):
            pass
    return dict(kind='blocked', engine='chrome', blocked_id=item['id'], blocked_url=item['url'],
                blocked_at=now().isoformat(), next_probe_at=deadline.isoformat(),
                last_request_at=state.get('last_request_at'), http_code=code, reason=reason)


async def wait_until_ready(stop_file, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_file.exists():
            return False
        await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))
    return not stop_file.exists()


async def collect(rows, out, state, args, page_factory=None, recover=None):
    """One Chrome page, sequential navigation, no automatic retries."""
    if gate(state):
        raise RuntimeError(gate(state))
    writer = RawOutput(out)
    attempted = ok = 0
    reason = 'queue_finished'
    stop_file = out / 'STOP'
    last_save = time.monotonic()
    previous = timestamp(state.get('last_request_at'))
    wait_seconds = max(0, args.delay - (now() - previous).total_seconds()) if previous else 0
    selected = rows[:1] if args.probe else rows if args.scan else rows[:args.batch_size]
    if args.limit is not None:
        selected = selected[:args.limit]
    try:
        for row in (recover or {}).values():
            writer.add_status(row)
        ready = await wait_until_ready(stop_file, wait_seconds)
        if not ready:
            reason = 'STOP'
        async with AsyncExitStack() as stack:
            page = await stack.enter_async_context(page_factory() if page_factory else chrome_page(args.channel)) if selected and ready else None
            for item in selected if ready else []:
                if not is_vov(item['url']):
                    raise ValueError('Refusing a non-VOV source URL')
                if stop_file.exists():
                    reason = 'STOP'
                    break
                state['last_request_at'] = now().isoformat()
                atomic_json(out / 'cooldown.json', state)
                row = dict.fromkeys(STATUS_COLUMNS)
                row.update(id=item['id'], url=item['url'], final_url=item['url'], lang_expected='vi',
                           attempts=item['attempts'] + 1, fetched_at=now().isoformat(),
                           status='http_error', n_bytes=0)
                attempted += 1
                try:
                    response = await page.goto(item['url'], wait_until='domcontentloaded', timeout=60000)
                    raw = await response.body() if response else b''
                    headers = response.headers if response else {}
                    code = response.status if response else None
                    status, error = classify(raw, code, headers, page.url)
                    row.update(final_url=page.url, http_code=code, content_type=headers.get('content-type', ''),
                               n_bytes=len(raw), status=status, error_msg=error)
                except Exception as exc:
                    row.update(error_msg=f'{type(exc).__name__}: {exc}'[:500])
                    writer.add_status(row)
                    reason = 'navigation_error'
                    state.update(kind='unprobed', reason=reason)
                    break
                if status == 'ok':
                    row['sha1_raw'] = hashlib.sha1(raw).hexdigest()
                    record = dict(id=item['id'], final_url=page.url, http_code=200,
                                  content_type=row['content_type'], encoding=encoding(raw, row['content_type']),
                                  body_b64=base64.b64encode(raw).decode('ascii'))
                    writer.add_raw(record, row)
                    ok += 1
                    state = dict(kind='probe_ok' if args.probe else 'scan_ready', engine='chrome',
                                 last_request_at=state.get('last_request_at'),
                                 probed_at=now().isoformat(), next_probe_at=None)
                else:
                    if code == 200 and status == 'http_error':
                        inspection = out / 'inspection'
                        inspection.mkdir(exist_ok=True)
                        (inspection / f'{item["id"]}.html').write_bytes(raw)
                        atomic_json(inspection / f'{item["id"]}.json',
                                    dict(id=item['id'], url=item['url'], final_url=page.url,
                                         http_code=code, headers=headers, reason=error,
                                         fetched_at=row['fetched_at']))
                        row['error_msg'] += f'; HTML saved: {inspection / str(item["id"])}.html'
                    writer.add_status(row)
                    if status == 'blocked':
                        state = blocked_state(state, item, code, headers, error)
                    elif code not in {404, 410}:
                        state.update(kind='unprobed', reason=error)
                    if code not in {404, 410}:
                        reason = status
                        log.warning('VOV stopped: id=%s HTTP=%s %s', item['id'], code, error)
                        break
                atomic_json(out / 'cooldown.json', state)
                log.info('id=%s status=%s HTTP=%s bytes=%s completed=%s ok=%s',
                         item['id'], row['status'], code, len(raw), attempted, ok)
                if len(writer.pending) >= args.batch_size or time.monotonic() - last_save >= 60:
                    writer.flush()
                    last_save = time.monotonic()
                if attempted < len(selected) and not await wait_until_ready(stop_file, args.delay):
                    reason = 'STOP'
                    break
    finally:
        # Save domain state even if closing the browser or flushing HTML raises.
        atomic_json(out / 'cooldown.json', state)
        writer.close()
        summary = dict(engine='chrome', domain=DOMAIN, mode='probe' if args.probe else 'scan' if args.scan else 'batch',
                       attempted=attempted, ok=ok, remaining_this_run=len(selected) - attempted,
                       stop_reason=reason, delay_seconds=args.delay, cooldown=state, output=str(out))
        atomic_json(out / 'run_summary.json', summary)
    return summary


@asynccontextmanager
async def chrome_page(channel):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, channel=channel)
        try:
            page = await browser.new_page()
            async def restrict_navigation(route):
                request = route.request
                if request.is_navigation_request() and request.frame == page.main_frame and not is_vov(request.url):
                    await route.abort()
                else:
                    await route.continue_()
            await page.route('**/*', restrict_navigation)
            yield page
        finally:
            await browser.close()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    repo = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get('R2AI_DATA_ROOT', 'D:/r2ai_data'))
    default_db = root / 'crawl/status.duckdb'
    if not default_db.is_file():
        default_db = repo / 'status.duckdb'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status-db', type=Path, default=Path(os.environ.get('R2AI_STATUS_DB', str(default_db))))
    parser.add_argument('--out', type=Path, default=root / 'crawl/vov')
    parser.add_argument('--delay', type=float, default=MIN_DELAY)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--limit', '--max-urls', type=int, help='Maximum URLs in this run')
    parser.add_argument('--channel', default='chrome', choices=['chrome', 'msedge'])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--probe', action='store_true', help='Fetch one URL')
    modes.add_argument('--run', action='store_true', help='Fetch one batch')
    modes.add_argument('--scan', '--all-vov', action='store_true', help='Collect the remaining VOV queue')
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or args.delay < MIN_DELAY or not 1 <= args.batch_size <= 1000:
        parser.error('Use --delay >=2 seconds and --batch-size between 1 and 1000.')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive.')
    args.out = args.out.expanduser().resolve()
    args.status_db = args.status_db.expanduser().resolve()
    rows = source_rows(args.status_db)
    by_id = {r['id']: r for r in rows}

    def prepare_queue():
        local = last_status(args.out / 'status.csv')
        saved = saved_records(args.out, by_id)
        selected = queue_rows(rows, local, saved)
        path = args.out / 'cooldown.json'
        state = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else dict(kind='unprobed', next_probe_at=None)
        recover = {rid: row for rid, row in saved.items() if local.get(rid, {}).get('sha1_raw') != row['sha1_raw'] or local.get(rid, {}).get('shard_file') != row['shard_file'] or local.get(rid, {}).get('status') != 'ok'}
        return selected, state, recover, len(saved)

    active = args.probe or args.run or args.scan
    if not active:
        selected, state, _, count = prepare_queue()
        print(f'Source DB (read-only snapshot): {args.status_db}')
        print(f'VOV source IDs: {len(rows):,}; eligible: {len(selected):,}; raw saved here: {count:,}')
        print(f'Output: {args.out}')
        print(gate(state) or 'Ready. Includes historical blocked IDs, all source ID parities; excludes VOV2.')
        print('Offline plan only: zero requests. Use --probe or --scan.')
        return 0
    with exclusive_lock(args.out.parent / 'locks/vov.lock'), keep_system_awake():
        selected, state, recover, count = prepare_queue()
        if gate(state):
            print(gate(state))
            return 2
        if (args.out / 'STOP').exists():
            print(f'STOP exists: {args.out / "STOP"}; zero requests.')
            return 2
        print(f'VOV source IDs: {len(rows):,}; eligible: {len(selected):,}; saved: {count:,}')
        if not selected and not recover:
            print('No eligible VOV IDs; zero requests.')
            return 0
        args.out.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(args.out / 'crawl.log', encoding='utf-8')])
        summary = asyncio.run(collect(selected, args.out, state, args, recover=recover))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        success = summary['stop_reason'] in {'queue_finished', 'STOP'}
        return 0 if success and (not args.probe or summary['ok']) else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed checkpoints are retained.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
