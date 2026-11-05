"""Resume non-ok corpus URLs with Crawl4AI and stage main-compatible raw HTML.

The queue comes from crawl_status, excluding Long Chau, zysjonline, PubMed,
and VOV domains. ``plan`` is read-only; ``run`` stages raw responses and auto-syncs
validated checkpoints to the DB. ``merge`` validates complete staging shards.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlparse

import duckdb
import zstandard as zstd

try:
    from crawl.longchau_odd import CHALLENGE_MARKERS, STATUS_COLUMNS, atomic_json, encoding, exclusive_lock
    from crawl.longchau_crawl4ai_fast import create_optimized_crawler
    from crawl.db_file_lock import db_file_lock
except ModuleNotFoundError:  # python crawl/retry_crawl4ai.py
    from longchau_odd import CHALLENGE_MARKERS, STATUS_COLUMNS, atomic_json, encoding, exclusive_lock
    from longchau_crawl4ai_fast import create_optimized_crawler
    from db_file_lock import db_file_lock


EXCLUDED = {
    'nhathuoclongchau.com.vn',
    'tiemchunglongchau.com.vn',
    'zysjonline.com',
    'pubmed.ncbi.nlm.nih.gov',
    'vov.vn',
}
RAW_FIELDS = {'id', 'final_url', 'http_code', 'content_type', 'encoding', 'body_b64'}
BLOCK_CODES = {401, 403, 429, 503, 521}
HOST_COOLDOWN = timedelta(minutes=5)


def host(url):
    return (urlparse(str(url)).hostname or '').lower().removeprefix('www.')


def request_url(url):
    """Use Pharmacity's public article URL for a matching beta article slug."""
    source = str(url)
    parsed = urlparse(source)
    if host(source) == 'pmc-ecm-healthblog.beta.pharmacity.io':
        slug = parsed.path.strip('/')
        if slug and '/' not in slug and not parsed.query and not parsed.fragment:
            return f'https://www.pharmacity.vn/{slug}.htm'
    return source


def in_scope(url):
    domain = host(url)
    return bool(domain) and not any(
        domain == excluded or domain.endswith('.' + excluded)
        for excluded in EXCLUDED
    )


def utc_now():
    return datetime.now(timezone.utc)


def snapshot_rows(database):
    """Copy DB and WAL together so a separate writer does not lock the run."""
    database = Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    with tempfile.TemporaryDirectory(prefix='retry_crawl4ai_db_') as folder:
        copy = Path(folder) / 'status.duckdb'
        with db_file_lock(database):
            shutil.copy2(database, copy)
            wal = Path(str(database) + '.wal')
            if wal.is_file():
                shutil.copy2(wal, Path(str(copy) + '.wal'))
        con = duckdb.connect(str(copy), read_only=True)
        try:
            rows = con.execute("""SELECT id,url,lang_expected,status,http_code,attempts
                FROM crawl_status WHERE status != 'ok' ORDER BY id""").fetchall()
        finally:
            con.close()
    return [dict(id=int(rid), url=url, lang_expected=lang, db_status=status,
                 db_code=code, attempts=int(attempts or 0))
            for rid, url, lang, status, code, attempts in rows if in_scope(url)]


def local_status(path):
    latest = {}
    if path.is_file():
        with path.open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                if row.get('id', '').isdigit() and row.get('status') and None not in row:
                    latest[int(row['id'])] = row
    return latest


def active_cooldowns(previous, now=None):
    """Keep a host on cooldown only for a real refusal or verification page."""
    now = now or utc_now()
    until = {}
    for row in previous.values():
        if row.get('status') != 'blocked' or not row.get('fetched_at'):
            continue
        code = str(row.get('http_code') or '')
        if code.isdigit() and int(code) not in BLOCK_CODES | {200}:
            continue  # e.g. a 404 page containing the words "access denied"
        try:
            when = datetime.fromisoformat(row['fetched_at'].replace('Z', '+00:00'))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        deadline = when + HOST_COOLDOWN
        domain = host(request_url(row['url']))
        if domain and deadline > now:
            until[domain] = max(until.get(domain, deadline), deadline)
    return until


def repeated_404(row, prior):
    """Treat a second HTTP 404 as needing manual review, not another blind retry."""
    if prior and prior.get('status') == 'http_error' and str(prior.get('http_code')) == '404':
        return int(prior.get('attempts') or 0) >= 2
    return (row['db_status'] == 'http_error' and row['db_code'] == 404
            and row['attempts'] >= 2)


def pending_rows(database, out, retry_failed=False, retry_404=False, only_hosts=None):
    rows = snapshot_rows(database)
    if only_hosts:
        rows = [row for row in rows if host(row['url']) in only_hosts]
    previous = local_status(out / 'status.csv')
    if retry_failed:
        selected = [row for row in rows if previous.get(row['id'], {}).get('status') not in {'ok', 'non_html'}]
    else:
        selected = [row for row in rows if row['id'] not in previous]
    if not retry_404:
        selected = [row for row in selected if not repeated_404(row, previous.get(row['id']))]
    return rows, selected


class StageWriter:
    """Commit status rows for saved bodies only after their shard is durable."""
    def __init__(self, out, shard_size=1000):
        self.out = out
        self.html = out / 'html'
        self.html.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.index = max((int(p.stem.split('_')[-1].split('.')[0])
                          for p in self.html.glob('retry_shard_*.jsonl.zst')), default=-1) + 1
        self.buffer = []
        path = out / 'status.csv'
        new = not path.exists() or path.stat().st_size == 0
        self.status_file = path.open('a', newline='', encoding='utf-8')
        self.status_writer = csv.DictWriter(self.status_file, fieldnames=STATUS_COLUMNS)
        if new:
            self.status_writer.writeheader()
            self._sync()

    def _sync(self):
        self.status_file.flush()
        os.fsync(self.status_file.fileno())

    def add_error(self, row):
        self.status_writer.writerow(row)
        self._sync()

    def add_body(self, record, row):
        if set(record) != RAW_FIELDS:
            raise ValueError('Raw record must have the six main crawler fields')
        self.buffer.append((record, row))
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        target = self.html / f'retry_shard_{self.index:05d}.jsonl.zst'
        temp = target.with_name(target.name + '.tmp')
        if target.exists() or temp.exists():
            raise FileExistsError(target)
        try:
            with temp.open('wb') as handle:
                with zstd.ZstdCompressor(level=3).stream_writer(handle, closefd=False) as packed:
                    for record, _ in self.buffer:
                        packed.write((json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8'))
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(target)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        for _, row in self.buffer:
            row['shard_file'] = target.name
            self.status_writer.writerow(row)
        self._sync()
        self.buffer.clear()
        self.index += 1

    def close(self):
        try:
            self.flush()
        finally:
            self.status_file.close()


def classify(result, captured, item):
    """Use original navigation bytes; never substitute Crawl4AI's rendered HTML."""
    stamp = utc_now().isoformat()
    row = dict.fromkeys(STATUS_COLUMNS)
    row.update(id=item['id'], url=item['url'], final_url=item['url'],
               lang_expected=item['lang_expected'], status='http_error',
               attempts=item['attempts'] + 1, fetched_at=stamp, n_bytes=0)
    if 'body' not in captured:
        detail = str(getattr(result, 'error_message', '') or captured.get('error') or 'No navigation response')
        row['status'] = ('timeout' if 'timeout' in detail.lower() else
                         'blocked' if 'robots.txt' in detail.lower() else 'http_error')
        row['error_msg'] = detail[:500]
        return None, row
    raw = captured['body']
    code = captured['status']
    headers = captured['headers']
    final = captured['url']
    content_type = headers.get('content-type', '')
    sample = raw[:200000].decode('utf-8', errors='ignore').lower()
    title_match = re.search(r'<title[^>]*>(.*?)</title>', sample, re.I | re.S)
    title = title_match.group(1) if title_match else ''
    challenge = (headers.get('cf-mitigated', '').lower() == 'challenge' or
                 any(marker in sample for marker in CHALLENGE_MARKERS if marker != 'captcha') or
                 'captcha' in title or
                 any(marker in final.lower() for marker in ('verify.html', 'captcha')))
    row.update(final_url=final, http_code=code, content_type=content_type, n_bytes=len(raw))
    if code in BLOCK_CODES or (code == 200 and challenge):
        row.update(status='blocked', error_msg='Access refused or verification page')
        return None, row
    if code != 200:
        row.update(status='http_error', error_msg=f'HTTP {code}')
        return None, row
    if not raw:
        row.update(status='empty', error_msg='HTTP 200 with empty body')
        return None, row
    mime = content_type.split(';', 1)[0].strip().lower()
    is_html = mime in {'text/html', 'application/xhtml+xml'} or (
        not mime and raw.lstrip()[:20].lower().startswith((b'<!doctype html', b'<html')))
    row.update(status='ok' if is_html else 'non_html', sha1_raw=hashlib.sha1(raw).hexdigest())
    record = dict(id=item['id'], final_url=final, http_code=200,
                  content_type=content_type, encoding=encoding(raw, content_type) if is_html else None,
                  body_b64=base64.b64encode(raw).decode('ascii'))
    return record, row


async def crawl(rows, out, *, max_urls, delay, shard_size, crawler_factory=None,
                run_config=None, auto_sync=False, status_db=None, html_dir=None):
    if crawler_factory is None:
        crawler_factory, default_run_config = create_optimized_crawler()
        if run_config is None:
            run_config = default_run_config
    out.mkdir(parents=True, exist_ok=True)
    writer = StageWriter(out, shard_size)
    counts = dict(attempted=0, ok=0, non_html=0, blocked=0, failed=0)
    cooldowns = active_cooldowns(local_status(out / 'status.csv'))
    last_request = {}
    pending = list(enumerate(rows, 1))
    try:
        if auto_sync:
            sync_incremental(out, status_db, html_dir)
        while pending and not (out / 'STOP').exists():
            now = utc_now()
            ready = []
            delayed = []
            for entry in pending:
                deadline = cooldowns.get(host(request_url(entry[1]['url'])))
                (delayed if deadline and deadline > now else ready).append(entry)
            if not ready:
                if not delayed:
                    break
                deadline = min(cooldowns[host(request_url(item['url']))] for _, item in delayed)
                print(f'All {len(delayed):,} remaining URLs are on host cooldown. '
                      f'Next probe: {deadline.astimezone().isoformat()}.', flush=True)
                if max_urls:
                    print('Run again after cooldown, or use --max-urls 0 to wait.', flush=True)
                    break
                while utc_now() < deadline and not (out / 'STOP').exists():
                    await asyncio.sleep(min(60, max(0.1, (deadline - utc_now()).total_seconds())))
                    if utc_now() < deadline and not (out / 'STOP').exists():
                        print(f'Waiting for host cooldown; next probe '
                              f'{deadline.astimezone().isoformat()}.', flush=True)
                pending = delayed
                continue
            if delayed:
                print(f'{len(ready):,} URLs ready; {len(delayed):,} waiting on '
                      f'{len({host(request_url(item["url"])) for _, item in delayed})} host cooldown(s).', flush=True)
            print(f'Starting Crawl4AI browser; {len(ready):,} ready URLs. '
                  'Progress will print after every request.', flush=True)
            stopped = False
            async with crawler_factory() as crawler:
                strategy = getattr(crawler, 'crawler_strategy', None)
                if strategy is None or not hasattr(strategy, 'set_hook'):
                    raise RuntimeError('Crawl4AI after_goto hook is required for raw HTTP bytes')
                captured = {}

                async def after_goto(page, *, response=None, **_kwargs):
                    if response is None:
                        captured['error'] = 'Navigation returned no HTTP response'
                        return
                    try:
                        captured['body'] = await response.body()
                        captured['headers'] = {str(k).lower(): str(v)
                                               for k, v in (await response.all_headers()).items()}
                        captured['url'] = str(response.url)
                        captured['status'] = int(response.status)
                    except Exception as exc:
                        captured['error'] = f'{type(exc).__name__}: {exc}'[:500]

                strategy.set_hook('after_goto', after_goto)
                print('Browser ready; requesting URLs.', flush=True)
                for index, (position, item) in enumerate(ready):
                    if (max_urls and counts['attempted'] >= max_urls) or (out / 'STOP').exists():
                        stopped = True
                        break
                    target = request_url(item['url'])
                    domain = host(target)
                    if cooldowns.get(domain, now) > utc_now():
                        delayed.append((position, item))
                        continue
                    wait = delay - (time.monotonic() - last_request.get(domain, -1e12))
                    if wait > 0:
                        await asyncio.sleep(wait)
                    if (out / 'STOP').exists():
                        stopped = True
                        break
                    last_request[domain] = time.monotonic()
                    captured.clear()
                    result = None
                    try:
                        result = await crawler.arun(url=target, config=run_config)
                    except Exception as exc:
                        captured.setdefault('error', f'{type(exc).__name__}: {exc}')
                    record, row = classify(result, captured, item)
                    counts['attempted'] += 1
                    if record:
                        writer.add_body(record, row)
                        counts[row['status']] += 1
                    else:
                        writer.add_error(row)
                        counts['blocked' if row['status'] == 'blocked' else 'failed'] += 1
                        if row['status'] == 'blocked':
                            cooldowns[domain] = utc_now() + HOST_COOLDOWN
                    print(f'{utc_now().astimezone().strftime("%H:%M:%S")} '
                          f'queue={position:,}/{len(rows):,} tried={counts["attempted"]:,} '
                          f'ok={counts["ok"]:,} blocked={counts["blocked"]:,} '
                          f'failed={counts["failed"]:,} id={item["id"]} '
                          f'host={domain} status={row["status"]} '
                          f'HTTP={row["http_code"] if row["http_code"] is not None else "-"}',
                          flush=True)
                    if counts['attempted'] % 50 == 0:
                        writer.flush()
                        if auto_sync:
                            sync_incremental(out, status_db, html_dir)
                        print(f'Checkpoint saved after {counts["attempted"]:,} requests.', flush=True)
                if stopped:
                    delayed.extend(ready[index:])
            pending = delayed
            if stopped:
                break
    finally:
        try:
            writer.close()
            if auto_sync:
                sync_incremental(out, status_db, html_dir)
        finally:
            (out / 'run_summary.json').write_text(json.dumps({**counts, 'blocked_hosts': sorted(
                domain for domain, deadline in cooldowns.items() if deadline > utc_now()),
                 'output': str(out), 'capture': 'http_response_body',
                 'auto_sync': auto_sync}, indent=2), encoding='utf-8')
    return counts


def staged_records(out, statuses):
    seen = set()
    for path in sorted((out / 'html').glob('retry_shard_*.jsonl.zst')):
        with path.open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as stream:
            for line in io.TextIOWrapper(stream, encoding='utf-8'):
                record = json.loads(line)
                if set(record) != RAW_FIELDS:
                    raise ValueError(f'{path}: invalid raw fields')
                rid = int(record['id'])
                row = statuses.get(rid)
                if not row or row.get('shard_file') != path.name:
                    continue  # older attempt superseded by a newer one
                if rid in seen:
                    raise ValueError(f'Duplicate active ID {rid}')
                seen.add(rid)
                body = base64.b64decode(record['body_b64'], validate=True)
                if (len(body) != int(row['n_bytes']) or
                    hashlib.sha1(body).hexdigest() != row['sha1_raw'] or
                    record['final_url'] != row['final_url'] or
                    int(record['http_code']) != int(row['http_code']) or
                    record['content_type'] != row['content_type']):
                    raise ValueError(f'{path}: status/body mismatch for ID {rid}')
                yield record, row
    expected = {rid for rid, row in statuses.items() if row['status'] in {'ok', 'non_html'}}
    if seen != expected:
        raise ValueError(f'{len(expected - seen)} successful IDs lack a staged body')


def _new_status_rows(path, offset):
    """Read only complete, append-only CSV rows since the last sync."""
    with path.open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        if offset > end:
            raise ValueError('status.csv is shorter than the saved sync offset')
        handle.seek(offset)
        data = handle.read()
    if not data:
        return {}, end
    if not data.endswith(b'\n'):
        raise ValueError('status.csv has a partial last row; stop the writer before syncing')
    decoded = data.decode('utf-8-sig' if offset == 0 else 'utf-8')
    reader = csv.DictReader(io.StringIO(decoded, newline=''),
                            fieldnames=STATUS_COLUMNS)
    rows = {}
    for row in reader:
        if offset == 0 and row.get('id') == 'id':
            continue
        if not row.get('id', '').isdigit() or not row.get('status') or None in row:
            raise ValueError('Malformed status.csv row in unsynced range')
        rows[int(row['id'])] = row
    return rows, end


def _new_raw_records(out, rows):
    """Validate only staging shards mentioned by new successful statuses."""
    expected = {rid for rid, row in rows.items() if row['status'] in {'ok', 'non_html'}}
    seen = set()
    for name in sorted({rows[rid]['shard_file'] for rid in expected}):
        if not name.startswith('retry_shard_') or not name.endswith('.jsonl.zst'):
            raise ValueError(f'Invalid staging shard name: {name}')
        path = out / 'html' / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as stream:
            for line in io.TextIOWrapper(stream, encoding='utf-8'):
                record = json.loads(line)
                if set(record) != RAW_FIELDS:
                    raise ValueError(f'{path}: invalid raw fields')
                rid = int(record['id'])
                row = rows.get(rid)
                if not row or row.get('shard_file') != name:
                    continue  # another ID in the same shard was synced earlier
                if rid in seen:
                    raise ValueError(f'{path}: duplicate active ID {rid}')
                seen.add(rid)
                body = base64.b64decode(record['body_b64'], validate=True)
                if (len(body) != int(row['n_bytes']) or
                    hashlib.sha1(body).hexdigest() != row['sha1_raw'] or
                    record['final_url'] != row['final_url'] or
                    int(record['http_code']) != int(row['http_code']) or
                    record['content_type'] != row['content_type']):
                    raise ValueError(f'{path}: status/body mismatch for ID {rid}')
                yield record, row
    if seen != expected:
        raise ValueError(f'{len(expected - seen)} newly successful IDs lack a staged body')


def sync_incremental(out, database, html_dir):
    """Promote only the status.csv suffix since the last successful DB commit."""
    out, database, html_dir = Path(out), Path(database), Path(html_dir)
    status_path = out / 'status.csv'
    if not status_path.is_file():
        return dict(bodies=0, errors=0)
    state_path = out / 'sync_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    offset = int(state.get('status_offset', 0))
    rows, end = _new_status_rows(status_path, offset)
    if not rows:
        return dict(bodies=0, errors=0)
    if not database.is_file() or not html_dir.is_dir():
        raise FileNotFoundError('Main status DB and HTML directory must exist for auto-sync')
    # Validate all new bodies before creating any canonical shard or DB update.
    new_records = list(_new_raw_records(out, rows))
    with exclusive_lock(html_dir.parent / 'locks' / 'retry_crawl4ai.lock'), db_file_lock(database):
        con = duckdb.connect(str(database))
        try:
            ids = list(rows)
            source = {int(rid): (url, status, shard) for rid, url, status, shard in con.execute(
                'SELECT id,url,status,shard FROM crawl_status WHERE id IN (SELECT unnest(?))',
                [ids]).fetchall()}
            for rid, row in rows.items():
                # A host may be excluded after its results were staged. Keep
                # those saved results syncable, while pending_rows excludes it
                # from all future crawl requests.
                if rid not in source or source[rid][0] != row['url']:
                    raise ValueError(f'Staged ID/URL is outside source DB: {rid}')
            records = [(record, row) for record, row in new_records
                       if source[int(row['id'])][1] != 'ok' and not (
                           source[int(row['id'])][1] == 'non_html'
                           and source[int(row['id'])][2] is not None
                           and row['status'] == 'non_html')]
            errors = [row for rid, row in rows.items()
                      if row['status'] not in {'ok', 'non_html'}
                      and source[rid][1] not in {'ok', 'non_html'}]
            db_max = con.execute('SELECT coalesce(max(shard),-1) FROM crawl_status').fetchone()[0]
            if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='home_import_shards'").fetchone()[0]:
                db_max = max(db_max, con.execute('SELECT coalesce(max(shard),-1) FROM home_import_shards').fetchone()[0])
            disk_max = max((int(p.name.split('_')[1].split('.')[0])
                            for p in html_dir.glob('shard_*.jsonl.zst')), default=-1)
            idx = max(db_max, disk_max) + 1
            created = []
            updates = []
            try:
                for start in range(0, len(records), 10000):
                    chunk = records[start:start + 10000]
                    target = html_dir / f'shard_{idx:05d}.jsonl.zst'
                    temp = target.with_name(target.name + '.tmp')
                    if target.exists() or temp.exists():
                        raise FileExistsError(target)
                    try:
                        with temp.open('wb') as handle:
                            with zstd.ZstdCompressor(level=3).stream_writer(handle, closefd=False) as packed:
                                for record, row in chunk:
                                    packed.write((json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8'))
                                    updates.append((row['final_url'], row['status'], int(row['http_code']),
                                                    row['content_type'], int(row['n_bytes']),
                                                    int(row['attempts']), row['fetched_at'], idx,
                                                    row['sha1_raw'], row.get('error_msg') or None,
                                                    int(row['id']), row['url']))
                            handle.flush()
                            os.fsync(handle.fileno())
                        temp.replace(target)
                    except Exception:
                        temp.unlink(missing_ok=True)
                        raise
                    created.append(target)
                    idx += 1
                con.execute('BEGIN TRANSACTION')
                try:
                    if updates:
                        con.executemany('''UPDATE crawl_status SET final_url=?,status=?,http_code=?,
                            content_type=?,n_bytes=?,attempts=?,fetched_at=?,shard=?,sha1_raw=?,error_msg=?
                            WHERE id=? AND url=? AND status != 'ok' ''', updates)
                    if errors:
                        error_updates = []
                        for row in errors:
                            code = row.get('http_code') or ''
                            error_updates.append((row.get('final_url') or row['url'], row['status'],
                                                  int(code) if str(code).isdigit() else None,
                                                  row.get('content_type') or None,
                                                  int(row.get('n_bytes') or 0), int(row.get('attempts') or 1),
                                                  row.get('fetched_at') or None, row.get('error_msg') or None,
                                                  int(row['id']), row['url']))
                        con.executemany('''UPDATE crawl_status SET final_url=?,status=?,http_code=?,
                            content_type=?,n_bytes=?,attempts=?,fetched_at=?,shard=NULL,
                            sha1_raw=NULL,error_msg=? WHERE id=? AND url=?
                            AND status NOT IN ('ok','non_html')''', error_updates)
                    con.execute('COMMIT')
                except Exception:
                    con.execute('ROLLBACK')
                    raise
            except Exception:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
        finally:
            con.close()
    atomic_json(state_path, {'status_offset': end, 'synced_at': utc_now().isoformat()})
    result = dict(bodies=len(records), errors=len(errors))
    print(f'Auto-sync: {result["bodies"]} bodies, {result["errors"]} errors; offset {end}', flush=True)
    return result


def merge(out, database, html_dir, apply=False):
    """Validate staging first; copy new canonical shards and then commit DB."""
    statuses = local_status(out / 'status.csv')
    if not statuses:
        raise FileNotFoundError(out / 'status.csv')
    saved_count = sum(1 for _ in staged_records(out, statuses))
    errors = [row for row in statuses.values() if row['status'] not in {'ok', 'non_html'}]
    with exclusive_lock(html_dir.parent / 'locks' / 'retry_crawl4ai.lock'), db_file_lock(database):
        con = duckdb.connect(str(database), read_only=not apply)
        try:
            source = {int(rid): (url, status, shard) for rid, url, status, shard in con.execute(
                'SELECT id,url,status,shard FROM crawl_status WHERE status != \'ok\'').fetchall()}
            for row in statuses.values():
                rid = int(row['id'])
                if rid in source and source[rid][0] != row['url']:
                    raise ValueError(f'Staged ID/URL is outside source DB: {rid}')
            selected_count = sum(1 for row in statuses.values()
                                 if row['status'] in {'ok', 'non_html'}
                                 and int(row['id']) in source
                                 and not (source[int(row['id'])][1] == 'non_html'
                                          and source[int(row['id'])][2] is not None
                                          and row['status'] == 'non_html'))
            pending_errors = [row for row in errors if int(row['id']) in source]
            db_max = con.execute('SELECT coalesce(max(shard),-1) FROM crawl_status').fetchone()[0]
            if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='home_import_shards'").fetchone()[0]:
                db_max = max(db_max, con.execute('SELECT coalesce(max(shard),-1) FROM home_import_shards').fetchone()[0])
            disk_max = max((int(p.name.split('_')[1].split('.')[0])
                            for p in html_dir.glob('shard_*.jsonl.zst')), default=-1)
            first = max(db_max, disk_max) + 1
            print(f'Validated {saved_count:,} staged bodies; merge {selected_count:,} bodies and '
                  f'{len(pending_errors):,} failed statuses; first main shard {first:05d}')
            if not apply:
                return
            created = []
            updates = []
            idx = first
            try:
                output = packed = temp = target = None
                in_shard = 0
                try:
                    for record, row in staged_records(out, statuses):
                        rid = int(row['id'])
                        if rid not in source or (source[rid][1] == 'non_html'
                                                and source[rid][2] is not None
                                                and row['status'] == 'non_html'):
                            continue
                        if packed is None or in_shard == 10000:
                            if packed is not None:
                                packed.close()
                                output.flush()
                                os.fsync(output.fileno())
                                output.close()
                                temp.replace(target)
                                created.append(target)
                                idx += 1
                            target = html_dir / f'shard_{idx:05d}.jsonl.zst'
                            temp = target.with_name(target.name + '.tmp')
                            if target.exists() or temp.exists():
                                raise FileExistsError(target)
                            output = temp.open('wb')
                            packed = zstd.ZstdCompressor(level=3).stream_writer(output, closefd=False)
                            in_shard = 0
                        packed.write((json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8'))
                        in_shard += 1
                        updates.append((row['final_url'], row['status'], int(row['http_code']),
                                        row['content_type'], int(row['n_bytes']),
                                        int(row['attempts']), row['fetched_at'], idx,
                                        row['sha1_raw'], row.get('error_msg') or None,
                                        rid, row['url']))
                    if packed is not None:
                        packed.close()
                        output.flush()
                        os.fsync(output.fileno())
                        output.close()
                        temp.replace(target)
                        created.append(target)
                        packed = output = temp = None
                finally:
                    if packed is not None:
                        packed.close()
                    if output is not None:
                        output.close()
                    if temp is not None:
                        temp.unlink(missing_ok=True)
                if len(updates) != selected_count:
                    raise ValueError(f'Only {len(updates)}/{selected_count} selected bodies written')
                con.execute('BEGIN TRANSACTION')
                try:
                    if updates:
                        con.executemany('''UPDATE crawl_status SET final_url=?,status=?,http_code=?,
                            content_type=?,n_bytes=?,attempts=?,fetched_at=?,shard=?,sha1_raw=?,error_msg=?
                            WHERE id=? AND url=? AND status != 'ok' ''', updates)
                    for row in pending_errors:
                        code = row.get('http_code') or ''
                        con.execute('''UPDATE crawl_status SET final_url=?,status=?,http_code=?,
                            content_type=?,n_bytes=?,attempts=?,fetched_at=?,shard=NULL,sha1_raw=NULL,error_msg=?
                            WHERE id=? AND url=? AND status NOT IN ('ok','non_html')''',
                            [row.get('final_url') or row['url'], row['status'],
                             int(code) if str(code).isdigit() else None,
                             row.get('content_type') or None, int(row.get('n_bytes') or 0),
                             int(row.get('attempts') or 1), row.get('fetched_at') or None,
                             row.get('error_msg') or None, int(row['id']), row['url']])
                    con.execute('COMMIT')
                except Exception:
                    con.execute('ROLLBACK')
                    raise
            except Exception:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
            print(f'Merged {len(updates):,} bodies into {len(created)} main shards.')
        finally:
            con.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', 'sync', 'merge'])
    parser.add_argument('--status-db', type=Path, default=Path('D:/r2ai_data/crawl/status.duckdb'))
    parser.add_argument('--out', type=Path, default=Path('D:/r2ai_data/crawl/retry_crawl4ai'))
    parser.add_argument('--html-dir', type=Path, default=Path('D:/r2ai_data/crawl/html'))
    parser.add_argument('--max-urls', type=int, default=50,
                        help='Run at most this many requests; 0 means the whole queue')
    parser.add_argument('--delay', type=float, default=5.0, help='minimum seconds between requests to one host')
    parser.add_argument('--shard-size', type=int, default=1000)
    parser.add_argument('--retry-failed', action='store_true', help='retry failed staging rows on a later run')
    parser.add_argument('--retry-404', action='store_true',
                        help='also revisit repeated HTTP 404s (skipped by default after two attempts)')
    parser.add_argument('--source-host', action='append', default=[],
                        help='limit plan/run to this source host; may be repeated')
    parser.add_argument('--no-auto-sync', action='store_true',
                        help='leave run output in staging; default run auto-syncs each checkpoint to main')
    parser.add_argument('--apply', action='store_true', help='merge into main DB and HTML directory')
    args = parser.parse_args(argv)
    if args.max_urls < 0 or args.delay < 1 or not 1 <= args.shard_size <= 10000:
        parser.error('Require max-urls >= 0, delay >= 1, shard-size 1..10000')
    args.status_db = args.status_db.resolve()
    args.out = args.out.resolve()
    args.html_dir = args.html_dir.resolve()
    only_hosts = {h.lower().removeprefix('www.') for h in args.source_host}
    if args.command == 'merge':
        merge(args.out, args.status_db, args.html_dir, args.apply)
        return
    if args.command == 'sync':
        sync_incremental(args.out, args.status_db, args.html_dir)
        return
    retry_failed = args.retry_failed or args.retry_404 or (args.command == 'run' and args.max_urls == 0)
    all_rows, selected = pending_rows(args.status_db, args.out, retry_failed,
                                      args.retry_404, only_hosts)
    print(f'Source non-ok excluding Long Chau, zysjonline, PubMed and VOV: {len(all_rows):,}; '
          f'selected for crawl: {len(selected):,}; '
          f'retry failed staging rows: {retry_failed}; '
          f'source hosts: {", ".join(sorted(only_hosts)) if only_hosts else "all"}', flush=True)
    if args.command == 'plan':
        by_host = {}
        for row in selected:
            by_host[host(row['url'])] = by_host.get(host(row['url']), 0) + 1
        for domain, count in sorted(by_host.items(), key=lambda pair: -pair[1]):
            print(f'{domain},{count}')
        return
    with exclusive_lock(args.out / 'crawl.lock'):
        asyncio.run(crawl(selected, args.out, max_urls=args.max_urls,
                          delay=args.delay, shard_size=args.shard_size,
                          auto_sync=not args.no_auto_sync,
                          status_db=args.status_db, html_dir=args.html_dir))


if __name__ == '__main__':
    main()
