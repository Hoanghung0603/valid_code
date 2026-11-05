"""Promote completed Long Chau staging shards to the main crawl contract.

The crawler keeps its resumable tv_shard_* files and status.csv. This command
validates them, packs new records into shard_XXXXX.jsonl.zst (at most 10,000
IDs each), and points crawl_status at the new shards. It never sends requests.
Run only after the Long Chau crawler has stopped.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import duckdb
import zstandard as zstd

from longchau_odd import exclusive_lock, is_longchau
from db_file_lock import db_file_lock
FIELDS = {'id', 'final_url', 'http_code', 'content_type', 'encoding', 'body_b64'}
SHARD_NAME = re.compile(r'^shard_(\d+)\.jsonl\.zst$')


def navigation_timed_out(error: str | None) -> bool:
    message = str(error or '').lower()
    return 'timeout' in message or 'timed out' in message


def latest_status(source: Path) -> dict[int, dict]:
    path = source / 'status.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[int, dict] = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if (row.get('id') or '').isdigit() and row.get('status') and None not in row:
                rows[int(row['id'])] = row
    return rows


def records(source: Path, statuses: dict[int, dict]):
    expected = {rid for rid, row in statuses.items()
                if row['status'] in {'ok', 'non_html'}}
    paths = sorted((source / 'html').glob('*_shard_*.jsonl.zst'))
    if not paths:
        if expected:
            raise FileNotFoundError(f'No staging shards in {source / "html"}')
        return  # A run stopped at its first block; only failed statuses need syncing.
    seen: set[int] = set()
    for path in paths:
        with path.open('rb') as compressed, zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding='utf-8') as lines:
                for line_no, line in enumerate(lines, 1):
                    record = json.loads(line)
                    if set(record) != FIELDS:
                        raise ValueError(f'{path.name}:{line_no}: expected exactly six main-schema fields')
                    rid = int(record['id'])
                    status = statuses.get(rid)
                    if status is None or status.get('shard_file') != path.name:
                        continue  # superseded retry or a record already accounted for elsewhere
                    seen.add(rid)
                    if rid % 2 != 1 or not is_longchau(record['final_url']):
                        raise ValueError(f'{path.name}:{line_no}: invalid odd Long Chau ID/URL')
                    if status['status'] not in {'ok', 'non_html'}:
                        raise ValueError(f'{path.name}:{line_no}: saved body has status {status["status"]}')
                    body = base64.b64decode(record['body_b64'], validate=True)
                    if (len(body) != int(status['n_bytes'])
                            or hashlib.sha1(body).hexdigest() != status['sha1_raw']
                            or int(record['http_code']) != int(status['http_code'])
                            or record['final_url'] != status['final_url']
                            or record['content_type'] != status['content_type']):
                        raise ValueError(f'{path.name}:{line_no}: body/status mismatch for ID {rid}')
                    yield record, status
    if missing := expected - seen:
        raise ValueError(f'{source}: {len(missing)} saved IDs have no matching shard record')


def choose_records(sources: list[Path], db_rows: dict[int, tuple]):
    """Raw HTTP sources take precedence over older Crawl4AI snapshots."""
    selected: dict[int, tuple[Path, dict]] = {}
    seen_in_source: set[tuple[Path, int]] = set()
    for source in sources:
        statuses = latest_status(source)
        for record, status in records(source, statuses):
            rid = int(record['id'])
            if (source, rid) in seen_in_source:
                raise ValueError(f'Duplicate active ID {rid} in {source}')
            seen_in_source.add((source, rid))
            db_row = db_rows.get(rid)
            if db_row is None or db_row[0] != status['url']:
                raise ValueError(f'ID/URL {rid} does not match the selected crawl_status DB')
            if db_row[1] in {'ok', 'non_html'} and db_row[2] is not None:
                continue
            selected.setdefault(rid, (source, status))
    return selected


def choose_errors(sources: list[Path], db_rows: dict[int, tuple], saved_ids: set[int]):
    """Keep latest failed attempts in crawl_status without assigning a shard."""
    errors: dict[int, dict] = {}
    for source in sources:
        for rid, row in latest_status(source).items():
            if rid in saved_ids or row['status'] not in {
                    'timeout', 'http_error', 'dns_error', 'blocked', 'soft_404', 'empty'}:
                continue
            db_row = db_rows.get(rid)
            if db_row is None or db_row[0] != row['url']:
                raise ValueError(f'Failed ID/URL {rid} does not match the selected crawl_status DB')
            if db_row[1] in {'ok', 'non_html'}:
                continue
            row = dict(row)
            if row['status'] == 'http_error' and navigation_timed_out(row.get('error_msg')):
                row['status'] = 'timeout'  # classify an older Crawl4AI navigation error
            errors.setdefault(rid, row)
    return errors


def next_shard_index(html_dir: Path, db_max: int | None) -> int:
    disk_max = max((int(match.group(1)) for path in html_dir.glob('shard_*.jsonl.zst')
                    if (match := SHARD_NAME.fullmatch(path.name))), default=-1)
    return max(disk_max, db_max if db_max is not None else -1) + 1


def promote(sources: list[Path], status_db: Path, html_dir: Path, apply: bool = False):
    if not status_db.is_file():
        raise FileNotFoundError(status_db)
    if not html_dir.is_dir():
        raise FileNotFoundError(html_dir)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Provide each staging source exactly once')
    lock = html_dir.parent / 'locks' / 'nhathuoclongchau.lock'
    with exclusive_lock(lock), db_file_lock(status_db):
        con = duckdb.connect(str(status_db), read_only=not apply)
        try:
            db_rows = {int(rid): (url, state, shard) for rid, url, state, shard in con.execute(
                "SELECT id,url,status,shard FROM crawl_status WHERE lower(url) LIKE '%nhathuoclongchau.com.vn%'").fetchall()}
            db_max = con.execute('SELECT max(shard) FROM crawl_status').fetchone()[0]
            selected = choose_records(sources, db_rows)
            errors = choose_errors(sources, db_rows, set(selected))
            start = next_shard_index(html_dir, db_max)
            shard_count = (len(selected) + 9999) // 10000
            print(f'Validated {len(selected):,} new odd IDs and {len(errors):,} failed IDs; '
                  f'{shard_count} main shard(s), starting at {start:05d}')
            print(f'Status DB: {status_db}; HTML: {html_dir}')
            if not apply or not (selected or errors):
                print('Dry run; no files or DB rows changed.' if not apply else 'Already up to date.')
                return len(selected)

            wanted = set(selected)
            written: set[int] = set()
            db_updates = []
            created: list[Path] = []
            target = None
            temp = None
            output = None
            encoder = None
            count_in_shard = 0
            idx = start - 1
            committed = False
            try:
                for source in sources:
                    statuses = latest_status(source)
                    for record, status in records(source, statuses):
                        rid = int(record['id'])
                        if rid not in wanted or rid in written or selected[rid][0] != source:
                            continue
                        if encoder is None or count_in_shard == 10000:
                            if encoder is not None:
                                encoder.close()
                                output.close()
                                temp.replace(target)
                                created.append(target)
                            idx += 1
                            target = html_dir / f'shard_{idx:05d}.jsonl.zst'
                            if target.exists():
                                raise FileExistsError(target)
                            temp = target.with_name(target.name + '.tmp')
                            if temp.exists():
                                raise FileExistsError(temp)
                            output = temp.open('wb')
                            encoder = zstd.ZstdCompressor(level=3).stream_writer(output, closefd=False)
                            count_in_shard = 0
                        encoder.write((json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8'))
                        count_in_shard += 1
                        written.add(rid)
                        db_updates.append((status['final_url'], status['status'], int(status['http_code']),
                                           status['content_type'], int(status['n_bytes']),
                                           int(status['attempts'] or 1), status['fetched_at'], idx,
                                           status['sha1_raw'], status['error_msg'] or None, rid, status['url']))
                if encoder is not None:
                    encoder.close()
                    output.close()
                    temp.replace(target)
                    created.append(target)
                    encoder = output = temp = None
                if written != wanted:
                    raise ValueError(f'Missing {len(wanted - written)} selected records during export')
                con.execute('BEGIN TRANSACTION')
                try:
                    if db_updates:
                        con.executemany('''UPDATE crawl_status SET final_url=?,status=?,http_code=?,
                            content_type=?,n_bytes=?,attempts=?,fetched_at=?,shard=?,sha1_raw=?,error_msg=?
                            WHERE id=? AND url=?''', db_updates)
                        changed = con.execute('SELECT count(*) FROM crawl_status WHERE id IN (SELECT unnest(?)) AND status IN (\'ok\',\'non_html\') AND shard >= ?',
                                              [list(written), start]).fetchone()[0]
                        if changed != len(written):
                            raise ValueError(f'Only {changed}/{len(written)} DB rows point to exported shards')
                    if errors:
                        error_updates = []
                        for rid, row in errors.items():
                            code = row.get('http_code') or ''
                            error_updates.append((row.get('final_url') or row['url'], row['status'],
                                                  int(code) if code.isdigit() else None,
                                                  row.get('content_type') or None,
                                                  int(row.get('n_bytes') or 0),
                                                  int(row.get('attempts') or 1), row.get('fetched_at') or None,
                                                  row.get('error_msg') or None, rid, row['url']))
                        con.executemany('''UPDATE crawl_status SET final_url=?,status=?,http_code=?,
                            content_type=?,n_bytes=?,attempts=?,fetched_at=?,shard=NULL,
                            sha1_raw=NULL,error_msg=? WHERE id=? AND url=?
                            AND status NOT IN ('ok','non_html')''', error_updates)
                    con.execute('COMMIT')
                    committed = True
                except Exception:
                    con.execute('ROLLBACK')
                    raise
            except Exception:
                if encoder is not None:
                    encoder.close()
                if output is not None:
                    output.close()
                if temp is not None and temp.exists():
                    temp.unlink()
                if not committed:
                    for path in created:
                        path.unlink()
                raise
            print(f'Exported {len(written):,} IDs to {len(created)} shard(s); '
                  f'updated {len(errors):,} failed statuses in crawl_status.')
            return len(written)
        finally:
            con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True,
                        help='staging directory; pass raw HTTP first, then legacy snapshots')
    parser.add_argument('--status-db', type=Path, required=True)
    parser.add_argument('--html-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='write validated shards and update DB')
    args = parser.parse_args()
    promote([p.resolve() for p in args.source], args.status_db.resolve(), args.html_dir.resolve(), args.apply)


if __name__ == '__main__':
    main()
