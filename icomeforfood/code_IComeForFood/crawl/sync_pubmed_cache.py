"""Reconcile downloaded Europe PMC batch caches with crawl_status.

Default is a read-only dry run. The cache remains the source of PubMed content;
this command records its provenance and marks only validated PMID matches ok.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

import duckdb
import pandas as pd
import zstandard as zstd

try:
    from crawl.db_file_lock import db_file_lock
    from crawl.longchau_odd import exclusive_lock
except ModuleNotFoundError:  # python crawl/sync_pubmed_cache.py
    from db_file_lock import db_file_lock
    from longchau_odd import exclusive_lock


PMID_PATH = re.compile(r'^/(\d+)/?$')


def pmid_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname != 'pubmed.ncbi.nlm.nih.gov':
        return None
    found = PMID_PATH.fullmatch(parsed.path)
    return found.group(1) if found else None


def scan_cache(cache_dir: Path) -> tuple[dict[str, tuple[str, datetime]], int]:
    files = sorted(cache_dir.glob('batch_*.json.zst'))
    if not files:
        raise FileNotFoundError(f'No PubMed batch files in {cache_dir}')
    found: dict[str, tuple[str, datetime]] = {}
    duplicates = 0
    decompressor = zstd.ZstdDecompressor()
    for path in files:
        with path.open('rb') as source, decompressor.stream_reader(source) as stream:
            records = json.load(io.TextIOWrapper(stream, encoding='utf-8'))
        if not isinstance(records, list):
            raise ValueError(f'Expected a JSON list in {path}')
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f'Invalid PubMed record in {path}')
            pmid = str(record.get('pmid') or '').strip()
            if not pmid.isdigit() or not (record.get('title') or record.get('abstractText')):
                raise ValueError(f'Missing PMID/title/abstract in {path}: {pmid!r}')
            if pmid in found:
                duplicates += 1
            else:
                found[pmid] = (path.name, fetched_at)
    return found, duplicates


def backup_database(database: Path) -> Path:
    """Copy DuckDB and its WAL while the shared file lock is held."""
    folder = database.parent / 'backups'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target = folder / f'{database.name}.before_pubmed_{stamp}'
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(database, target)
    wal = Path(str(database) + '.wal')
    if wal.is_file():
        shutil.copy2(wal, Path(str(target) + '.wal'))
    print(f'Backup: {target}', flush=True)
    return target


def reconcile(cache_dir: Path, database: Path, *, apply: bool = False,
              min_coverage: float = 0.99, retry_out: Path | None = None) -> dict[str, int]:
    cache_dir, database = Path(cache_dir).resolve(), Path(database).resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if not 0 <= min_coverage <= 1:
        raise ValueError('min_coverage must be between 0 and 1')
    cache, duplicates = scan_cache(cache_dir)
    retry_out = Path(retry_out) if retry_out else database.parent / 'retry_crawl4ai'
    # A running retry process has snapshotted the old queue. Do not update the
    # DB until it stops, or it could request PubMed again despite this import.
    retry_lock = exclusive_lock(retry_out / 'crawl.lock') if apply else nullcontext()
    with retry_lock, db_file_lock(database):
        # A failed backup must abort before the DB is opened for writing.
        if apply:
            backup_database(database)
        con = duckdb.connect(str(database), read_only=not apply)
        try:
            rows = con.execute("""SELECT id,url,status FROM crawl_status
                WHERE lower(url) LIKE '%pubmed.ncbi.nlm.nih.gov%'""").fetchall()
            source: dict[str, tuple[int, str]] = {}
            for rid, url, status in rows:
                pmid = pmid_from_url(str(url))
                if pmid is None:
                    continue
                if pmid in source:
                    raise ValueError(f'Duplicate PubMed PMID in DB: {pmid}')
                source[pmid] = (int(rid), str(status or ''))
            if not source:
                raise ValueError('No PubMed URL rows in crawl_status')
            matched = [(rid, pmid, cache[pmid][0], cache[pmid][1])
                       for pmid, (rid, _status) in source.items() if pmid in cache]
            already_ok = sum(status == 'ok' for pmid, (_rid, status) in source.items()
                             if pmid in cache)
            report = dict(db_pubmed=len(source), cache_pmids=len(cache),
                          matched=len(matched), missing=len(source) - len(matched),
                          already_ok=already_ok, to_update=len(matched) - already_ok,
                          duplicate_records=duplicates,
                          cache_not_in_db=len(set(cache) - set(source)))
            print(', '.join(f'{key}={value:,}' for key, value in report.items()), flush=True)
            if report['missing']:
                print('Missing sample: ' + ', '.join(
                    pmid for pmid in source if pmid not in cache)[:200], flush=True)
            if not apply:
                print('Dry run; no DB rows changed.', flush=True)
                return report
            if len(matched) / len(source) < min_coverage:
                raise ValueError(f'Coverage below {min_coverage:.1%}; refusing partial import')
            stage = pd.DataFrame(matched, columns=['id', 'pmid', 'batch_file', 'fetched_at'])
            con.register('_pubmed_import', stage)
            con.execute('BEGIN TRANSACTION')
            try:
                con.execute("""CREATE TABLE IF NOT EXISTS pubmed_cache_index (
                    id BIGINT PRIMARY KEY, pmid VARCHAR, batch_file VARCHAR,
                    fetched_at TIMESTAMP, imported_at TIMESTAMP)""")
                con.execute("""DELETE FROM pubmed_cache_index AS idx
                    USING _pubmed_import AS src WHERE idx.id=src.id""")
                con.execute("""INSERT INTO pubmed_cache_index
                    SELECT id,pmid,batch_file,fetched_at,current_timestamp
                    FROM _pubmed_import""")
                con.execute("""UPDATE crawl_status AS dst SET
                    status='ok', final_url=dst.url, http_code=NULL,
                    content_type='application/json', n_bytes=NULL,
                    fetched_at=src.fetched_at, shard=NULL, sha1_raw=NULL,
                    error_msg=NULL
                    FROM _pubmed_import AS src
                    WHERE dst.id=src.id AND dst.status IS DISTINCT FROM 'ok'""")
                confirmed = con.execute("""SELECT count(*) FROM crawl_status AS dst
                    JOIN _pubmed_import AS src ON dst.id=src.id
                    WHERE dst.status='ok'""").fetchone()[0]
                if confirmed != len(matched):
                    raise ValueError(f'Only {confirmed:,}/{len(matched):,} matched IDs are ok')
                con.execute('COMMIT')
            except Exception:
                con.execute('ROLLBACK')
                raise
            print(f'Updated {report["to_update"]:,} PubMed statuses; '
                  f'indexed {len(matched):,} PMID-to-batch mappings.', flush=True)
            return report
        finally:
            con.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=Path('D:/r2ai_data/crawl/pubmed'))
    parser.add_argument('--status-db', type=Path, default=Path('D:/r2ai_data/crawl/status.duckdb'))
    parser.add_argument('--retry-out', type=Path, help='Retry crawler output directory (for its active-run lock)')
    parser.add_argument('--min-coverage', type=float, default=0.99)
    parser.add_argument('--apply', action='store_true', help='Write validated PMID matches to DB')
    args = parser.parse_args(argv)
    reconcile(args.cache_dir, args.status_db, apply=args.apply,
              min_coverage=args.min_coverage, retry_out=args.retry_out)


if __name__ == '__main__':
    main()
