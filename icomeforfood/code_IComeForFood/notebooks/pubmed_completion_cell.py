"""Complete and extract PubMed records in Colab.

This source is embedded in notebooks 03 and 04. Keep all copies in sync.
"""

import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd


NCBI_EFETCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
NCBI_BATCH_SIZE = 200
PUBMED_DOC_SHARD_SIZE = 10_000


def _pubmed_source_rows(status_db):
    con = duckdb.connect(str(status_db), read_only=True)
    try:
        rows = con.execute("""SELECT id, url, status FROM crawl_status
            WHERE lower(url) LIKE '%pubmed.ncbi.nlm.nih.gov%'""").fetchall()
    finally:
        con.close()
    source = {}
    for rid, url, status in rows:
        parsed = urlparse(str(url))
        pmid = parsed.path.strip('/')
        if parsed.hostname != 'pubmed.ncbi.nlm.nih.gov' or not pmid.isdigit():
            continue
        if pmid in source:
            raise ValueError(f'Duplicate PMID in crawl_status: {pmid}')
        source[pmid] = (int(rid), str(url), str(status or ''))
    if not source:
        raise ValueError('No PubMed PMID URLs in crawl_status')
    return source


def _xml_text(node):
    return ''.join(node.itertext()).strip() if node is not None else ''


def _ncbi_date(article):
    for base in ('./MedlineCitation/Article/ArticleDate',
                 './MedlineCitation/Article/Journal/JournalIssue/PubDate',
                 './BookDocument/ArticleDate'):
        date = article.find(base)
        if date is None:
            continue
        year, month, day = (date.findtext(name) for name in ('Year', 'Month', 'Day'))
        if not (year and month and day):
            continue
        if not month.isdigit():
            for fmt in ('%b', '%B'):
                try:
                    month = str(datetime.strptime(month, fmt).month)
                    break
                except ValueError:
                    pass
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except (TypeError, ValueError):
            continue
    return None


def _parse_ncbi_xml(raw):
    root = ET.fromstring(raw)
    if root.tag.rsplit('}', 1)[-1] != 'PubmedArticleSet':
        raise ValueError(f'Unexpected NCBI EFetch root: {root.tag}')
    if root.find('.//ERROR') is not None:
        raise ValueError('NCBI EFetch returned an XML error instead of articles')
    results = {}
    for article in list(root.findall('PubmedArticle')) + list(root.findall('PubmedBookArticle')):
        pmid = (article.findtext('./MedlineCitation/PMID')
                or article.findtext('./BookDocument/PMID') or '').strip()
        if not pmid.isdigit():
            continue
        title_node = article.find('./MedlineCitation/Article/ArticleTitle')
        if title_node is None:
            title_node = article.find('./BookDocument/ArticleTitle')
        title = _xml_text(title_node)
        citation = article.find('./MedlineCitation')
        if citation is None:
            citation = article.find('./BookDocument')
        abstract_parts = []
        if citation is not None:
            for part in citation.findall('.//Abstract/AbstractText'):
                body = _xml_text(part)
                if not body:
                    continue
                label = (part.get('Label') or '').strip()
                if label:
                    abstract_parts.append('<h4>' + html.escape(label, quote=False)
                                          + '</h4>' + html.escape(body, quote=False))
                else:
                    abstract_parts.append(html.escape(body, quote=False))
        keywords = []
        if citation is not None:
            keywords = [_xml_text(k) for k in citation.findall('.//KeywordList/Keyword')]
        journal = _xml_text(article.find('./MedlineCitation/Article/Journal/Title'))
        if title or abstract_parts:
            results[pmid] = {
                'pmid': pmid, 'title': title,
                'abstractText': ' '.join(abstract_parts),
                'keywordList': {'keyword': [k for k in keywords if k]},
                'firstPublicationDate': _ncbi_date(article),
                'journalInfo': {'journal': {'title': journal or None}},
            }
    return results


def _read_zstd(path):
    with path.open('rb') as compressed, zstd.ZstdDecompressor().stream_reader(compressed) as stream:
        return stream.read()


def _iter_pubmed_cache(pubmed_dir):
    for path in sorted(pubmed_dir.glob('batch_*.json.zst')):
        records = json.loads(_read_zstd(path).decode('utf-8'))
        if not isinstance(records, list):
            raise ValueError(f'Expected a JSON list: {path}')
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f'Invalid PubMed record: {path}')
            yield path, 'europe_pmc', record
    for path in sorted(pubmed_dir.glob('ncbi_efetch_*.xml.zst')):
        for record in _parse_ncbi_xml(_read_zstd(path)).values():
            yield path, 'ncbi_efetch', record


def _cache_inventory(pubmed_dir, source):
    inventory, duplicate_records = {}, 0
    last_path, files_seen = None, 0
    for path, kind, record in _iter_pubmed_cache(pubmed_dir):
        if path != last_path:
            files_seen += 1
            last_path = path
            if files_seen == 1 or files_seen % 10 == 0:
                print(f'PubMed cache: reading file {files_seen}, '
                      f'{len(inventory):,} unique PMIDs so far: {path.name}', flush=True)
        pmid = str(record.get('pmid') or '').strip()
        if pmid not in source or not (record.get('title') or record.get('abstractText')):
            continue
        if pmid in inventory:
            duplicate_records += 1
            continue
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
        inventory[pmid] = (path.name, kind, fetched_at)
    return inventory, duplicate_records


def _ncbi_attempted_pmids(pubmed_dir):
    attempted = set()
    for path in pubmed_dir.glob('ncbi_efetch_*.request.json'):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if not (pubmed_dir / meta['xml_file']).is_file():
            raise FileNotFoundError(f'NCBI request metadata has no XML: {path}')
        attempted.update(str(pmid) for pmid in meta['pmids'])
    return attempted


def _fetch_ncbi_missing(pubmed_dir, pmids, client=None):
    pending = sorted(set(pmids) - _ncbi_attempted_pmids(pubmed_dir), key=int)
    if len(pending) > 500:
        raise ValueError(f'{len(pending):,} PMID absent; expected at most 229. Check Drive/cache first.')
    print(f'NCBI EFetch: {len(pending):,} PMID(s) to request in '
          f'{(len(pending) + NCBI_BATCH_SIZE - 1) // NCBI_BATCH_SIZE} batch(es).', flush=True)
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=90.0, http2=True)
    fetched_batches = 0
    try:
        for start in range(0, len(pending), NCBI_BATCH_SIZE):
            chunk = pending[start:start + NCBI_BATCH_SIZE]
            print(f'NCBI EFetch: requesting batch {start // NCBI_BATCH_SIZE + 1}, '
                  f'{len(chunk):,} PMID(s)...', flush=True)
            digest = hashlib.sha1(','.join(chunk).encode('ascii')).hexdigest()[:12]
            stem = f'ncbi_efetch_{start // NCBI_BATCH_SIZE:03d}_{digest}'
            xml_file = pubmed_dir / f'{stem}.xml.zst'
            meta_file = pubmed_dir / f'{stem}.request.json'
            params = {'db': 'pubmed', 'id': ','.join(chunk), 'retmode': 'xml',
                      'tool': 'ViBioMIR'}
            api_key = os.environ.get('NCBI_API_KEY')
            if api_key:
                params['api_key'] = api_key
            for attempt, delay in enumerate((1, 3, 10, 30), 1):
                try:
                    response = client.get(NCBI_EFETCH_URL, params=params)
                    response.raise_for_status()
                    hits = _parse_ncbi_xml(response.content)
                    unexpected = set(hits) - set(chunk)
                    if unexpected:
                        raise ValueError(f'NCBI returned unrequested PMIDs: {sorted(unexpected)[:5]}')
                    break
                except (httpx.HTTPError, ET.ParseError) as exc:
                    if attempt == 4 or (isinstance(exc, httpx.HTTPStatusError)
                                        and exc.response.status_code in (401, 403)):
                        raise RuntimeError(f'NCBI EFetch failed; DB was not changed: {exc}') from exc
                    time.sleep(delay)
            raw = zstd.ZstdCompressor(level=6).compress(response.content)
            temp_xml = xml_file.with_suffix(xml_file.suffix + '.tmp')
            temp_xml.write_bytes(raw)
            os.replace(temp_xml, xml_file)
            meta = {'pmids': chunk, 'xml_file': xml_file.name,
                    'returned': sorted(hits, key=int),
                    'fetched_at': datetime.now(timezone.utc).isoformat()}
            temp_meta = meta_file.with_suffix(meta_file.suffix + '.tmp')
            temp_meta.write_text(json.dumps(meta, indent=2), encoding='utf-8')
            os.replace(temp_meta, meta_file)
            fetched_batches += 1
            print(f'NCBI EFetch batch {fetched_batches}: {len(hits):,}/{len(chunk):,} PMID; {xml_file.name}', flush=True)
            time.sleep(0.4 if not api_key else 0.12)
    finally:
        if own_client:
            client.close()
    return fetched_batches


def _sync_pubmed_status(status_db, pubmed_dir, source, inventory, attempted):
    wal = Path(str(status_db) + '.wal')
    if wal.exists():
        raise RuntimeError(f'DuckDB WAL exists; stop other DB writers first: {wal}')
    print('PubMed: backing up and updating status.duckdb...', flush=True)
    backup_dir = status_db.parent / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = backup_dir / f'{status_db.name}.before_ncbi_{stamp}'
    shutil.copy2(status_db, backup)
    matched = [(source[pmid][0], pmid, file, fetched_at,
                'application/xml' if kind == 'ncbi_efetch' else 'application/json')
               for pmid, (file, kind, fetched_at) in inventory.items()]
    missing = [(source[pmid][0], pmid) for pmid in sorted(attempted - set(inventory), key=int)
               if pmid in source and source[pmid][2] != 'ok']
    con = duckdb.connect(str(status_db))
    try:
        stage = pd.DataFrame(matched, columns=['id', 'pmid', 'batch_file', 'fetched_at', 'content_type'])
        con.register('_pubmed_found', stage)
        con.register('_pubmed_missing', pd.DataFrame(missing, columns=['id', 'pmid']))
        con.execute('BEGIN TRANSACTION')
        try:
            con.execute("""CREATE TABLE IF NOT EXISTS pubmed_cache_index (
                id BIGINT PRIMARY KEY, pmid VARCHAR, batch_file VARCHAR,
                fetched_at TIMESTAMP, imported_at TIMESTAMP)""")
            con.execute("""DELETE FROM pubmed_cache_index AS dst
                USING _pubmed_found AS src WHERE dst.id=src.id""")
            con.execute("""INSERT INTO pubmed_cache_index
                SELECT id,pmid,batch_file,fetched_at,current_timestamp FROM _pubmed_found""")
            con.execute("""UPDATE crawl_status AS dst SET
                status='ok', final_url=dst.url, http_code=NULL,
                content_type=src.content_type, n_bytes=NULL,
                fetched_at=src.fetched_at, shard=NULL, sha1_raw=NULL, error_msg=NULL,
                attempts=COALESCE(dst.attempts,0)+1
                FROM _pubmed_found AS src
                WHERE dst.id=src.id AND dst.status IS DISTINCT FROM 'ok'""")
            if missing:
                con.execute("""UPDATE crawl_status AS dst SET
                    status='http_error', http_code=NULL,
                    fetched_at=current_timestamp,
                    error_msg='NCBI EFetch returned no article for PMID',
                    attempts=COALESCE(dst.attempts,0)+1
                    FROM _pubmed_missing AS src
                    WHERE dst.id=src.id AND dst.status IS DISTINCT FROM 'ok'
                      AND (dst.status IS DISTINCT FROM 'http_error'
                        OR dst.error_msg IS DISTINCT FROM 'NCBI EFetch returned no article for PMID')""")
            confirmed = con.execute("""SELECT count(*) FROM crawl_status AS dst
                JOIN _pubmed_found AS src ON dst.id=src.id WHERE dst.status='ok'""").fetchone()[0]
            if confirmed != len(matched):
                raise RuntimeError(f'Only {confirmed:,}/{len(matched):,} cached PMIDs are ok')
            con.execute('COMMIT')
        except Exception:
            con.execute('ROLLBACK')
            raise
        con.execute('CHECKPOINT')
    finally:
        con.close()
    print(f'Status DB updated in place: {len(matched):,} ok, {len(missing):,} NCBI missing; backup: {backup}', flush=True)
    return backup


def _pubmed_doc_schema():
    block = pa.struct([('type', pa.string()), ('level', pa.int64()),
                       ('role', pa.string()), ('text', pa.string()),
                       ('char_start', pa.int64()), ('char_end', pa.int64())])
    return pa.schema([
        ('id', pa.int64()), ('url', pa.string()), ('final_url', pa.string()),
        ('lang', pa.string()), ('lang_conf', pa.float64()),
        ('title', pa.string()), ('published_date', pa.string()),
        ('site', pa.string()), ('category', pa.string()),
        ('text', pa.string()), ('blocks', pa.list_(block)),
        ('n_chars', pa.int64()), ('sha1_text', pa.string()),
        ('dup_group', pa.int64()), ('extractor', pa.string()),
        ('quality_flags', pa.list_(pa.string())), ('schema_version', pa.int64()),
    ])


def _check_pubmed_doc(doc):
    text = doc['text']
    blocks = doc['blocks']
    if doc['n_chars'] != len(text) or text != '\n\n'.join(block['text'] for block in blocks):
        raise ValueError(f'PubMed text/block mismatch: {doc["id"]}')
    if any(text[block['char_start']:block['char_end']] != block['text'] for block in blocks):
        raise ValueError(f'PubMed block offset mismatch: {doc["id"]}')


def _write_pubmed_docs(pubmed_dir, docs_dir, report_dir, source, cleaner, inventory):
    print(f'PubMed: extracting {len(source):,} docs into Parquet...', flush=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = docs_dir / f'pubmed_{run_id}'
    output.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    schema = _pubmed_doc_schema()
    seen, duplicates, text_groups, files = set(), 0, {}, []
    counts = {'europe_pmc': 0, 'ncbi_efetch': 0, 'placeholder': 0}
    batch = []

    def flush():
        if not batch:
            return
        target = output / f'docs_{len(files):05d}.parquet'
        temp = target.with_suffix('.parquet.tmp')
        pq.write_table(pa.Table.from_pylist(batch, schema=schema), temp, compression='zstd')
        os.replace(temp, target)
        files.append(target.name)
        print(f'PubMed docs: {sum(counts.values()):,}/{len(source):,}; {target.name}', flush=True)
        batch.clear()

    for _path, kind, record in _iter_pubmed_cache(pubmed_dir):
        pmid = str(record.get('pmid') or '').strip()
        if (pmid not in inventory or inventory[pmid][0] != _path.name
                or not (record.get('title') or record.get('abstractText'))):
            continue
        if pmid in seen:
            duplicates += 1
            continue
        rid, url, _status = source[pmid]
        doc = cleaner(record, rid, url)
        if kind == 'ncbi_efetch':
            doc['extractor'] = 'pubmed_ncbi_efetch_v1'
        _check_pubmed_doc(doc)
        digest = doc['sha1_text']
        if doc['text']:
            doc['dup_group'] = text_groups.setdefault(digest, rid)
            if doc['dup_group'] == rid:
                doc['dup_group'] = None
        batch.append(doc)
        seen.add(pmid)
        counts[kind] += 1
        if len(batch) >= PUBMED_DOC_SHARD_SIZE:
            flush()

    missing = sorted(set(source) - seen, key=int)
    with (report_dir / 'pubmed_missing_after_ncbi.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'pmid', 'url'])
        for pmid in missing:
            rid, url, _status = source[pmid]
            writer.writerow([rid, pmid, url])
            batch.append({'id': rid, 'url': url, 'final_url': url,
                          'lang': 'en', 'lang_conf': 0.0, 'title': None,
                          'published_date': None, 'site': 'pubmed.ncbi.nlm.nih.gov',
                          'category': None, 'text': '', 'blocks': [], 'n_chars': 0,
                          'sha1_text': None, 'dup_group': None, 'extractor': 'placeholder',
                          'quality_flags': ['missing', 'ncbi_not_found'], 'schema_version': 2})
            counts['placeholder'] += 1
            if len(batch) >= PUBMED_DOC_SHARD_SIZE:
                flush()
    flush()
    if sum(counts.values()) != len(source) or len(seen) != len(inventory):
        raise RuntimeError('PubMed output coverage does not match cache/source inventory')
    report = {'expected_ids': len(source), 'content_docs': len(seen),
              'placeholder_docs': len(missing), 'europe_pmc_docs': counts['europe_pmc'],
              'ncbi_docs': counts['ncbi_efetch'], 'duplicate_cache_records': duplicates,
              'output': str(output), 'files': files}
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (report_dir / 'pubmed_extraction_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: val for key, val in report.items() if key != 'files'},
                     ensure_ascii=False, indent=2), flush=True)
    return report


def run_pubmed_completion(status_db, pubmed_dir, docs_dir, report_dir, cleaner, client=None):
    """Fetch uncached PMIDs, sync the mounted Drive DB in place, and clean docs."""
    status_db, pubmed_dir = Path(status_db), Path(pubmed_dir)
    print(f'PubMed: starting; DB={status_db}; cache={pubmed_dir}', flush=True)
    if not status_db.is_file():
        raise FileNotFoundError(status_db)
    if not list(pubmed_dir.glob('batch_*.json.zst')):
        raise FileNotFoundError(f'No Europe PMC batch files: {pubmed_dir}')
    if Path(str(status_db) + '.wal').exists():
        raise RuntimeError('Stop other DuckDB writers before PubMed completion')
    print('PubMed: reading PMID IDs from status.duckdb...', flush=True)
    source = _pubmed_source_rows(status_db)
    print(f'PubMed: scanning cached batches for {len(source):,} PMID IDs...', flush=True)
    inventory, duplicates = _cache_inventory(pubmed_dir, source)
    missing = set(source) - set(inventory)
    print(f'PubMed DB={len(source):,}, cached={len(inventory):,}, missing={len(missing):,}, duplicates={duplicates:,}', flush=True)
    if len(missing) > 500:
        raise ValueError('Too many PubMed PMIDs are missing; check the mounted Drive account and cache')
    fetched_batches = _fetch_ncbi_missing(pubmed_dir, missing, client=client)
    inventory, duplicates = _cache_inventory(pubmed_dir, source)
    attempted = _ncbi_attempted_pmids(pubmed_dir)
    backup = _sync_pubmed_status(status_db, pubmed_dir, source, inventory, attempted)
    report = _write_pubmed_docs(pubmed_dir, Path(docs_dir), Path(report_dir),
                                source, cleaner, inventory)
    report.update(ncbi_new_batches=fetched_batches,
                  ncbi_attempted_pmids=len(attempted & set(source)),
                  status_backup=str(backup),
                  missing_report=str(Path(report_dir) / 'pubmed_missing_after_ncbi.csv'))
    summary = json.dumps(report, ensure_ascii=False, indent=2)
    (Path(report['output']) / 'manifest.json').write_text(summary, encoding='utf-8')
    (Path(report_dir) / 'pubmed_extraction_summary.json').write_text(summary, encoding='utf-8')
    return report


print('PubMed helper loaded. Run the next cell to fetch missing PMIDs and export docs.', flush=True)
