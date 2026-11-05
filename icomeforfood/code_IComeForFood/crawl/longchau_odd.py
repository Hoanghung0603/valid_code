"""Collect raw Long Chau HTML for odd IDs present in crawl_status.

Default is an offline plan. --scan collects remaining odd IDs without a separate probe,
pausing between small batches and stopping immediately on an access block.
--probe and --run remain available for the manual two-step workflow.
The source of IDs and URLs is the selected DuckDB file, not a Parquet corpus.
The shared crawl_status database is read through a snapshot and is never reset.
--retry-blocked includes previously blocked odd IDs only after the domain cooldown.
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
import httpx
import zstandard as zstd
from charset_normalizer import from_bytes

DOMAIN = 'nhathuoclongchau.com.vn'
COOLDOWN_HOURS = 6
MIN_DELAY = 30.0
MAX_BATCH = 50
STATUS_COLUMNS = ['id','url','final_url','lang_expected','status','http_code','content_type',
                  'n_bytes','attempts','fetched_at','shard','sha1_raw','error_msg','shard_file']
CHALLENGE_MARKERS = ('cf-chl-','challenge-platform','just a moment','verify you are human',
                     'checking your browser','captcha','verify.html',
                     'sorry, you have been blocked','access denied')
log = logging.getLogger('longchau_odd')


def now():
    return datetime.now(timezone.utc)


def timestamp(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace('Z','+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def is_longchau(url):
    return (urlparse(str(url)).hostname or '').lower().removeprefix('www.') == DOMAIN


def atomic_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as handle:
        json.dump(data,handle,ensure_ascii=False,indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp,path)


def last_status(path):
    rows = {}
    if path.is_file():
        with path.open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                if (row.get('id','').isdigit() and row.get('status') and None not in row
                        and all(value is not None for value in row.values())):
                    rows[int(row['id'])] = row
    return rows


def source_rows(corpus=None, database=None):
    """Read only DB-backed IDs/URLs; corpus is ignored for old caller compatibility."""
    if database is None:
        if corpus is None:
            raise ValueError('A source --status-db is required')
        database = corpus
    database = Path(database).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f'Source status DB not found: {database}')
    rows = []
    statuses = {}
    with tempfile.TemporaryDirectory(prefix='longchau_odd_status_') as directory:
        copy = Path(directory)/'status.duckdb'
        try:
            shutil.copy2(database,copy)
            wal = Path(str(database)+'.wal')
            if wal.is_file():
                shutil.copy2(wal,Path(str(copy)+'.wal'))
        except PermissionError as exc:
            raise RuntimeError('Close the writer using status.duckdb before reading a snapshot.') from exc
        con = duckdb.connect(str(copy),read_only=True)
        try:
            for rid,url,state,code,fetched in con.execute("""SELECT id,url,status,http_code,fetched_at
                FROM crawl_status WHERE
                regexp_replace(lower(split_part(url,'/',3)), '^www[.]', '')=?
                ORDER BY id""",[DOMAIN]).fetchall():
                statuses[int(rid)] = dict(status=state,http_code=code,fetched_at=str(fetched) if fetched else None)
                if int(rid) % 2 == 1 and is_longchau(url):
                    rows.append(dict(id=int(rid), url=str(url)))
        finally:
            con.close()
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('crawl_status contains duplicate Long Chau IDs')
    return sorted(rows,key=lambda r:r['id']),statuses


def select_corpus_rows(rows, corpus):
    """Keep the reviewed CSV order and refuse IDs/URLs outside the source DB."""
    by_id = {row['id']: row['url'] for row in rows}
    selected = []
    seen = set()
    with Path(corpus).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'id', 'url'} <= set(reader.fieldnames or ()):
            raise ValueError('Long Chau corpus CSV must contain id,url columns')
        for entry in reader:
            rid = int(entry['id'])
            url = entry['url']
            if rid in seen or rid % 2 != 1 or not is_longchau(url):
                raise ValueError(f'Invalid or duplicate Long Chau odd ID in corpus: {rid}')
            if by_id.get(rid) != url:
                raise ValueError(f'Corpus ID/URL differs from status DB: {rid}')
            selected.append(dict(id=rid, url=url))
            seen.add(rid)
    return selected


def raw_ids(out):
    found = set()
    for path in sorted((out/'html').glob('home_shard_*.jsonl.zst')):
        with path.open('rb') as compressed,zstd.ZstdDecompressor().stream_reader(compressed) as reader,io.TextIOWrapper(reader,encoding='utf-8') as lines:
            for line in lines:
                record = json.loads(line)
                rid = int(record['id'])
                if rid % 2 != 1:
                    raise ValueError(f'Even ID in odd-only raw output: {rid}')
                base64.b64decode(record['body_b64'],validate=True)
                found.add(rid)
    return found


def queue_rows(rows, database_rows, local_rows, saved_raw_ids, retry_blocked=False):
    skip_states = {'ok','non_html','empty'}
    skip_codes = {'404','410'}
    if not retry_blocked:
        skip_states.add('blocked')
        skip_codes |= {'401','403','429'}
    selected = []
    for row in rows:
        rid = row['id']
        if rid not in database_rows or rid % 2 != 1 or not is_longchau(row['url']):
            continue
        if database_rows[rid].get('status') in {'ok','non_html','empty'}:
            continue
        status = local_rows.get(rid) or database_rows.get(rid) or {}
        code = str(status.get('http_code') or '')
        if rid in saved_raw_ids or status.get('status') in skip_states or code in skip_codes:
            continue
        selected.append(row)
    return selected


def initial_state(rows, database_rows, local_rows):
    by_id = {r['id']:r for r in rows}
    blocks = []
    for rid,row in {**database_rows,**local_rows}.items():
        if row.get('status')=='blocked' or str(row.get('http_code') or '') in {'401','403','429'}:
            stamp = timestamp(row.get('fetched_at'))
            if stamp:
                blocks.append((stamp,rid))
    if not blocks:
        return dict(kind='unprobed',next_probe_at=None)
    stamp,rid = max(blocks)
    candidate = by_id.get(rid) or (rows[0] if rows else None)
    if candidate is None:
        return dict(kind='unprobed',next_probe_at=None)
    return dict(kind='blocked',blocked_at=stamp.isoformat(),next_probe_at=(stamp+timedelta(hours=COOLDOWN_HOURS)).isoformat(),
                blocked_id=candidate['id'],blocked_url=candidate['url'],reason='Existing blocked status; do not reset or retry continuously')


def gate(state, probe, current=None, scan=False):
    current = current or now()
    deadline = timestamp(state.get('next_probe_at'))
    if deadline and current < deadline:
        return f'Cooldown active until {deadline.astimezone().isoformat()}; zero requests allowed.'
    if state.get('kind') in {'blocked','unprobed'} and not (probe or scan):
        return 'Use --scan, or a single --probe before --run. Blocked IDs are never reset to pending.'
    return None


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another Long Chau job is already running.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt':
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


@contextmanager
def keep_system_awake():
    """Keep a long local crawl running while the Windows machine is on."""
    if os.name!='nt':
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
    def __init__(self,out):
        self.out = out
        self.html = out/'html'
        self.html.mkdir(parents=True,exist_ok=True)
        self.pending = []
        self.idx = max((int(p.name.split('_')[2].split('.')[0]) for p in self.html.glob('home_shard_*.jsonl.zst')),default=-1)+1
        path = out/'status.csv'
        new = not path.exists() or path.stat().st_size==0
        cut = not new and not path.read_bytes().endswith(b'\n')
        self.handle = path.open('a',newline='',encoding='utf-8')
        if cut:
            self.handle.write('\n')
        self.writer = csv.DictWriter(self.handle,fieldnames=STATUS_COLUMNS)
        if new:
            self.writer.writeheader()
            self.sync()

    def sync(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def add_status(self,row):
        self.writer.writerow(row)
        self.sync()

    def add_raw(self,record,row):
        if int(record['id'])%2!=1 or int(row['id'])%2!=1:
            raise ValueError('Only odd source IDs may be saved')
        self.pending.append((record,row))

    def flush(self):
        if not self.pending:
            return
        path = self.html/f'home_shard_{self.idx:05d}.jsonl.zst'
        if path.exists():
            raise FileExistsError(path)
        temp = path.with_name(path.name+'.tmp')
        raw = ''.join(json.dumps(rec,ensure_ascii=False)+'\n' for rec,_ in self.pending).encode('utf-8')
        with temp.open('wb') as handle:
            handle.write(zstd.ZstdCompressor(level=3).compress(raw))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp,path)
        for _,row in self.pending:
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


class StopRun(Exception):
    pass


def encoding(raw,content_type):
    match = __import__('re').search(r'charset=["\s]*([\w-]+)',content_type or '',__import__('re').I)
    if match:
        return match.group(1)
    best = from_bytes(raw[:32768]).best()
    return best.encoding if best else 'utf-8'


def wait_for_rest(stop_file,seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if stop_file.exists():
            return False
        time.sleep(min(1.0,max(0.0,deadline-time.monotonic())))
    return not stop_file.exists()


def collect(rows,out,state,args,transport=None):
    """One client, one request at a time; no HTTP retries, proxy, browser or IP changes."""
    scan=getattr(args,'scan',False)
    writer = RawOutput(out)
    attempted = ok = 0
    request_count = 0
    previous_request = None
    previous_stamp = timestamp(state.get('last_request_at') or state.get('probed_at'))
    if previous_stamp:
        previous_request = time.monotonic()-max(0.0,(now()-previous_stamp).total_seconds())
    last_save = time.monotonic()
    stop_file = out/'STOP'
    def before_request(request):
        nonlocal previous_request,request_count
        if not is_longchau(str(request.url)):
            raise StopRun('Cross-domain redirect refused; no request was sent to that host.')
        if args.probe and request_count>=1:
            raise StopRun('Probe used its one-request budget; redirect was not followed.')
        if previous_request is not None:
            wait = max(0.0,args.delay-(time.monotonic()-previous_request))
            deadline = time.monotonic()+wait
            while time.monotonic()<deadline:
                if stop_file.exists():
                    raise StopRun('STOP requested')
                time.sleep(min(0.5,max(0.0,deadline-time.monotonic())))
        if stop_file.exists():
            raise StopRun('STOP requested')
        previous_request = time.monotonic()
        request_count += 1
        state['last_request_at'] = now().isoformat()
        atomic_json(out/'cooldown.json',state)
    try:
        with httpx.Client(http2=False,trust_env=False,timeout=30.0,transport=transport,
                          limits=httpx.Limits(max_connections=1,max_keepalive_connections=1),
                          headers={'User-Agent':'ViBioMIRResearchCrawler/1.0','Accept':'text/html,application/xhtml+xml',
                                   'Accept-Language':'vi'},event_hooks={'request':[before_request]}) as client:
            selected=rows if scan else rows[:1 if args.probe else args.batch_size]
            for index,item in enumerate(selected):
                if int(item['id'])%2!=1 or not is_longchau(item['url']):
                    raise ValueError('Refusing to request an even ID or another domain')
                if stop_file.exists():
                    break
                row = dict.fromkeys(STATUS_COLUMNS)
                row.update(id=item['id'],url=item['url'],final_url=item['url'],lang_expected='vi',attempts=1,
                           fetched_at=now().isoformat(),status='http_error',n_bytes=0)
                try:
                    response = client.get(item['url'],follow_redirects=not args.probe)
                    attempted += 1
                    raw = response.content
                    content_type = response.headers.get('content-type','')
                    row.update(final_url=str(response.url),http_code=response.status_code,content_type=content_type,n_bytes=len(raw))
                    low = raw[:200000].decode('utf-8',errors='ignore').lower()
                    challenge = (response.headers.get('cf-mitigated', '').lower() == 'challenge'
                                 or any(m in low or m in str(response.url).lower() for m in CHALLENGE_MARKERS))
                    blocked = response.status_code in {401,403,429,503} or challenge
                    if blocked:
                        row.update(status='blocked',error_msg='Access refused or verification page; domain stopped immediately')
                        writer.add_status(row)
                        deadline = now()+timedelta(hours=COOLDOWN_HOURS)
                        retry = response.headers.get('retry-after')
                        if retry:
                            try:
                                allowed = now()+timedelta(seconds=int(retry)) if retry.isdigit() else parsedate_to_datetime(retry)
                                deadline = max(deadline,allowed)
                            except (ValueError,TypeError):
                                pass
                        state = dict(kind='blocked',last_request_at=state.get('last_request_at'),blocked_id=item['id'],blocked_url=item['url'],blocked_at=now().isoformat(),
                                     next_probe_at=deadline.isoformat(),http_code=response.status_code,reason=row['error_msg'])
                        atomic_json(out/'cooldown.json',state)
                        log.warning('BLOCKED id=%s HTTP=%s; stop all requests until %s',item['id'],response.status_code,deadline.isoformat())
                        break
                    if response.status_code==200 and raw and (not content_type or 'html' in content_type.lower()):
                        row.update(status='ok',sha1_raw=hashlib.sha1(raw).hexdigest())
                        record = dict(id=item['id'],final_url=str(response.url),http_code=200,content_type=content_type,
                                      encoding=encoding(raw,content_type),body_b64=base64.b64encode(raw).decode('ascii'))
                        writer.add_raw(record,row)
                        ok += 1
                        state = dict(kind='probe_ok' if args.probe else 'scan_ready',
                                     last_request_at=state.get('last_request_at'),
                                     probed_at=now().isoformat(),next_probe_at=None)
                        atomic_json(out/'cooldown.json',state)
                    else:
                        row['error_msg'] = f'HTTP {response.status_code}; no retry'
                        writer.add_status(row)
                        if args.probe or response.status_code>=500:
                            break
                    log.info('id=%s status=%s HTTP=%s bytes=%s completed=%s',item['id'],row['status'],row['http_code'],len(raw),attempted)
                    if time.monotonic()-last_save>=60:
                        writer.flush()
                        last_save = time.monotonic()
                    if scan and (index+1)%args.batch_size==0 and index+1<len(selected):
                        writer.flush()
                        state=dict(kind='batch_rest',last_request_at=state.get('last_request_at'),
                                   next_probe_at=(now()+timedelta(hours=3)).isoformat(),reason='Rest between scan batches')
                        atomic_json(out/'cooldown.json',state)
                        log.info('Scan saved %s URLs; rest 3 hours before next batch. STOP can interrupt this rest.',attempted)
                        if not wait_for_rest(stop_file,3*60*60):
                            break
                        state=dict(kind='scan_ready',last_request_at=state.get('last_request_at'),next_probe_at=None)
                        atomic_json(out/'cooldown.json',state)
                except (httpx.HTTPError,StopRun) as exc:
                    row['error_msg'] = str(exc)
                    writer.add_status(row)
                    log.warning('Stopped without retry: %s',exc)
                    break
    finally:
        writer.close()
        if not args.probe and state.get('kind')!='blocked' and attempted:
            state = dict(kind='batch_rest',last_request_at=state.get('last_request_at'),next_probe_at=(now()+timedelta(hours=3)).isoformat(),reason='Rest between small batches')
            atomic_json(out/'cooldown.json',state)
        summary = dict(mode='scan' if scan else 'probe' if args.probe else 'batch',odd_ids_only=True,partial_domain=True,attempted=attempted,ok=ok,requests=request_count,
                       delay_seconds=args.delay,cooldown=state,output=str(out))
        atomic_json(out/'run_summary.json',summary)
    return summary


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument('--corpus',type=Path,default=repo/'crawl/longchau_tv_crawl_odd.csv',
                        help='Reviewed priority id,url CSV (default: crawl/longchau_tv_crawl_odd.csv)')
    default_db = repo/'statusduck.db' if (repo/'statusduck.db').is_file() else repo/'status.duckdb'
    parser.add_argument('--status-db',type=Path,default=Path(os.environ.get('R2AI_STATUS_DB', str(default_db))),
                        help='Existing DuckDB file containing crawl_status(id,url,status,http_code,fetched_at)')
    parser.add_argument('--retry-blocked',action='store_true',
                        help='Include DB/local blocked odd IDs after cooldown; stop again on the first block')
    parser.add_argument('--skip-first-odd',type=int,default=0,
                        help='Skip this many lowest Long Chau odd IDs from the source DB')
    parser.add_argument('--out',type=Path,default=Path('D:/r2ai_data/crawl/longchau_odd'))
    parser.add_argument('--delay',type=float,default=MIN_DELAY)
    parser.add_argument('--batch-size',type=int,default=MAX_BATCH)
    parser.add_argument('--engine',choices=('httpx','crawl4ai'),default='httpx',
                        help='Fetch engine; crawl4ai uses Chromium and saves its HTML snapshot')
    parser.add_argument('--continuous',action='store_true',
                        help='With Crawl4AI --scan, save each batch without a fixed 3-hour rest')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--probe',action='store_true')
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--scan',action='store_true',help='Scan remaining odd IDs without a separate probe; retain cooldown and stop-on-block')
    args = parser.parse_args(argv)
    if args.delay<MIN_DELAY or not 1<=args.batch_size<=MAX_BATCH:
        parser.error('Use --delay >=30 seconds and --batch-size between 1 and 50.')
    if args.continuous and (args.engine!='crawl4ai' or not args.scan):
        parser.error('--continuous requires --engine crawl4ai --scan.')
    if args.engine=='crawl4ai':
        try:
            import crawl4ai  # noqa: F401
        except ImportError as exc:
            parser.error(f'Crawl4AI is missing from this Python environment: {exc}')
    args.status_db = args.status_db.expanduser().resolve()
    db_rows,database = source_rows(database=args.status_db)
    all_rows = select_corpus_rows(db_rows,args.corpus)
    if not 0<=args.skip_first_odd<=len(all_rows):
        parser.error(f'--skip-first-odd must be between 0 and {len(all_rows)}.')
    rows = all_rows[args.skip_first_odd:]
    local = last_status(args.out/'status.csv')
    saved = raw_ids(args.out)
    selected = queue_rows(rows,database,local,saved,retry_blocked=args.retry_blocked)
    state_path = args.out/'cooldown.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else initial_state(rows,database,local)
    print(f'Source DB (read-only snapshot): {args.status_db}')
    print(f'Long Chau odd IDs in reviewed CSV: {len(all_rows):,}; skipped first: {args.skip_first_odd:,}; '
          f'eligible: {len(selected):,}; raw already saved here: {len(saved):,}')
    if args.skip_first_odd and rows:
        print(f'First ID still in queue: {rows[0]["id"]}')
    print(f'Include previously blocked IDs after cooldown: {args.retry_blocked}')
    print(f'Fetch engine: {args.engine}')
    print('Only partial Long Chau data is expected; this does not claim full-domain coverage.')
    restriction = gate(state,args.probe,scan=args.scan)
    if (args.engine=='crawl4ai' and args.probe and state.get('kind')=='blocked'
            and state.get('engine','httpx')=='httpx'):
        print('Prior block was from HTTPX; allowing one Crawl4AI browser probe.')
        restriction = None
    if (args.engine=='crawl4ai' and args.scan
            and state.get('kind') in {'blocked','unprobed'} and not restriction):
        restriction = 'Run --probe with Crawl4AI for one URL before --scan.'
    if restriction:
        print(restriction)
    if not args.probe and not args.run and not args.scan:
        print('Offline plan only: zero HTTP requests.')
        return 0
    if restriction:
        return 2
    if args.probe and state.get('kind')=='blocked':
        # Never send a stale cooldown URL or an ID absent from the selected source DB.
        candidate = next((row for row in rows
                          if row['id'] == int(state.get('blocked_id', -1))
                          and row['id'] not in saved
                          and database[row['id']].get('status') not in {'ok','non_html','empty'}), None)
        if candidate is not None:
            selected = [candidate]
    if not selected:
        print('No eligible odd IDs left; zero HTTP requests.')
        return 0
    with exclusive_lock(args.out.parent/'locks/nhathuoclongchau.lock'), keep_system_awake():
        args.out.mkdir(parents=True,exist_ok=True)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(args.out/'crawl.log',encoding='utf-8')])
        if args.engine=='crawl4ai':
            import asyncio
            import longchau_crawl4ai
            summary = asyncio.run(longchau_crawl4ai.collect(selected,args.out,state,args,base=sys.modules[__name__]))
        else:
            summary = collect(selected,args.out,state,args)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        if summary['cooldown'].get('kind')=='blocked':
            return 3
        return 4 if args.probe and summary['ok']!=1 else 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Stopped; completed raw shards and status rows were saved.',file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
