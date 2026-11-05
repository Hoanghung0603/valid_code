"""Local entry point for the reviewed, self-contained notebook implementations."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
REVIEW = 'reviews/2026-10-22_longchau_common_crawl.md'
REQUIRED = {'pandas':'pandas','pyarrow':'pyarrow','duckdb':'duckdb','httpx':'httpx',
            'h2':'h2','zstandard':'zstandard','charset_normalizer':'charset-normalizer',
            'dotenv':'python-dotenv','tqdm':'tqdm','nest_asyncio':'nest-asyncio',
            'selectolax':'selectolax','trafilatura':'trafilatura','resiliparse':'resiliparse'}


def paths(args):
    root = Path(args.data_root).expanduser().resolve()
    if args.status_db:
        database = Path(args.status_db).expanduser().resolve()
    elif (root/'crawl/status.duckdb').is_file():
        database = root/'crawl/status.duckdb'
    elif (REPO/'statusduck.db').is_file():
        database = REPO/'statusduck.db'
    else:
        database = REPO/'status.duckdb'
    return root, database


def check_environment():
    missing = [package for module,package in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError('Missing packages: '+', '.join(missing)+
                           '\nRun: .\\scripts\\run_local.ps1 install')


def prepare(root):
    for name in ('crawl/html','crawl/pubmed','docs','reports','logs'):
        (root/name).mkdir(parents=True,exist_ok=True)
    for name in ('domain_plan.csv','container_hints.csv','pilot2_sample.csv'):
        source, target = REPO/'crawl'/name, root/'crawl'/name
        if source.is_file() and not target.exists():
            shutil.copy2(source,target)
            print(f'Copied configuration: {target}')


def preflight(args):
    root, database = paths(args)
    print(f'Repository: {REPO}\nData root: {root}\nStatus DB: {database}\nReview: {REVIEW}')
    missing = [package for module,package in REQUIRED.items() if importlib.util.find_spec(module) is None]
    print('Missing packages: '+(', '.join(missing) if missing else 'none'))
    corpus = root/'raw/links_corpus_vi_en_zh.parquet'
    print(f'Corpus URL table: {corpus.is_file()}')
    html = sorted((root/'crawl/html').glob('shard_*.jsonl.zst'))
    print(f'Closed HTML shards: {len(html)}')
    print(f'PubMed cache batches: {len(list((root/"crawl/pubmed").glob("batch_*.json.zst")))}')
    print(f'Home status ready: {(root/"crawl/html_home/status.csv").is_file()}')
    for name in ('gold_html.zip','gold2_html.zip','gold3_html.zip'):
        print(f'{name}: {any((base/name).is_file() for base in (REPO,root))}')
    if database.is_file() and importlib.util.find_spec('duckdb'):
        import tempfile
        import duckdb
        with tempfile.TemporaryDirectory(prefix='r2ai_preflight_') as folder:
            copy = Path(folder)/'status.duckdb'
            try:
                shutil.copy2(database,copy)
                wal = Path(str(database)+'.wal')
                if wal.is_file():
                    shutil.copy2(wal,Path(str(copy)+'.wal'))
            except PermissionError as exc:
                raise RuntimeError('Status DB is locked. Stop the local writer before preflight.') from exc
            con = duckdb.connect(str(copy),read_only=True)
            try:
                total,ok,largest = con.execute("SELECT count(*),count(*) FILTER(WHERE status='ok'),max(shard) FROM crawl_status").fetchone()
                print(f'DB coverage: {ok:,}/{total:,} ok; largest previous shard: {largest}')
                print(con.execute("SELECT status,http_code,count(*) AS urls FROM crawl_status GROUP BY 1,2 ORDER BY urls DESC LIMIT 12").df().to_string(index=False))
            finally:
                con.close()
    if not html:
        print('Full extraction needs existing crawl/html shards synced from Drive/VPS first.')
        print('New local crawl shards will start above the shard numbers recorded in the DB.')
    print('No crawler or browser was started.')
    return 1 if missing or not corpus.is_file() or not database.is_file() else 0


def notebook(name):
    return json.loads((REPO/'notebooks'/f'{name}.ipynb').read_text(encoding='utf-8'))


def execute(nb, ident, ns):
    cell = next(c for c in nb['cells'] if c.get('id')==ident)
    exec(compile(''.join(cell['source']),f'{nb.get("name","notebook")}:{ident}','exec'),ns)


def crawler_context(args):
    root, database = paths(args)
    if not database.is_file():
        raise FileNotFoundError(f'Status DB not found: {database}; sync the existing DB before resuming.')
    prepare(root)
    os.environ['R2AI_DATA_ROOT'] = str(root)
    os.environ['R2AI_STATUS_DB'] = str(database)
    ns = {'__name__':'_r2ai_local','COLAB':False,'display':print}
    nb = notebook('01_data_pipeline')
    for ident in ('ee948b86','03180897'):
        execute(nb,ident,ns)
    corpus = root/'raw/links_corpus_vi_en_zh.parquet'
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    ns['df_links'] = ns['pd'].read_parquet(corpus)
    for ident in ('6048f8cf','d31dc248','0d3118f8','d356471b','ea80b6a7','7c6a152d',
                  '2a08247e','8e8120c6','crawl-review-23'):
        execute(nb,ident,ns)
    return nb,ns


def extract(args):
    root,database = paths(args)
    shards = sorted((root/'crawl/html').glob('shard_*.jsonl.zst'))
    if not shards:
        raise FileNotFoundError(f'No closed HTML shards in {root/"crawl/html"}; sync Drive/VPS shards first.')
    if not database.is_file():
        raise FileNotFoundError(database)
    prepare(root)
    os.environ['R2AI_DATA_ROOT'] = str(root)
    ns = {'__name__':'_r2ai_local','COLAB':False,'display':print}
    nb = notebook('03_gold_review_and_sample2')
    for ident in ('gold-project-paths','gold-project-root','gold-logging','gold-encoding',
                  'gold-pubmed-api','gold-clean-helpers','gold-extractors','gold-cleaner-selftest',
                  'gold-run','shard-runner','review-v112-cell-25'):
        execute(nb,ident,ns)
    return ns['run_shard_extraction'](shard_paths=shards,status_db=database,nt_approved=True,overwrite=False)


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Local crawler / v26 extraction from the reviewed notebooks')
    parser.add_argument('command',choices=['preflight','prepare','plan','crawl','longchau-odd','pubmed','vov','merge-home','extract','status','stop'])
    parser.add_argument('--data-root',default=os.environ.get('R2AI_DATA_ROOT','D:/r2ai_data'))
    parser.add_argument('--status-db',default=os.environ.get('R2AI_STATUS_DB'))
    parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--max-urls',type=int,help='VOV: cap this batch/scan')
    parser.add_argument('--all-vov',action='store_true',help='VOV: alias for --scan')
    parser.add_argument('--out',type=Path,help='VOV output directory (default: <data-root>/crawl/vov)')
    parser.add_argument('--home-dir',type=Path)
    parser.add_argument('--probe',action='store_true',help='Long Chau / VOV: probe one URL after cooldown')
    parser.add_argument('--scan',action='store_true',help='Long Chau / VOV: scan the remaining queue; respect cooldown')
    parser.add_argument('--retry-blocked',action='store_true',help='Long Chau odd: include blocked IDs after cooldown; stop on the first new block')
    parser.add_argument('--skip-first-odd',type=int,default=0,
                        help='Long Chau odd: skip this many lowest odd IDs already collected elsewhere')
    parser.add_argument('--run',action='store_true',help='Long Chau / VOV: collect one batch')
    parser.add_argument('--delay',type=float,help='Seconds between pages (Long Chau: 30; VOV: 2)')
    parser.add_argument('--batch-size',type=int,default=50)
    parser.add_argument('--engine',choices=['httpx','crawl4ai'],default='httpx',
                        help='Long Chau odd fetch engine')
    parser.add_argument('--continuous',action='store_true',
                        help='Long Chau Crawl4AI scan: no fixed rest between saved batches')
    args = parser.parse_args(argv)
    os.chdir(REPO)
    if args.command=='preflight':
        return preflight(args)
    root,database = paths(args)
    if args.command=='vov':
        import subprocess
        command=[sys.executable,str(REPO/'crawl/vov.py'),
                 '--status-db',str(database),'--out',str(args.out or root/'crawl/vov'),
                 '--delay',str(args.delay if args.delay is not None else 2.0),
                 '--batch-size',str(args.batch_size)]
        if args.probe:command.append('--probe')
        if args.run:command.append('--run')
        if args.scan or args.all_vov:command.append('--scan')
        if args.max_urls is not None:command.extend(['--limit',str(args.max_urls)])
        return subprocess.call(command,cwd=REPO)
    if args.command=='longchau-odd':
        import subprocess
        command=[sys.executable,str(REPO/'crawl/longchau_odd.py'),
                 '--status-db',str(database),'--out',str(root/'crawl/longchau_odd'),
                 '--delay',str(args.delay if args.delay is not None else 30.0),'--batch-size',str(args.batch_size),
                 '--engine',args.engine,'--skip-first-odd',str(args.skip_first_odd)]
        if args.probe:command.append('--probe')
        if args.run:command.append('--run')
        if args.scan:command.append('--scan')
        if args.retry_blocked:command.append('--retry-blocked')
        if args.continuous:command.append('--continuous')
        return subprocess.call(command,cwd=REPO)
    if args.command=='stop':
        if not (root/'crawl').is_dir():
            raise FileNotFoundError(root/'crawl')
        (root/'crawl/STOP').touch()
        if (root/'crawl/longchau_odd').is_dir():
            (root/'crawl/longchau_odd/STOP').touch()
        if (root/'crawl/vov').is_dir():
            (root/'crawl/vov/STOP').touch()
        print('Stop requested; wait for the current batch/shard to be saved.')
        return 0
    check_environment()
    if args.command=='prepare':
        prepare(root)
        return preflight(args)
    ns = None
    awake = None
    if os.name=='nt':
        import ctypes
        awake = ctypes.windll.kernel32.SetThreadExecutionState
        awake(0x80000001)
    try:
        if args.command=='plan':
            import tempfile
            with tempfile.TemporaryDirectory(prefix='r2ai_crawl_plan_') as folder:
                snapshot=Path(folder)/'status.duckdb'
                shutil.copy2(database,snapshot)
                wal=Path(str(database)+'.wal')
                if wal.is_file():
                    shutil.copy2(wal,Path(str(snapshot)+'.wal'))
                from argparse import Namespace
                dry_args=Namespace(**vars(args))
                dry_args.status_db=str(snapshot)
                nb,ns=crawler_context(dry_args)
                execute(nb,'003f4ec3',ns)
                selected=ns['sample_df']
                report=root/'reports/local_crawl_plan.csv'
                selected.to_csv(report,index=False,encoding='utf-8-sig')
                print(selected.groupby('domain').size().sort_values(ascending=False).to_string())
                print(f'Queued: {len(selected):,}; next shard >= '+str(ns['ShardWriter'](root/'crawl/html').shard_idx))
                print(f'Plan: {report}; source DB unchanged; no requests sent.')
                ns['con'].close()
                ns.pop('con',None)
            return 0
        if args.command=='extract':
            extract(args)
            return 0
        if args.command=='pubmed':
            prepare(root)
            os.environ['R2AI_DATA_ROOT'] = str(root)
            nb = notebook('01_data_pipeline')
            ns = {'__name__':'_r2ai_local','COLAB':False,'display':print}
            for ident in ('ee948b86','03180897'):
                execute(nb,ident,ns)
            ns['df_links'] = ns['pd'].read_parquet(root/'raw/links_corpus_vi_en_zh.parquet')
            execute(nb,'d9e60b0d',ns)
            return 0
        if args.command=='status':
            os.environ['R2AI_DATA_ROOT'] = str(root)
            nb=notebook('01_data_pipeline')
            ns={'__name__':'_r2ai_local','COLAB':False,'display':print}
            execute(nb,'ee948b86',ns)
            cell=next(c for c in nb['cells'] if c.get('id')=='crawl-domain-status')
            tree=ast.parse(''.join(cell['source']))
            definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions,type_ignores=[]),'local_status','exec'),ns)
            ns['snapshot_crawl_domains'](database)
            return 0
        nb,ns = crawler_context(args)
        if args.command=='crawl':
            if args.workers<=0:
                raise ValueError('--workers must be positive')
            stop = root/'crawl/STOP'
            if stop.exists():
                raise RuntimeError(f'STOP is present. Remove {stop} before intentionally starting another crawl.')
            execute(nb,'crawl-review-24',ns)
            execute(nb,'iiyi-daily-probe',ns)
            execute(nb,'003f4ec3',ns)
            ns['N_WORKERS'] = args.workers
            execute(nb,'b05d5ae1',ns)
            execute(nb,'crawl-domain-status',ns)
        elif args.command=='merge-home':
            execute(nb,'home-merge',ns)
            ns['merge_home_crawl'](args.home_dir)
        return 0
    finally:
        if ns and 'con' in ns:
            ns['con'].close()
        if awake:
            awake(0x80000000)


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed checkpoints/cache batches remain available.',file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
