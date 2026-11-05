"""Offline regression checks for the 20 October crawler/full-shard notebook updates.

Run: python -m unittest discover -s tests -p test_notebook_review_updates.py
No production database, Drive data, or HTTP requests are used.
"""
import asyncio
import ast
import base64
import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import time
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch
from urllib.parse import urlparse

import duckdb
import httpx
import nbformat
import pandas as pd
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / 'notebooks' / (name + '.ipynb')).read_text(encoding='utf-8'))


def source(nb, ident):
    return ''.join(next(c for c in nb['cells'] if c.get('id') == ident)['source'])


class Progress:
    def __init__(self, **kw):
        pass
    def update(self, n):
        pass
    def close(self):
        pass


class NotebookUpdates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.ns = dict(globals(), __name__='_notebook_test', DATA_ROOT=root, CRAWL_DIR=root/'crawl',
                       HTML_DIR=root/'crawl'/'html', REPORT_DIR=root/'reports', DOCS_DIR=root/'docs',
                       CONFIG_ROOT=ROOT/'crawl', RUN_ID='offline',
                       logger=logging.getLogger('notebook_test'), flush_log_handlers=lambda **kw: None,
                       display=lambda value: None, tqdm=Progress)
        for key in ('CRAWL_DIR','HTML_DIR','REPORT_DIR','DOCS_DIR'):
            self.ns[key].mkdir(parents=True)
        self.ns['COLAB'] = False
        environment = patch.dict(os.environ, {'R2AI_STATUS_DB': str(root/'crawl/status.duckdb')})
        environment.start()
        self.addCleanup(environment.stop)
        self.crawler = notebook('01_data_pipeline')
        self.gold = notebook('03_gold_review_and_sample2')
        guard_tree=ast.parse(source(self.crawler,'7c6a152d'))
        guard=next(node for node in guard_tree.body if isinstance(node,ast.AsyncFunctionDef) and node.name=='guard_regular_crawler_request')
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'regular_request_guard.py','exec'),self.ns)


    def execute(self, nb, ident):
        exec(compile(source(nb, ident), f'{ident}.py', 'exec'), self.ns)

    def database(self):
        self.execute(self.crawler, '6048f8cf')
        con = self.ns['con']
        self.addCleanup(con.close)
        return con

    def seed(self, con, rows):
        con.executemany('INSERT INTO crawl_status(id,url,status,attempts,http_code,sha1_raw) VALUES (?,?,?,0,?,?)', rows)

    def raw(self, rid, body, final_url=None, code=200):
        return dict(id=rid, final_url=final_url or f'https://example.org/{rid}', http_code=code,
                    content_type='text/html', encoding='utf-8', body_b64=base64.b64encode(body).decode())

    def shard(self, idx, records, home=False):
        directory = self.ns['CRAWL_DIR']/'html_home' if home else self.ns['HTML_DIR']
        directory.mkdir(exist_ok=True)
        path = directory / (f'home_shard_{idx:05d}.jsonl.zst' if home else f'shard_{idx:05d}.jsonl.zst')
        path.write_bytes(zstd.ZstdCompressor().compress(('\n'.join(map(json.dumps, records))+'\n').encode()))
        return path

    def cleaner(self):
        self.execute(self.gold, 'gold-clean-helpers')
        self.execute(self.gold, 'gold-extractors')

    def test_notebook_structure_and_python_syntax(self):
        for name in ('01_data_pipeline','03_gold_review_and_sample2'):
            path = ROOT/'notebooks'/f'{name}.ipynb'
            nbformat.validate(nbformat.read(path, as_version=4))
            for c in notebook(name)['cells']:
                if c['cell_type']=='code':
                    ast.parse(''.join(c['source']))

    def test_existing_cleaner_regressions_unchanged(self):
        self.cleaner()
        self.execute(self.gold, 'gold-cleaner-selftest')

    def test_full_selection_mismatch_coverage_duplicates_and_offsets(self):
        con = self.database()
        self.cleaner()
        self.execute(self.gold, 'shard-runner')
        self.execute(self.gold, 'review-v112-cell-25')
        gate, parity = Mock(return_value=[]), Mock(return_value=pd.DataFrame())
        self.ns.update(require_review_gold_pass=gate, validate_shard_gold_parity=parity,
                       GOLD_NOTEBOOK_BUILD='2026-10-20-v26-shard-html-selection-postfilter-v1.12')
        paragraph = 'This is the complete medical article describing symptoms and treatment. ' * 12
        body = f'<html><title>Medical article</title><article><h1>Medical article</h1><p>{paragraph}</p></article></html>'.encode()
        old = b'<html><article><h1>Old article</h1><p>Old text</p></article></html>'
        self.seed(con, [(1,'https://example.org/1','blocked',403,hashlib.sha1(old).hexdigest()),
                        (2,'https://example.org/2','ok',200,hashlib.sha1(body).hexdigest()),
                        (3,'https://care.39.net/3','blocked',403,''),
                        (4,'https://example.org/4','pending',None,''),
                        (5,'https://example.org/5','ok',200,''),
                        (6,'https://example.org/6','ok',200,'')])
        # Windows locks open DuckDB files; stop the writer before taking the snapshot.
        a = self.shard(0,[self.raw(1,old),self.raw(2,body),self.raw(5,body,code=404)])
        b = self.shard(1,[self.raw(1,body),self.raw(3,b'<html>verify</html>',final_url='https://care.39.net/verify.html'),self.raw(6,b'')])
        con.close()
        result = self.ns['run_shard_extraction']([a,b],nt_approved=True).iloc[0]
        gate.assert_called_once()
        parity.assert_called_once()
        self.assertEqual(result['coverage_rows'],6)
        self.assertEqual(result['extracted_ids'],2)
        self.assertEqual(result['missing_ids'],4)
        self.assertEqual(result['bad_offsets'],0)
        frames = [pd.read_parquet(p) for p in (self.ns['DOCS_DIR']/'full_offline').glob('docs_*.parquet')]
        frame = pd.concat(frames).set_index('id')
        self.assertEqual(len(frame),6)
        self.assertEqual(frame.loc[1,'text'],frame.loc[2,'text'])
        self.assertIn('db_status_mismatch',frame.loc[1,'quality_flags'])
        self.assertTrue(frame.loc[1,'selected_source'].startswith('shard_00001'))
        self.assertEqual(frame.loc[3,'text'],'')
        report = self.ns['REPORT_DIR']
        for suffix in ('extraction_summary','missing','db_status_mismatch','domain_quality','duplicate_text_groups','offset_errors'):
            self.assertTrue((report/f'full_{suffix}.csv').is_file())
        duplicates = pd.read_csv(report/'full_duplicate_text_groups.csv')
        self.assertEqual(duplicates['documents'].tolist(),[2])
        # A rerun cannot silently overwrite the already-completed run.
        with self.assertRaises(FileExistsError):
            self.ns['run_shard_extraction']([a,b],nt_approved=True)

    def test_full_refuses_failed_gold_before_reading_data(self):
        self.execute(self.gold,'shard-runner')
        self.ns['require_review_gold_pass'] = Mock(side_effect=RuntimeError('gold mismatch'))
        with self.assertRaisesRegex(RuntimeError,'gold mismatch'):
            self.ns['run_shard_extraction'](nt_approved=True)

    def test_language_repair_and_full_queue_excludes_home_and_pharmacity(self):
        con = self.database()
        rows = [(i,f'https://{dom}/article','pending',None,'') for i,dom in enumerate([
            'care.39.net','www.39.net','nhathuoclongchau.com.vn','tamanhhospital.vn',
            'pmc-ecm-healthblog.beta.pharmacity.io','pubmed.ncbi.nlm.nih.gov','zysjonline.com','120ask.com','thanhnien.vn'],1)]
        self.seed(con,rows)
        con.execute("UPDATE crawl_status SET status='http_error',http_code=NULL,attempts=1,error_msg='RemoteProtocolError: StreamReset' WHERE id=9")
        self.ns['df_links'] = pd.DataFrame([{'id':r[0],'url':r[1]} for r in rows])
        self.execute(self.crawler,'d31dc248')
        self.execute(self.crawler,'0d3118f8')
        self.assertEqual(con.execute('SELECT DISTINCT lang_expected FROM crawl_status WHERE id IN (1,2)').fetchall(),[('zh',)])
        self.ns['RETRY_DELAYS'] = [1,3,10]
        self.execute(self.crawler,'003f4ec3')
        self.assertEqual(self.ns['CRAWL_MODE'],'full')
        self.assertEqual(set(self.ns['sample_df']['id']),{1,2,9})

    def test_home_merge_last_valid_row_and_idempotence(self):
        con = self.database()
        url = 'https://nhathuoclongchau.com.vn/article'
        self.seed(con,[(1,url,'blocked',403,''),(2,'https://tamanhhospital.vn/article','pending',None,'')])
        body = b'<html><article>Valid home page</article></html>'
        self.shard(0,[self.raw(1,body,final_url=url)],home=True)
        home = self.ns['CRAWL_DIR']/'html_home'
        fields = ['id','url','status','http_code','final_url','content_type','n_bytes','attempts','fetched_at','sha1_raw','shard_file']
        with (home/'status.csv').open('w',newline='',encoding='utf-8') as handle:
            writer = csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerow(dict(id=1,url=url,status='blocked',http_code=403,fetched_at='2026-10-19T00:00:00Z'))
            writer.writerow(dict(id=1,url=url,status='ok',http_code=200,final_url=url,content_type='text/html',
                n_bytes=len(body),attempts=1,fetched_at='2026-10-20T00:00:00Z',sha1_raw=hashlib.sha1(body).hexdigest(),shard_file='home_shard_00000.jsonl.zst'))
            writer.writerow(dict(id=2,url='https://tamanhhospital.vn/article',status='http_error',http_code=404,fetched_at='2026-10-20T00:00:00Z'))
            handle.write('1,'+url+',\n')
        self.execute(self.crawler,'home-merge')
        self.ns['merge_home_crawl']()
        first = con.execute('SELECT id,status,http_code,shard FROM crawl_status ORDER BY id').fetchall()
        self.assertEqual(first,[(1,'ok',200,0),(2,'http_error',404,None)])
        self.ns['merge_home_crawl']()
        self.assertEqual(con.execute('SELECT id,status,http_code,shard FROM crawl_status ORDER BY id').fetchall(),first)
        self.assertEqual(len(list(self.ns['HTML_DIR'].glob('shard_*.jsonl.zst'))),1)
        # A later local success must survive importing this older home status again.
        con.execute("UPDATE crawl_status SET status='ok',fetched_at='2026-10-21' WHERE id=2")
        self.ns['merge_home_crawl']()
        self.assertEqual(con.execute('SELECT status FROM crawl_status WHERE id=2').fetchone()[0],'ok')

    def crawler_fixture(self):
        con = self.database()
        self.seed(con,[(1,'https://example.org/1','pending',None,'')])
        self.ns.update(domain_plan_dict={},tqdm=Progress)
        self.execute(self.crawler,'ea80b6a7')
        self.execute(self.crawler,'2a08247e')
        self.execute(self.crawler,'8e8120c6')
        body = b'<html>Good page</html>'
        async def fetch(client, manager, row, shard_idx):
            return dict(id=1,final_url=row['url'],status='ok',http_code=200,content_type='text/html',
                n_bytes=len(body),attempts=1,fetched_at='2026-10-20T00:00:00',shard=shard_idx,
                sha1_raw=hashlib.sha1(body).hexdigest(),error_msg=None),self.raw(1,body)
        self.ns['fetch_with_retry'] = fetch
        rows = con.execute('SELECT id,url,status,attempts FROM crawl_status').df()
        return con,rows

    def test_checkpoint_persists_raw_before_status_and_resume(self):
        con,rows = self.crawler_fixture()
        run = self.ns['run_crawler_pipeline']
        metrics = asyncio.run(run(rows,n_workers=1,save_interval_sec=1e-12))
        self.assertEqual((metrics['processed'],metrics['ok'],metrics['fail']),(1,1,0))
        self.assertEqual(con.execute('SELECT status,shard FROM crawl_status').fetchone(),('ok',0))
        self.assertTrue((self.ns['HTML_DIR']/'shard_00000.jsonl.zst').is_file())
        again = asyncio.run(run(rows,n_workers=1,save_interval_sec=1e-12))
        self.assertEqual(again['processed'],0)

    def test_checkpoint_failure_does_not_count_url_twice(self):
        con,rows = self.crawler_fixture()
        class FailOnce:
            failed = False
            def __getattr__(self,name):
                return getattr(con,name)
            def execute(self,sql,*args):
                if sql=='CHECKPOINT' and not self.failed:
                    self.failed = True
                    raise OSError('simulated checkpoint failure')
                return con.execute(sql,*args)
        self.ns['con'] = FailOnce()
        with self.assertRaisesRegex(OSError,'checkpoint failure'):
            asyncio.run(self.ns['run_crawler_pipeline'](rows,n_workers=1,save_interval_sec=1e-12))
        metrics = json.loads((self.ns['REPORT_DIR']/'crawl_metrics_latest.json').read_text(encoding='utf-8'))
        self.assertEqual((metrics['processed'],metrics['ok'],metrics['fail']),(1,1,0))

    def test_iiyi_probe_runs_once_per_day_and_requeues_only_on_recovery(self):
        con = self.database()
        self.seed(con,[(1,'https://article.iiyi.com/1','blocked',521,''),
                       (2,'https://article.iiyi.com/2','blocked',403,'')])
        self.execute(self.crawler,'crawl-review-23')
        calls=[]
        verdict=['http_error']
        async def probe(domains,delay_sec):
            domain=domains[0]
            calls.append(domain)
            con.execute("INSERT OR REPLACE INTO domain_gate(domain,checked_at,verdict,http_code) VALUES (?,current_timestamp,?,?)",
                        [domain,verdict[0],200 if verdict[0]=='ok' else 521])
            return [dict(domain=domain,verdict=verdict[0])]
        self.ns['run_domain_gate']=probe
        self.execute(self.crawler,'iiyi-daily-probe')
        asyncio.run(self.ns['retry_iiyi_daily']())
        self.assertEqual(calls,['article.iiyi.com'])
        self.assertEqual(con.execute('SELECT status FROM crawl_status WHERE id=1').fetchone()[0],'blocked')
        con.execute("UPDATE domain_gate SET checked_at=current_timestamp-INTERVAL '2 days'")
        verdict[0]='ok'
        asyncio.run(self.ns['retry_iiyi_daily']())
        self.assertEqual(len(calls),2)
        self.assertEqual(con.execute('SELECT status,attempts,http_code FROM crawl_status WHERE id=1').fetchone(),('pending',0,None))
        self.assertEqual(con.execute('SELECT status,http_code FROM crawl_status WHERE id=2').fetchone(),('blocked',403))

    def test_vov_headless_stops_after_first_blocked_page(self):
        from types import ModuleType,SimpleNamespace
        from unittest.mock import AsyncMock,patch
        con=self.database()
        self.seed(con,[(1,'https://vov.vn/1','blocked',403,''),(2,'https://vov.vn/2','blocked',403,'')])
        self.execute(self.crawler,'2a08247e')
        self.ns.update(STOP_FILE=self.ns['CRAWL_DIR']/'STOP',GATE_MARKERS=('captcha',))
        self.execute(self.crawler,'vov-browser-probe')
        response=SimpleNamespace(status=403,headers={'content-type':'text/html'},body=AsyncMock(return_value=b'<html>Forbidden</html>'))
        page=SimpleNamespace(url='https://vov.vn/1',goto=AsyncMock(return_value=response))
        browser=SimpleNamespace(new_page=AsyncMock(return_value=page),close=AsyncMock())
        playwright=SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(return_value=browser)))
        class Session:
            async def __aenter__(self):return playwright
            async def __aexit__(self,*args):return False
        api=ModuleType('playwright.async_api')
        api.async_playwright=Session
        with patch.dict(sys.modules,{'playwright':ModuleType('playwright'),'playwright.async_api':api}):
            result=asyncio.run(self.ns['crawl_vov_headless'](max_urls=None))
        self.assertEqual(len(result),1)
        page.goto.assert_awaited_once()
        browser.close.assert_awaited_once()
        self.assertEqual(con.execute('SELECT attempts FROM crawl_status WHERE id=2').fetchone()[0],0)
        self.assertFalse(list(self.ns['HTML_DIR'].glob('shard_*.jsonl.zst')))
        self.assertTrue((self.ns['REPORT_DIR']/'vov_browser_probe.csv').is_file())

    def test_pubmed_reuses_legacy_batch_cache_without_requests(self):
        from unittest.mock import patch
        tree=ast.parse(source(self.crawler,'d9e60b0d'))
        helper=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fetch_pubmed_batches')
        self.ns['PUBMED_DIR']=self.ns['CRAWL_DIR']/'pubmed'
        self.ns['PUBMED_DIR'].mkdir()
        self.ns['tqdm']=lambda values,**kw:values
        self.ns['EUROPE_PMC_POST_URL']='https://example.invalid'
        cached=[{'pmid':'123','title':'Already cached'},{'pmid':'999','title':'Not requested'}]
        (self.ns['PUBMED_DIR']/'batch_legacy.json.zst').write_bytes(zstd.ZstdCompressor().compress(json.dumps(cached).encode()))
        exec(compile(ast.Module(body=[helper],type_ignores=[]),'pubmed_helper.py','exec'),self.ns)
        with patch.object(httpx,'Client') as client:
            hits=self.ns['fetch_pubmed_batches'](['123'])
            client.return_value.__enter__.return_value.post.assert_not_called()
        self.assertEqual([hit['pmid'] for hit in hits],['123'])

    def test_gold_gate_rejects_old_run_directories(self):
        self.execute(self.gold,'review-v112-cell-25')
        old=self.ns['DATA_ROOT']/'reports_previous'
        old.mkdir()
        self.ns['_gold_run_summaries']=[dict(gold_set=name,output_root=str(old/name)) for name in ('gold1','gold2','gold3')]
        with self.assertRaisesRegex(RuntimeError,'REPORT_DIR'):
            self.ns['require_review_gold_pass']()

    @unittest.skipUnless(os.name=='nt','Windows file-lock behavior')
    def test_snapshot_explains_windows_writer_lock(self):
        self.database()
        self.execute(self.gold,'review-v112-cell-25')
        with self.assertRaisesRegex(RuntimeError,'Windows'):
            self.ns['_load_latest_status']()


    def test_local_shard_numbers_continue_after_db_without_old_html(self):
        con=self.database()
        self.seed(con,[(1,'https://example.org/1','ok',200,'')])
        con.execute('UPDATE crawl_status SET shard=77 WHERE id=1')
        self.execute(self.crawler,'2a08247e')
        writer=self.ns['ShardWriter'](self.ns['HTML_DIR'])
        self.assertEqual(writer.shard_idx,78)
        writer.add(self.raw(2,b'<html>New local page</html>'))
        self.assertEqual(writer.flush().name,'shard_00078.jsonl.zst')



if __name__=='__main__':
    unittest.main()
