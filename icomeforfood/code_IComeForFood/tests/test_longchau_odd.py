import base64
from datetime import timedelta
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import duckdb
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('longchau_odd',ROOT/'crawl/longchau_odd.py')
lc=importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


class LongChauOdd(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def args(self,probe=False):
        # Tests use zero pacing delay; CLI enforces >=30 seconds for real requests.
        return SimpleNamespace(delay=0,probe=probe,batch_size=50)

    def rows(self):
        return [dict(id=1,url='https://nhathuoclongchau.com.vn/a'),
                dict(id=3,url='https://nhathuoclongchau.com.vn/b')]

    def test_review_subset_preserves_priority_and_rejects_url_mismatch(self):
        corpus = self.root/'review.csv'
        corpus.write_text('id,url\n3,https://nhathuoclongchau.com.vn/b\n'
                          '1,https://nhathuoclongchau.com.vn/a\n',encoding='utf-8')
        self.assertEqual([row['id'] for row in lc.select_corpus_rows(self.rows(),corpus)],[3,1])
        corpus.write_text('id,url\n3,https://nhathuoclongchau.com.vn/other\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'differs from status DB'):
            lc.select_corpus_rows(self.rows(),corpus)

    def test_source_only_odd_longchau_and_known_blocks_are_preserved(self):
        corpus=self.root/'corpus.parquet'
        pq.write_table(pa.Table.from_pylist(self.rows()+[
            dict(id=2,url='https://nhathuoclongchau.com.vn/even'),
            dict(id=7,url='https://nhathuoclongchau.com.vn/not-in-db'),
            dict(id=5,url='https://tiemchunglongchau.com.vn/other')]),corpus)
        database=self.root/'status.duckdb'
        con=duckdb.connect(str(database))
        con.execute('CREATE TABLE crawl_status(id BIGINT,url VARCHAR,status VARCHAR,http_code INTEGER,fetched_at TIMESTAMP)')
        con.execute("INSERT INTO crawl_status VALUES (1,'https://nhathuoclongchau.com.vn/a','blocked',403,'2026-10-20')")
        con.execute("INSERT INTO crawl_status VALUES (3,'https://nhathuoclongchau.com.vn/from-db','pending',NULL,NULL)")
        con.close()
        before=database.read_bytes()
        rows,statuses=lc.source_rows(corpus,database)
        self.assertEqual([r['id'] for r in rows],[1,3])
        self.assertEqual(rows[1]['url'],'https://nhathuoclongchau.com.vn/from-db')
        self.assertEqual([r['id'] for r in lc.queue_rows(rows,statuses,{},set())],[3])
        self.assertEqual(database.read_bytes(),before)
        self.assertEqual(statuses[1]['status'],'blocked')

    def test_403_429_and_challenge_stop_after_one_request(self):
        for code,body in [(403,b'Forbidden'),(429,b'Too many requests'),(200,b'<html>Just a moment</html>')]:
            with self.subTest(code=code,body=body):
                out=self.root/str(code)/str(len(body))
                calls=[]
                def handler(request):
                    calls.append(str(request.url))
                    return httpx.Response(code,headers={'content-type':'text/html'},content=body)
                result=lc.collect(self.rows(),out,dict(kind='probe_ok'),self.args(),httpx.MockTransport(handler))
                self.assertEqual(len(calls),1)
                self.assertEqual(result['ok'],0)
                self.assertEqual(result['cooldown']['kind'],'blocked')
                self.assertFalse(list((out/'html').glob('*.jsonl.zst')))
                state=json.loads((out/'cooldown.json').read_text())
                self.assertIsNotNone(lc.gate(state,True))
                self.assertEqual(lc.last_status(out/'status.csv')[1]['status'],'blocked')

    def test_raw_bytes_roundtrip_has_exact_six_fields_and_resumes(self):
        raw='<html><meta charset="utf-8"><article>Dữ liệu nguyên gốc</article></html>'.encode()
        transport=httpx.MockTransport(lambda request:httpx.Response(200,headers={'content-type':'text/html; charset=utf-8'},content=raw))
        result=lc.collect(self.rows(),self.root,dict(kind='unprobed'),self.args(probe=True),transport)
        self.assertEqual((result['ok'],result['requests']),(1,1))
        path=next((self.root/'html').glob('home_shard_*.jsonl.zst'))
        with path.open('rb') as compressed,zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            record=json.loads(reader.read())
        self.assertEqual(set(record),{'id','final_url','http_code','content_type','encoding','body_b64'})
        self.assertEqual(base64.b64decode(record['body_b64']),raw)
        self.assertEqual(lc.raw_ids(self.root),{1})
        local=lc.last_status(self.root/'status.csv')
        self.assertEqual([r['id'] for r in lc.queue_rows(self.rows(),{3:dict(status='pending')},local,{1})],[3])

    def test_even_ids_are_rejected_before_network(self):
        calls=[]
        transport=httpx.MockTransport(lambda request:calls.append(request))
        with self.assertRaises(ValueError):
            lc.collect([dict(id=2,url=self.rows()[0]['url'])],self.root,dict(kind='probe_ok'),self.args(),transport)
        self.assertEqual(calls,[])

    def test_probe_never_follows_redirect_or_sends_a_second_request(self):
        calls=[]
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(302,headers={'location':self.rows()[1]['url']})
        result=lc.collect(self.rows(),self.root,dict(kind='unprobed'),self.args(probe=True),httpx.MockTransport(handler))
        self.assertEqual(len(calls),1)
        self.assertEqual(result['ok'],0)
        self.assertIsNotNone(lc.gate(result['cooldown'],False))

    def test_cross_domain_redirect_is_not_requested(self):
        calls=[]
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(302,headers={'location':'https://other.invalid/article'})
        lc.collect(self.rows(),self.root,dict(kind='probe_ok'),self.args(),httpx.MockTransport(handler))
        self.assertEqual(calls,[self.rows()[0]['url']])

    def test_cooldown_survives_restart_and_requires_single_probe(self):
        current=lc.now()
        state=dict(kind='blocked',next_probe_at=(current+timedelta(hours=6)).isoformat())
        lc.atomic_json(self.root/'cooldown.json',state)
        restored=json.loads((self.root/'cooldown.json').read_text())
        self.assertIsNotNone(lc.gate(restored,True,current))
        self.assertIsNone(lc.gate(restored,True,current+timedelta(hours=7)))
        self.assertIsNotNone(lc.gate(restored,False,current+timedelta(hours=7)))

    def test_lock_prevents_two_simultaneous_jobs(self):
        with lc.exclusive_lock(self.root/'domain.lock'):
            with self.assertRaises(RuntimeError):
                with lc.exclusive_lock(self.root/'domain.lock'):
                    self.fail('second lock unexpectedly acquired')

    def test_scan_skips_separate_probe_but_not_active_cooldown(self):
        current=lc.now()
        self.assertIsNone(lc.gate(dict(kind='unprobed'),False,current,scan=True))
        active=dict(kind='blocked',next_probe_at=(current+timedelta(hours=6)).isoformat())
        self.assertIsNotNone(lc.gate(active,False,current,scan=True))
        self.assertIsNone(lc.gate(active,False,current+timedelta(hours=7),scan=True))

    def test_scan_continues_across_batches_with_rest_and_no_probe(self):
        from unittest.mock import patch
        rows=self.rows()+[dict(id=5,url='https://nhathuoclongchau.com.vn/c')]
        args=self.args()
        args.scan=True
        args.batch_size=2
        calls=[]
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(200,headers={'content-type':'text/html'},content=b'<html>Article</html>')
        with patch.object(lc,'wait_for_rest',return_value=True) as rest:
            result=lc.collect(rows,self.root,dict(kind='unprobed'),args,httpx.MockTransport(handler))
        self.assertEqual(len(calls),3)
        self.assertEqual(result['ok'],3)
        rest.assert_called_once_with(self.root/'STOP',10800)

    def test_scan_stops_at_first_403_and_leaves_other_urls_unattempted(self):
        args=self.args()
        args.scan=True
        calls=[]
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(403,content=b'Forbidden')
        result=lc.collect(self.rows(),self.root,dict(kind='unprobed'),args,httpx.MockTransport(handler))
        self.assertEqual(len(calls),1)
        self.assertEqual(result['cooldown']['kind'],'blocked')
        self.assertNotIn(3,lc.last_status(self.root/'status.csv'))

    def test_scan_can_stop_during_batch_rest(self):
        from unittest.mock import patch
        args=self.args()
        args.scan=True
        args.batch_size=1
        transport=httpx.MockTransport(lambda request:httpx.Response(200,headers={'content-type':'text/html'},content=b'<html>Article</html>'))
        with patch.object(lc,'wait_for_rest',return_value=False):
            result=lc.collect(self.rows(),self.root,dict(kind='unprobed'),args,transport)
        self.assertEqual(result['requests'],1)
        self.assertEqual(lc.raw_ids(self.root),{1})



if __name__=='__main__':
    unittest.main()
