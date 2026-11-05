"""Offline checks for the 22 October Long Chau TV crawl contract."""
from collections import deque
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import sys

import httpx
import duckdb

from crawl import crawl_home


ROOT = Path(__file__).resolve().parents[1]


class CrawlHomeTVTest(unittest.TestCase):
    def test_review_corpus_contains_only_5890_unique_odd_longchau_ids(self):
        path = ROOT / 'crawl/longchau_tv_crawl_odd.csv'
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        ids = [int(row['id']) for row in rows]
        self.assertEqual(len(ids), 5890)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(rid % 2 == 1 for rid in ids))
        self.assertTrue(all(crawl_home.urlparse(row['url']).hostname ==
                            'nhathuoclongchau.com.vn' for row in rows))

    def test_first_challenge_stops_without_requesting_next_url(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            out = Path(folder)
            urls = ['https://nhathuoclongchau.com.vn/a',
                    'https://nhathuoclongchau.com.vn/b']
            requested = []

            def response(request):
                requested.append(str(request.url))
                return httpx.Response(200, headers={'content-type': 'text/html'},
                                      content=b'<html>Just a moment... challenge-platform</html>')

            domain = crawl_home.Domain('nhathuoclongchau.com.vn', 0.25,
                                       ramp=(0.05, 0.025, 1200))
            queue = deque({'id': rid, 'url': url}
                          for rid, url in zip((1, 3), urls))
            output = crawl_home.Output(out, 'tv')
            try:
                with patch.object(crawl_home, 'PAUSES', []), \
                     httpx.Client(transport=httpx.MockTransport(response)) as client:
                    crawl_home.worker(domain, queue, client, output, threading.Event())
            finally:
                output.close()
            self.assertEqual(requested, urls[:1])
            self.assertTrue(domain.stopped)
            self.assertEqual(len(queue), 1)
            with (out / 'status.csv').open(newline='', encoding='utf-8') as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual([(row['id'], row['status']) for row in saved], [('1', 'blocked')])
            self.assertFalse(list((out / 'html').glob('*.jsonl.zst')))

    def test_status_skip_list_does_not_discard_failed_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / 'status.csv'
            path.write_text('id,status,shard_file\n1,ok,tv_shard_00000.jsonl.zst\n'
                            '3,blocked,\n5,http_error,\n7,non_html,\n9,ok,\n', encoding='utf-8')
            self.assertEqual(crawl_home.read_ids(str(path)), {1})

    def test_offline_plan_does_not_create_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            corpus = root / 'subset.csv'
            corpus.write_text('id,url\n1,https://nhathuoclongchau.com.vn/a\n', encoding='utf-8')
            out = root / 'out'
            argv = ['crawl_home.py', '--corpus', str(corpus), '--out', str(out),
                    '--rate', 'nhathuoclongchau.com.vn=0.25', '--plan']
            printed = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(printed):
                crawl_home.main()
            self.assertIn('1 URLs remaining', printed.getvalue())
            self.assertFalse(out.exists())

    def test_shared_db_snapshot_skips_saved_ids(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            corpus = root / 'subset.csv'
            corpus.write_text('id,url\n1,https://nhathuoclongchau.com.vn/a\n'
                              '3,https://nhathuoclongchau.com.vn/b\n'
                              '5,https://nhathuoclongchau.com.vn/c\n', encoding='utf-8')
            database = root / 'status.duckdb'
            con = duckdb.connect(str(database))
            try:
                con.execute('CREATE TABLE crawl_status (id INTEGER, status VARCHAR, shard INTEGER)')
                con.executemany('INSERT INTO crawl_status VALUES (?,?,?)',
                                [(1, 'ok', 42), (3, 'pending', None), (5, 'ok', None)])
            finally:
                con.close()
            queues = crawl_home.load_queue(str(corpus), ['nhathuoclongchau.com.vn'],
                                           str(database), root / 'out', [], None,
                                           ['nhathuoclongchau.com.vn'])
            self.assertEqual([row['id'] for row in queues['nhathuoclongchau.com.vn']],
                             [3, 5])


if __name__ == '__main__':
    unittest.main()
