import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import duckdb
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crawl'))
from finalize_longchau import promote
from scripts.run_local import main as run_local_main


class FinalizeLongChau(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'longchau_odd_raw'
        (self.source / 'html').mkdir(parents=True)
        self.html = self.root / 'crawl' / 'html'
        self.html.mkdir(parents=True)
        self.db = self.root / 'status.duckdb'
        self.url = 'https://nhathuoclongchau.com.vn/bai-viet/example.html'
        self.body = b'<html><title>Example</title><article>Body</article></html>'
        con = duckdb.connect(str(self.db))
        con.execute('''CREATE TABLE crawl_status (
            id BIGINT PRIMARY KEY,url VARCHAR,final_url VARCHAR,lang_expected VARCHAR,
            status VARCHAR,http_code INTEGER,content_type VARCHAR,n_bytes BIGINT,
            attempts INTEGER,fetched_at TIMESTAMP,shard INTEGER,sha1_raw VARCHAR,error_msg VARCHAR)''')
        con.execute('INSERT INTO crawl_status(id,url,status) VALUES (1,?,\'pending\')', [self.url])
        con.close()
        self.write_stage()

    def write_stage(self, *, sha1=None):
        sha1 = sha1 or hashlib.sha1(self.body).hexdigest()
        name = 'tv_shard_00000.jsonl.zst'
        row = dict(id='1', url=self.url, final_url=self.url, lang_expected='vi',
                   status='ok', http_code='200', content_type='text/html; charset=utf-8',
                   n_bytes=str(len(self.body)), attempts='1', fetched_at='2026-10-21T00:00:00+00:00',
                   shard='', sha1_raw=sha1, error_msg='', shard_file=name)
        with (self.source / 'status.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        record = dict(id=1, final_url=self.url, http_code=200,
                      content_type=row['content_type'], encoding='utf-8',
                      body_b64=base64.b64encode(self.body).decode('ascii'))
        (self.source / 'html' / name).write_bytes(
            zstd.ZstdCompressor().compress((json.dumps(record) + '\n').encode()))

    def test_dry_run_apply_and_idempotence(self):
        self.assertEqual(promote([self.source], self.db, self.html), 1)
        self.assertFalse(list(self.html.glob('shard_*.jsonl.zst')))
        self.assertEqual(promote([self.source], self.db, self.html, apply=True), 1)
        shard = self.html / 'shard_00000.jsonl.zst'
        with shard.open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as reader:
            record = json.loads(io.TextIOWrapper(reader, encoding='utf-8').readline())
        self.assertEqual(set(record), {'id', 'final_url', 'http_code', 'content_type', 'encoding', 'body_b64'})
        self.assertEqual(base64.b64decode(record['body_b64']), self.body)
        con = duckdb.connect(str(self.db), read_only=True)
        self.assertEqual(con.execute('SELECT status,shard,n_bytes,sha1_raw FROM crawl_status WHERE id=1').fetchone(),
                         ('ok', 0, len(self.body), hashlib.sha1(self.body).hexdigest()))
        con.close()
        self.assertEqual(promote([self.source], self.db, self.html, apply=True), 0)
        self.assertEqual(len(list(self.html.glob('shard_*.jsonl.zst'))), 1)

    def test_hash_mismatch_does_not_publish(self):
        self.write_stage(sha1='bad')
        with self.assertRaisesRegex(ValueError, 'body/status mismatch'):
            promote([self.source], self.db, self.html, apply=True)
        self.assertFalse(list(self.html.glob('shard_*.jsonl.zst')))
        con = duckdb.connect(str(self.db), read_only=True)
        self.assertEqual(con.execute('SELECT status FROM crawl_status WHERE id=1').fetchone()[0], 'pending')
        con.close()

    def test_failed_url_updates_status_without_creating_shard(self):
        con = duckdb.connect(str(self.db))
        second = 'https://nhathuoclongchau.com.vn/bai-viet/slow.html'
        con.execute("INSERT INTO crawl_status(id,url,status) VALUES (3,?,'pending')", [second])
        con.close()
        with (self.source / 'status.csv').open('a', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([3, second, second, 'vi', 'http_error', '', '', 0, 1,
                             '2026-10-21T00:01:00+00:00', '', '', 'Page.goto: Timeout 30000ms exceeded', ''])
        self.assertEqual(promote([self.source], self.db, self.html, apply=True), 1)
        con = duckdb.connect(str(self.db), read_only=True)
        self.assertEqual(con.execute('SELECT status,shard FROM crawl_status WHERE id=3').fetchone(),
                         ('timeout', None))
        con.close()

    def test_first_block_without_saved_shards_can_be_finalized(self):
        (self.source / 'html' / 'tv_shard_00000.jsonl.zst').unlink()
        with (self.source / 'status.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'url', 'final_url', 'lang_expected', 'status',
                             'http_code', 'content_type', 'n_bytes', 'attempts',
                             'fetched_at', 'shard', 'sha1_raw', 'error_msg', 'shard_file'])
            writer.writerow([1, self.url, self.url, 'vi', 'blocked', 403,
                             'text/html', 100, 1, '2026-10-22T13:14:28+00:00',
                             '', '', 'HTTP 403: stopped at first block', ''])
        self.assertEqual(promote([self.source], self.db, self.html, apply=True), 0)
        self.assertFalse(list(self.html.glob('shard_*.jsonl.zst')))
        con = duckdb.connect(str(self.db), read_only=True)
        try:
            self.assertEqual(con.execute('SELECT status,http_code,shard FROM crawl_status WHERE id=1').fetchone(),
                             ('blocked', 403, None))
        finally:
            con.close()

    def test_local_runner_exports_after_raw_crawler_exits(self):
        with mock.patch('subprocess.call', side_effect=[0, 0]) as calls:
            result = run_local_main(['longchau-odd', '--engine', 'crawl4ai',
                                     '--raw-http', '--main-output', '--status-db', str(self.db),
                                     '--data-root', str(self.root), '--corpus', str(self.root / 'ranked.csv'),
                                     '--scan', '--continuous', '--full-scan', '--delay', '5'])
        self.assertEqual(result, 0)
        self.assertEqual(calls.call_count, 2)
        crawl_command = calls.call_args_list[0].args[0]
        export_command = calls.call_args_list[1].args[0]
        self.assertIn('--raw-http', crawl_command)
        self.assertIn('--apply', export_command)
        self.assertIn(str(self.db), export_command)


if __name__ == '__main__':
    unittest.main()
