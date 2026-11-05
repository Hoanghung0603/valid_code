"""Offline checks of retry queue, raw bytes, and main-output promotion."""
import asyncio
import base64
import csv
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb
import zstandard as zstd

from crawl import retry_crawl4ai as retry


class FakeResponse:
    def __init__(self, url, body, code=200, content_type='text/html; charset=iso-8859-1'):
        self.url, self._body, self.status = url, body, code
        self._headers = {'content-type': content_type}

    async def body(self):
        return self._body

    async def all_headers(self):
        return self._headers


class FakeCrawler:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.crawler_strategy = self
        self.urls = []

    def set_hook(self, name, hook):
        assert name == 'after_goto'
        self.hook = hook

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def arun(self, url, config):
        self.urls.append(url)
        response = next(self.replies)
        await self.hook(None, response=response)
        return SimpleNamespace(error_message=None)


class RetryCrawl4AITest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db = self.root / 'status.duckdb'
        self.out = self.root / 'retry'
        self.html = self.root / 'html'
        self.html.mkdir()
        con = duckdb.connect(str(self.db))
        con.execute('''CREATE TABLE crawl_status (
            id BIGINT PRIMARY KEY,url VARCHAR,final_url VARCHAR,lang_expected VARCHAR,
            status VARCHAR,http_code INTEGER,content_type VARCHAR,n_bytes BIGINT,
            attempts INTEGER,fetched_at TIMESTAMP,shard INTEGER,sha1_raw VARCHAR,error_msg VARCHAR)''')
        con.executemany('INSERT INTO crawl_status (id,url,status,attempts) VALUES (?,?,?,0)', [
            (1, 'https://example.org/a', 'pending'),
            (2, 'https://nhathuoclongchau.com.vn/a', 'pending'),
            (3, 'https://zysjonline.com/a', 'pending'),
            (4, 'https://example.org/b', 'blocked'),
            (5, 'https://example.org/done', 'ok'),
            (6, 'https://pubmed.ncbi.nlm.nih.gov/123/', 'pending'),
            (7, 'https://tiemchunglongchau.com.vn/a', 'pending'),
            (8, 'https://vov.vn/a', 'pending'),
            (9, 'https://en.vov.vn/a', 'blocked'),
            (10, 'https://news.pubmed.ncbi.nlm.nih.gov/a', 'pending'),
            (11, 'https://notvov.vn/a', 'pending'),
        ])
        con.close()

    def test_exact_queue_excludes_requested_domains_and_subdomains(self):
        with patch.object(retry, 'snapshot_rows', wraps=retry.snapshot_rows):
            rows, selected = retry.pending_rows(self.db, self.out)
        self.assertEqual([row['id'] for row in rows], [1, 4, 11])
        self.assertEqual([row['id'] for row in selected], [1, 4, 11])

    def test_repeated_404_is_skipped_unless_explicitly_requested(self):
        self.out.mkdir()
        status_path = self.out / 'status.csv'
        with status_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=retry.STATUS_COLUMNS)
            writer.writeheader()
            writer.writerow(dict(id=1, url='https://example.org/a',
                                 status='http_error', http_code=404, attempts=1))
        _, selected = retry.pending_rows(self.db, self.out, retry_failed=True)
        self.assertIn(1, [row['id'] for row in selected])
        with status_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=retry.STATUS_COLUMNS)
            writer.writerow(dict(id=1, url='https://example.org/a',
                                 status='http_error', http_code=404, attempts=2))
        _, selected = retry.pending_rows(self.db, self.out, retry_failed=True)
        self.assertNotIn(1, [row['id'] for row in selected])
        _, selected = retry.pending_rows(self.db, self.out, retry_failed=True,
                                         retry_404=True)
        self.assertIn(1, [row['id'] for row in selected])

    def test_404_challenge_words_do_not_quarantine_host(self):
        item = dict(id=1, url='https://example.org/a', lang_expected='en', attempts=0)
        captured = dict(body=b'<html>access denied</html>', status=404,
                        headers={'content-type': 'text/html'}, url=item['url'])
        record, row = retry.classify(None, captured, item)
        self.assertIsNone(record)
        self.assertEqual(row['status'], 'http_error')
        now = datetime.now(timezone.utc)
        previous = {1: dict(status='blocked', url=item['url'], http_code='404',
                            fetched_at=now.isoformat()),
                    2: dict(status='blocked', url='https://blocked.example/a',
                            http_code='403', fetched_at=now.isoformat())}
        self.assertNotIn('example.org', retry.active_cooldowns(previous, now))
        self.assertIn('blocked.example', retry.active_cooldowns(previous, now))

    def test_article_with_recaptcha_script_is_not_a_challenge_page(self):
        item = dict(id=1, url='https://tamanhhospital.vn/article/',
                    lang_expected='vi', attempts=0)
        body = (b'<html><head><title>Article title</title></head><body>'
                b'<article>Medical article content</article>'
                b'<script>if (result.indexOf("recaptcha") !== -1) { alert("retry") }</script>'
                b'</body></html>')
        captured = dict(body=body, status=200,
                        headers={'content-type': 'text/html'}, url=item['url'])
        record, row = retry.classify(None, captured, item)
        self.assertIsNotNone(record)
        self.assertEqual(row['status'], 'ok')

    def test_captcha_title_is_still_blocked(self):
        item = dict(id=1, url='https://example.org/article/',
                    lang_expected='en', attempts=0)
        captured = dict(body=b'<html><title>Captcha verification</title></html>',
                        status=200, headers={'content-type': 'text/html'},
                        url=item['url'])
        record, row = retry.classify(None, captured, item)
        self.assertIsNone(record)
        self.assertEqual(row['status'], 'blocked')

    def test_cooldown_explains_zero_requests_without_starting_browser(self):
        self.out.mkdir()
        with (self.out / 'status.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=retry.STATUS_COLUMNS)
            writer.writeheader()
            writer.writerow(dict(id=4, url='https://example.org/b', status='blocked',
                                 http_code=403, fetched_at=datetime.now(timezone.utc).isoformat()))
        row = dict(id=1, url='https://example.org/a', lang_expected='en', attempts=0)
        output = io.StringIO()
        with redirect_stdout(output):
            counts = asyncio.run(retry.crawl([row], self.out, max_urls=1, delay=0,
                                             shard_size=2, crawler_factory=lambda: 1 / 0,
                                             run_config=object()))
        self.assertEqual(counts['attempted'], 0)
        self.assertIn('host cooldown', output.getvalue())
        self.assertNotIn('Browser ready', output.getvalue())

    def test_host_cooldown_expires_after_five_minutes(self):
        now = datetime.now(timezone.utc)
        recent = now - timedelta(minutes=4)
        expired = now - timedelta(minutes=6)
        previous = {
            1: dict(status='blocked', url='https://recent.example/a',
                    http_code='403', fetched_at=recent.isoformat()),
            2: dict(status='blocked', url='https://expired.example/a',
                    http_code='403', fetched_at=expired.isoformat()),
        }
        cooldowns = retry.active_cooldowns(previous, now)
        self.assertEqual(retry.HOST_COOLDOWN, timedelta(minutes=5))
        self.assertEqual(cooldowns['recent.example'], recent + timedelta(minutes=5))
        self.assertNotIn('expired.example', cooldowns)

    def test_default_crawler_uses_longchau_optimized_profile(self):
        row = dict(id=1, url='https://example.org/a', lang_expected='en', attempts=0)
        fake = FakeCrawler([FakeResponse(row['url'], b'<html>page</html>')])
        config = object()
        with patch.object(retry, 'create_optimized_crawler',
                          return_value=(lambda: fake, config)) as optimized:
            counts = asyncio.run(retry.crawl([row], self.out, max_urls=1, delay=0,
                                             shard_size=2))
        optimized.assert_called_once_with()
        self.assertEqual((counts['attempted'], counts['ok']), (1, 1))
        self.assertEqual(fake.urls, [row['url']])

    def test_pharmacity_public_url_keeps_source_id_and_syncs_final_url(self):
        source = 'https://pmc-ecm-healthblog.beta.pharmacity.io/article-slug/'
        public = 'https://www.pharmacity.vn/article-slug.htm'
        con = duckdb.connect(str(self.db))
        con.execute('INSERT INTO crawl_status (id,url,status,attempts) VALUES (12,?,?,0)',
                    [source, 'pending'])
        con.close()
        rows, selected = retry.pending_rows(
            self.db, self.out, only_hosts={'pmc-ecm-healthblog.beta.pharmacity.io'})
        self.assertEqual([row['id'] for row in rows], [12])
        self.assertEqual([row['id'] for row in selected], [12])
        fake = FakeCrawler([FakeResponse(public, b'<html>public article</html>')])
        counts = asyncio.run(retry.crawl(selected, self.out, max_urls=1, delay=0,
                                         shard_size=2, crawler_factory=lambda: fake,
                                         run_config=object(), auto_sync=True,
                                         status_db=self.db, html_dir=self.html))
        self.assertEqual((counts['attempted'], counts['ok']), (1, 1))
        self.assertEqual(fake.urls, [public])
        self.assertEqual(retry.local_status(self.out / 'status.csv')[12]['url'], source)
        con = duckdb.connect(str(self.db), read_only=True)
        try:
            self.assertEqual(con.execute(
                'SELECT url,final_url,status FROM crawl_status WHERE id=12').fetchone(),
                (source, public, 'ok'))
        finally:
            con.close()

    def test_raw_body_and_merge_are_main_compatible_and_resumable(self):
        body = b'<html><h1>Caf\xe9</h1></html>'
        rows = [dict(id=1, url='https://example.org/a', lang_expected='en', attempts=0),
                dict(id=4, url='https://example.org/b', lang_expected='en', attempts=0)]
        fake = FakeCrawler([FakeResponse(rows[0]['url'], body),
                            FakeResponse(rows[1]['url'], b'blocked', code=403)])
        counts = asyncio.run(retry.crawl(rows, self.out, max_urls=0, delay=0, shard_size=2,
                                         crawler_factory=lambda: fake, run_config=object()))
        self.assertEqual((counts['attempted'], counts['ok'], counts['blocked']), (2, 1, 1))
        self.assertEqual(len(fake.urls), 2)
        shard = next((self.out / 'html').glob('retry_shard_*.jsonl.zst'))
        with shard.open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as stream:
            record = json.loads(io.TextIOWrapper(stream, encoding='utf-8').readline())
        self.assertEqual(set(record), retry.RAW_FIELDS)
        self.assertEqual(base64.b64decode(record['body_b64']), body)
        self.assertEqual(record['encoding'], 'iso-8859-1')
        retry.merge(self.out, self.db, self.html, apply=True)
        main = next(self.html.glob('shard_*.jsonl.zst'))
        with main.open('rb') as handle, zstd.ZstdDecompressor().stream_reader(handle) as stream:
            merged = json.loads(io.TextIOWrapper(stream, encoding='utf-8').readline())
        self.assertEqual(merged, record)
        con = duckdb.connect(str(self.db), read_only=True)
        try:
            got = con.execute('SELECT status,shard,sha1_raw FROM crawl_status WHERE id=1').fetchone()
            blocked = con.execute('SELECT status,shard FROM crawl_status WHERE id=4').fetchone()
        finally:
            con.close()
        self.assertEqual(got, ('ok', 0, hashlib.sha1(body).hexdigest()))
        self.assertEqual(blocked, ('blocked', None))
        retry.merge(self.out, self.db, self.html, apply=True)
        self.assertEqual(len(list(self.html.glob('shard_*.jsonl.zst'))), 1)

    def test_run_auto_syncs_checkpoint_and_recovers_missing_cursor(self):
        body = b'<html>auto sync</html>'
        rows = [dict(id=1, url='https://example.org/a', lang_expected='en', attempts=0),
                dict(id=4, url='https://example.org/b', lang_expected='en', attempts=0)]
        fake = FakeCrawler([FakeResponse(rows[0]['url'], body),
                            FakeResponse(rows[1]['url'], b'blocked', code=403)])
        output = io.StringIO()
        with redirect_stdout(output):
            asyncio.run(retry.crawl(rows, self.out, max_urls=0, delay=0, shard_size=2,
                                    crawler_factory=lambda: fake, run_config=object(),
                                    auto_sync=True, status_db=self.db, html_dir=self.html))
        self.assertIn('Browser ready; requesting URLs.', output.getvalue())
        self.assertIn('queue=2/2', output.getvalue())
        con = duckdb.connect(str(self.db), read_only=True)
        try:
            self.assertEqual(con.execute('SELECT status,shard FROM crawl_status WHERE id=1').fetchone(),
                             ('ok', 0))
            self.assertEqual(con.execute('SELECT status FROM crawl_status WHERE id=4').fetchone(),
                             ('blocked',))
        finally:
            con.close()
        self.assertEqual(len(list(self.html.glob('shard_*.jsonl.zst'))), 1)
        self.assertEqual(retry.sync_incremental(self.out, self.db, self.html),
                         dict(bodies=0, errors=0))
        (self.out / 'sync_state.json').unlink()
        self.assertEqual(retry.sync_incremental(self.out, self.db, self.html)['bodies'], 0)
        self.assertEqual(len(list(self.html.glob('shard_*.jsonl.zst'))), 1)

    def test_sync_saved_vaccination_longchau_after_queue_exclusion(self):
        old = dict(id=7, url='https://tiemchunglongchau.com.vn/a',
                   lang_expected='vi', attempts=0)
        current = dict(id=1, url='https://example.org/a', lang_expected='vi', attempts=0)
        fake = FakeCrawler([FakeResponse(old['url'], b'<html>old result</html>'),
                            FakeResponse(current['url'], b'<html>new result</html>')])
        asyncio.run(retry.crawl([old, current], self.out, max_urls=0, delay=0,
                                shard_size=2, crawler_factory=lambda: fake,
                                run_config=object()))
        queued, _ = retry.pending_rows(self.db, self.out)
        self.assertNotIn(7, [row['id'] for row in queued])
        self.assertEqual(retry.sync_incremental(self.out, self.db, self.html),
                         dict(bodies=2, errors=0))
        con = duckdb.connect(str(self.db), read_only=True)
        try:
            self.assertEqual(con.execute(
                'SELECT id FROM crawl_status WHERE id IN (1,7) AND status = \'ok\' '
                'AND shard IS NOT NULL ORDER BY id').fetchall(), [(1,), (7,)])
        finally:
            con.close()
        self.assertEqual(retry.sync_incremental(self.out, self.db, self.html),
                         dict(bodies=0, errors=0))


if __name__ == '__main__':
    unittest.main()
