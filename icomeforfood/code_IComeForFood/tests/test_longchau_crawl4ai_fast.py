"""Unit tests for the fast Crawl4AI Long Chau engine."""
import asyncio
import csv
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import duckdb

from crawl import longchau_odd_fast as base
from crawl import longchau_crawl4ai_fast as engine


class FakeCrawler:
    def __init__(self, results, on_enter=None):
        self.results = iter(results)
        self.urls = []
        self.on_enter = on_enter

    async def __aenter__(self):
        if self.on_enter:
            self.on_enter()
        return self

    async def __aexit__(self, *_):
        return None

    async def arun(self, url, config):
        self.urls.append(url)
        return next(self.results)


class Crawl4AIFastTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.rows = [dict(id=1, url='https://nhathuoclongchau.com.vn/a'),
                     dict(id=3, url='https://nhathuoclongchau.com.vn/b')]
        self.args = SimpleNamespace(probe=True, scan=False, batch_size=50, delay=0, jitter=False)
        sleeper = mock.patch.object(base, 'wait_for_rest', return_value=True)
        self.wait_mock = sleeper.start()
        self.addCleanup(sleeper.stop)

    def run_collect(self, results, state=None):
        crawler = FakeCrawler(results)
        summary = asyncio.run(engine.collect(
            self.rows, self.out, state or dict(kind='unprobed'), self.args, base,
            crawler_factory=lambda: crawler, run_config=object()))
        return summary, crawler

    def test_csv_is_exact_crawl_scope_and_preserves_order(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            database = root / 'status.duckdb'
            corpus = root / 'longchau.csv'
            con = duckdb.connect(str(database))
            con.execute('''CREATE TABLE crawl_status (
                id BIGINT,url VARCHAR,status VARCHAR,http_code INTEGER,fetched_at TIMESTAMP)''')
            con.executemany('INSERT INTO crawl_status(id,url,status) VALUES (?,?,?)', [
                (1, self.rows[0]['url'], 'pending'),
                (3, self.rows[1]['url'], 'pending'),
                (5, 'https://nhathuoclongchau.com.vn/not-in-csv', 'pending'),
            ])
            con.close()
            with corpus.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['id', 'url'])
                writer.writeheader()
                writer.writerows([self.rows[1], self.rows[0]])
            rows, statuses = base.source_rows(corpus=corpus, database=database)
            self.assertEqual([row['id'] for row in rows], [3, 1])
            self.assertEqual([row['id'] for row in base.queue_rows(rows, statuses, {}, set())], [3, 1])
            with corpus.open('a', newline='', encoding='utf-8') as handle:
                csv.writer(handle).writerow([2, 'https://nhathuoclongchau.com.vn/even'])
            with self.assertRaisesRegex(ValueError, 'even ID'):
                base.source_rows(corpus=corpus, database=database)

    def test_success_saves_pipeline_html_shape(self):
        result = SimpleNamespace(url=self.rows[0]['url'], status_code=200,
                                 redirected_status_code=None, response_headers={},
                                 html='<html><h1>Xin chao Long Chau</h1></html>',
                                 error_message=None, success=True)
        summary, crawler = self.run_collect([result])
        self.assertEqual(summary['ok'], 1)
        self.assertEqual(crawler.urls, [self.rows[0]['url']])
        self.assertEqual(base.raw_ids(self.out), {1})
        self.assertEqual(base.last_status(self.out / 'status.csv')[1]['status'], 'ok')
        self.assertGreaterEqual(self.wait_mock.call_args.args[1], 20.0)

    def test_rate_ramp_reaches_cap_after_three_hours_twenty_minutes(self):
        cases = [(0, 0.05, 20.0), (1199, 0.05, 20.0),
                 (1200, 0.075, 1 / 0.075), (2400, 0.10, 10.0),
                 (7200, 0.20, 5.0), (10800, 0.275, 1 / 0.275),
                 (12000, 0.30, 1 / 0.30), (100000, 0.30, 1 / 0.30)]
        for elapsed, expected_rate, expected_delay in cases:
            with self.subTest(elapsed=elapsed):
                _, rate, delay = engine.ramp_pacing(elapsed, base.DEFAULT_DELAY)
                self.assertAlmostEqual(rate, expected_rate)
                self.assertAlmostEqual(delay, expected_delay)
        self.assertEqual(engine.ramp_pacing(12000, 5.0)[2], 5.0)

    def test_new_run_starts_at_safe_rate_after_browser_startup(self):
        result = SimpleNamespace(url=self.rows[0]['url'], status_code=200,
                                 redirected_status_code=None, response_headers={},
                                 html='<html>Page</html>', error_message=None, success=True)
        offset = [0]
        real_monotonic = engine.time.monotonic
        crawler = FakeCrawler([result], on_enter=lambda: offset.__setitem__(0, 7200))
        with mock.patch.object(engine.time, 'monotonic',
                               side_effect=lambda: real_monotonic() + offset[0]), \
             mock.patch.object(engine, 'ramp_pacing', wraps=engine.ramp_pacing) as pacing:
            asyncio.run(engine.collect(
                self.rows, self.out, dict(kind='unprobed'), self.args, base,
                crawler_factory=lambda: crawler, run_config=object()))
        self.assertLess(pacing.call_args_list[0].args[0], 1.0)
        self.assertGreaterEqual(self.wait_mock.call_args.args[1], 20.0)

    def test_block_stops_immediately_and_sets_cooldown(self):
        self.args.probe = False
        self.args.scan = True
        result = SimpleNamespace(url=self.rows[0]['url'], status_code=403,
                                 redirected_status_code=None, response_headers={},
                                 html='<html>Access denied</html>',
                                 error_message='Forbidden', success=False)
        summary, crawler = self.run_collect([result])
        self.assertEqual(crawler.urls, [self.rows[0]['url']])
        self.assertEqual(summary['cooldown']['kind'], 'blocked')
        self.assertGreaterEqual(base.timestamp(summary['cooldown']['next_probe_at']),
                                base.now() + timedelta(minutes=9, seconds=50))
        self.assertNotIn(3, base.last_status(self.out / 'status.csv'))

    def test_recent_home_block_delays_fast_scan(self):
        data = self.out / 'data'
        home = data / 'crawl_home_tv'
        home.mkdir(parents=True)
        fast = data / 'longchau_odd_fast'
        recent = base.now() - timedelta(minutes=2)
        (home / 'status.csv').write_text(
            'id,url,status,http_code,fetched_at\n'
            f'1,https://nhathuoclongchau.com.vn/a,blocked,403,{recent.isoformat()}\n',
            encoding='utf-8')
        state = dict(kind='scan_ready', next_probe_at=None)
        guarded = base.home_cooldown_state(state, fast, 10)
        self.assertIn('Cooldown active', base.gate(guarded, probe=False, scan=True))
        self.assertEqual(base.home_cooldown_state(state, fast, 10), guarded)

    def test_one_navigation_timeout_skips_url_and_continues_scan(self):
        self.args.probe = False
        self.args.scan = True
        self.args.continuous = True
        timeout = SimpleNamespace(url=self.rows[0]['url'], status_code=None,
                                  redirected_status_code=None, response_headers={}, html='',
                                  error_message='Page.goto: Timeout 25000ms exceeded', success=False)
        success = SimpleNamespace(url=self.rows[1]['url'], status_code=200,
                                  redirected_status_code=None, response_headers={},
                                  html='<html>Next page</html>', error_message=None, success=True)
        summary, crawler = self.run_collect([timeout, success])
        self.assertEqual(crawler.urls, [row['url'] for row in self.rows])
        self.assertEqual((summary['attempted'], summary['ok']), (2, 1))
        self.assertEqual(base.last_status(self.out / 'status.csv')[1]['status'], 'timeout')
        self.assertEqual(base.raw_ids(self.out), {3})
        self.assertEqual(summary['cooldown']['kind'], 'scan_ready')

    def test_three_consecutive_navigation_timeouts_stop_scan(self):
        self.args.probe = False
        self.args.scan = True
        self.rows = [dict(id=i, url=f'https://nhathuoclongchau.com.vn/{i}')
                     for i in (1, 3, 5, 7)]
        timeout = SimpleNamespace(url='', status_code=None, redirected_status_code=None,
                                  response_headers={}, html='',
                                  error_message='Page.goto: Timeout 25000ms exceeded', success=False)
        summary, crawler = self.run_collect([timeout] * 4)
        self.assertEqual(len(crawler.urls), 3)
        self.assertEqual((summary['attempted'], summary['ok']), (3, 0))
        self.assertEqual({row['status'] for row in base.last_status(self.out / 'status.csv').values()},
                         {'timeout'})

    def test_fast_crawler_config_defaults(self):
        factory, run_config = engine.create_optimized_crawler()
        crawler = factory()
        bc = crawler.browser_config
        self.assertTrue(bc.text_mode)
        self.assertTrue(bc.light_mode)
        self.assertTrue(bc.memory_saving_mode)
        self.assertTrue(run_config.prefetch)
        self.assertTrue(run_config.check_robots_txt)
        self.assertEqual(run_config.page_timeout, 45000)


if __name__ == '__main__':
    unittest.main()
