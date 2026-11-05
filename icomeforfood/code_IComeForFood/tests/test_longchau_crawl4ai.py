"""Offline contract tests for the Crawl4AI Long Chau engine."""
import asyncio
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from crawl import longchau_odd as base
from crawl import longchau_crawl4ai as engine


class FakeCrawler:
    def __init__(self, results):
        self.results = iter(results)
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def arun(self, url, config):
        self.urls.append(url)
        return next(self.results)


class Crawl4AILongChauTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.rows = [dict(id=1,url='https://nhathuoclongchau.com.vn/a'),
                     dict(id=3,url='https://nhathuoclongchau.com.vn/b')]
        self.args = SimpleNamespace(probe=True,scan=False,batch_size=50,delay=0)

    def run_collect(self, results, state=None):
        crawler = FakeCrawler(results)
        summary = asyncio.run(engine.collect(
            self.rows,self.out,state or dict(kind='unprobed'),self.args,base,
            crawler_factory=lambda:crawler,run_config=object()))
        return summary,crawler

    def test_success_saves_pipeline_html_shape(self):
        result = SimpleNamespace(url=self.rows[0]['url'],status_code=200,
                                 redirected_status_code=None,response_headers={},
                                 html='<html><h1>Xin chào</h1></html>',
                                 error_message=None,success=True)
        summary,crawler = self.run_collect([result])
        self.assertEqual(summary['ok'],1)
        self.assertEqual(crawler.urls,[self.rows[0]['url']])
        self.assertEqual(base.raw_ids(self.out),{1})
        self.assertEqual(base.last_status(self.out/'status.csv')[1]['status'],'ok')

    def test_block_stops_after_first_page_and_sets_cooldown(self):
        self.args.probe = False
        self.args.scan = True
        result = SimpleNamespace(url=self.rows[0]['url'],status_code=403,
                                 redirected_status_code=None,response_headers={},
                                 html='<html>Access denied</html>',
                                 error_message='Forbidden',success=False)
        summary,crawler = self.run_collect([result])
        self.assertEqual(crawler.urls,[self.rows[0]['url']])
        self.assertEqual(summary['cooldown']['kind'],'blocked')
        self.assertGreaterEqual(base.timestamp(summary['cooldown']['next_probe_at']),
                                base.now()+timedelta(hours=5,minutes=59))
        self.assertNotIn(3,base.last_status(self.out/'status.csv'))

    def test_continuous_scan_saves_batches_without_fixed_rest(self):
        self.args.probe = False
        self.args.scan = True
        self.args.continuous = True
        self.args.batch_size = 1
        results = [SimpleNamespace(url=row['url'],status_code=200,
                                   redirected_status_code=None,response_headers={},
                                   html='<html>Article</html>',error_message=None,
                                   success=True) for row in self.rows]
        with patch.object(base,'wait_for_rest') as rest:
            summary,crawler = self.run_collect(results)
        self.assertEqual(crawler.urls,[row['url'] for row in self.rows])
        self.assertEqual(summary['ok'],2)
        self.assertEqual(base.raw_ids(self.out),{1,3})
        rest.assert_not_called()


if __name__=='__main__':
    unittest.main()
