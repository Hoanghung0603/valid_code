"""Offline contracts for VOV classification, durable resume and stop-on-block."""
import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import duckdb

from crawl import vov
from scripts import run_local


URL = 'https://vov.vn/suc-khoe/bai-post123.vov'
ARTICLE = ('<html><head><meta charset="utf-8"><style>.grecaptcha-badge { visibility:hidden }</style>'
           '<script src="https://www.google.com/recaptcha/api.js"></script></head>'
           '<body><h1>Bài sức khỏe</h1><div class="text-long"><p>' + 'Nội dung thật. ' * 30 +
           '</p></div></body></html>').encode('utf-8')
VIDEO = ('<html><head><meta charset="utf-8"></head><body><article class="detail detail--video">'
         '<div class="video"><div class="video_file" data-url="https://media.vov.vn/example.mp4"></div>'
         '<h3 class="video__title">Video sức khỏe</h3>'
         '<div class="video__summary">' + 'Mô tả video về sức khỏe. ' * 5 +
         '</div></div></article></body></html>').encode('utf-8')
EMBEDDED_VIDEO = ('<html><head><meta charset="utf-8"></head><body>'
                  '<article class="detail detail--normal"><h1>Video trong bài thường</h1>'
                  '<div class="article-summary">' + 'Mô tả thông tin video. ' * 5 + '</div>'
                  '<div class="article-content"><div class="text-long">'
                  '<video src="/example.mp4" controls></video>'
                  '</div></div></article></body></html>').encode('utf-8')
IMAGE_ARTICLE = EMBEDDED_VIDEO.replace(
    b'<video src="/example.mp4" controls></video>',
    b'<table class="contentimg"><tr><td><img src="https://media.vov.vn/infographic.jpg" alt=""></td></tr></table>')


class VovTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'out'
        self.items = [dict(id=2, url=URL, status='blocked', http_code=403, attempts=3),
                      dict(id=3, url='https://vov.vn/suc-khoe/khac-post456.vov', status='blocked', http_code=200, attempts=1)]
        self.args = SimpleNamespace(probe=False, scan=True, batch_size=50, delay=0, limit=None, channel='chrome')

    def collect(self, responses, state=None, recover=None, before_goto=None):
        responses = iter(responses)
        page = SimpleNamespace(url=URL)
        async def goto(url, **kwargs):
            page.url = url
            if before_goto:
                before_goto()
            result = next(responses)
            if isinstance(result, Exception):
                raise result
            return result
        page.goto = AsyncMock(side_effect=goto)
        @asynccontextmanager
        async def factory():
            yield page
        summary = asyncio.run(vov.collect(self.items, self.out, state or {'kind': 'unprobed'}, self.args,
                                          page_factory=factory, recover=recover))
        return summary, page

    def response(self, code=200, raw=ARTICLE, **headers):
        return SimpleNamespace(status=code, body=AsyncMock(return_value=raw),
                               headers={'content-type': 'text/html; charset=utf-8', **headers})

    def test_real_article_with_recaptcha_is_not_blocked(self):
        self.assertEqual(vov.classify(ARTICLE, 200, {}, URL), ('ok', None))
        # Conversely, an HTTP or response-header challenge wins even with article markup.
        self.assertEqual(vov.classify(ARTICLE, 403, {}, URL)[0], 'blocked')
        self.assertEqual(vov.classify(ARTICLE, 200, {'CF-Mitigated': 'challenge'}, URL)[0], 'blocked')
        self.assertEqual(vov.classify(b'<h1>Verify you are human</h1>', 200, {}, URL)[0], 'blocked')
        self.assertEqual(vov.classify(b'<h1>Short HTML</h1>', 200, {}, URL)[0], 'ok')

    def test_source_includes_even_ids_and_excludes_vov2_and_other_hosts(self):
        database = self.root / 'status.duckdb'
        with duckdb.connect(str(database)) as con:
            con.execute('CREATE TABLE crawl_status(id BIGINT,url VARCHAR,status VARCHAR,http_code INTEGER,attempts INTEGER,fetched_at TIMESTAMP)')
            con.executemany('INSERT INTO crawl_status VALUES (?,?,?,?,?,NULL)', [
                (2, URL, 'blocked', 403, 3),
                (3, 'https://www.vov.vn/other.vov', 'pending', None, 0),
                (4, 'https://vov2.vov.vn/other.vov2', 'ok', 200, 1),
                (5, 'https://notvov.vn/other', 'pending', None, 0)])
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        rows = vov.source_rows(database)
        self.assertEqual([r['id'] for r in rows], [2, 3])
        self.assertEqual(hashlib.sha256(database.read_bytes()).hexdigest(), before)

    def test_video_template_without_h1_or_text_long_is_saved_and_scan_continues(self):
        self.assertEqual(vov.classify(VIDEO, 200, {}, URL), ('ok', None))
        summary, page = self.collect([self.response(raw=VIDEO), self.response()])
        self.assertEqual((summary['attempted'], summary['ok']), (2, 2))
        self.assertEqual(page.goto.await_count, 2)
        saved = vov.saved_records(self.out, {r['id']: r for r in self.items})
        self.assertEqual(set(saved), {2, 3})

    def test_video_html_is_collected_without_template_requirements(self):
        for fragment in [b'data-url="https://media.vov.vn/example.mp4"',
                         b'video__title', b'video__summary', b'detail--video']:
            with self.subTest(fragment=fragment):
                self.assertEqual(vov.classify(VIDEO.replace(fragment, b''), 200, {}, URL)[0], 'ok')
        # Video markup does not override an explicit access refusal.
        self.assertEqual(vov.classify(VIDEO, 403, {}, URL)[0], 'blocked')
        self.assertEqual(vov.classify(VIDEO, 200, {'cf-mitigated': 'challenge'}, URL)[0], 'blocked')

    def test_normal_article_with_video_only_body_is_saved_and_scan_continues(self):
        self.assertEqual(vov.classify(EMBEDDED_VIDEO, 200, {}, URL), ('ok', None))
        summary, page = self.collect([self.response(raw=EMBEDDED_VIDEO), self.response()])
        self.assertEqual((summary['attempted'], summary['ok']), (2, 2))
        self.assertEqual(page.goto.await_count, 2)
        self.assertEqual(set(vov.saved_records(self.out, {r['id']: r for r in self.items})), {2, 3})

    def test_raw_collection_does_not_require_article_title_summary_or_media(self):
        for fragment in [b'src="/example.mp4"', b'<h1>Video trong b', b'article-summary', b'text-long']:
            with self.subTest(fragment=fragment):
                invalid = EMBEDDED_VIDEO.replace(fragment, b'')
                self.assertEqual(vov.classify(invalid, 200, {}, URL)[0], 'ok')
        # Even a short page with no media is still collected in full.
        invalid = EMBEDDED_VIDEO.replace(b'<video src="/example.mp4" controls></video>', b'')
        invalid += b'<aside><video src="/related.mp4"></video></aside>'
        self.assertEqual(vov.classify(invalid, 200, {}, URL)[0], 'ok')
        self.assertEqual(vov.classify(EMBEDDED_VIDEO, 403, {}, URL)[0], 'blocked')

    def test_image_only_article_saves_html_and_scan_continues(self):
        self.assertEqual(vov.classify(IMAGE_ARTICLE, 200, {}, URL), ('ok', None))
        summary, page = self.collect([self.response(raw=IMAGE_ARTICLE), self.response()])
        self.assertEqual((summary['attempted'], summary['ok']), (2, 2))
        self.assertEqual(page.goto.await_count, 2)
        self.assertEqual(set(vov.saved_records(self.out, {r['id']: r for r in self.items})), {2, 3})

    def test_image_attributes_do_not_filter_or_reject_the_raw_page(self):
        for image in [b'<img src="">', b'<img src="data:image/png;base64,aA==">',
                      b'<img src="/track.png" width="1" height="1">']:
            with self.subTest(image=image):
                invalid = IMAGE_ARTICLE.replace(b'<img src="https://media.vov.vn/infographic.jpg" alt="">', image)
                self.assertEqual(vov.classify(invalid, 200, {}, URL)[0], 'ok')
        invalid = IMAGE_ARTICLE.replace(b'<img src="https://media.vov.vn/infographic.jpg" alt="">', b'')
        invalid += b'<aside><img src="/related.jpg"></aside>'
        self.assertEqual(vov.classify(invalid, 200, {}, URL)[0], 'ok')
        self.assertEqual(vov.classify(IMAGE_ARTICLE, 200, {'cf-mitigated': 'challenge'}, URL)[0], 'blocked')
        # A video landing page with only a poster is collected too.
        invalid = VIDEO.replace(b'<div class="video_file" data-url="https://media.vov.vn/example.mp4"></div>', b'<img src="/poster.jpg">')
        self.assertEqual(vov.classify(invalid, 200, {}, URL)[0], 'ok')

    def test_probe_saves_exact_response_bytes_and_resume_skips_only_durable_ids(self):
        self.args.probe = True
        summary, page = self.collect([self.response()], state={'kind': 'unprobed', 'reason': 'Old inspection error', 'blocked_id': 2})
        self.assertEqual((summary['attempted'], summary['ok']), (1, 1))
        self.assertNotIn('reason', summary['cooldown'])
        self.assertNotIn('blocked_id', summary['cooldown'])
        page.goto.assert_awaited_once()
        saved = vov.saved_records(self.out, {r['id']: r for r in self.items})
        self.assertEqual(set(saved), {2})
        status = vov.last_status(self.out / 'status.csv')
        self.assertEqual(status[2]['attempts'], '4')
        self.assertEqual(status[2]['sha1_raw'], hashlib.sha1(ARTICLE).hexdigest())
        path = next((self.out / 'html').glob('*.jsonl.zst'))
        raw = json.loads(vov.zstd.ZstdDecompressor().decompress(path.read_bytes()))
        self.assertEqual(set(raw), {'id', 'final_url', 'http_code', 'content_type', 'encoding', 'body_b64'})
        self.assertEqual(base64.b64decode(raw['body_b64']), ARTICLE)
        self.assertEqual([r['id'] for r in vov.queue_rows(self.items, status, saved)], [3])
        self.assertEqual([r['id'] for r in vov.queue_rows(self.items, status, {})], [2, 3])

    def test_block_stops_after_first_request_and_retry_after_extends_cooldown(self):
        summary, page = self.collect([self.response(429, b'Forbidden', **{'retry-after': '86400'})])
        page.goto.assert_awaited_once()
        self.assertEqual(summary['ok'], 0)
        self.assertEqual(summary['stop_reason'], 'blocked')
        self.assertEqual(set(vov.last_status(self.out / 'status.csv')), {2})
        state = json.loads((self.out / 'cooldown.json').read_text())
        self.assertGreater(vov.timestamp(state['next_probe_at']), vov.now() + timedelta(hours=23))
        with self.assertRaisesRegex(RuntimeError, 'cooldown'):
            self.collect([], state)

    def test_default_block_cooldown_is_ten_minutes(self):
        current = vov.now()
        with patch.object(vov, 'now', return_value=current):
            state = vov.blocked_state({}, self.items[0], 403, {}, 'Forbidden')
            self.assertEqual(vov.timestamp(state['next_probe_at']), current + timedelta(minutes=10))
            self.assertIsNotNone(vov.gate(state))
        with patch.object(vov, 'now', return_value=current + timedelta(minutes=10)):
            self.assertIsNone(vov.gate(state))

    def test_stop_checkpoint_is_resumable_and_partial_status_can_be_recovered(self):
        def stop():
            (self.out / 'STOP').touch()
        summary, page = self.collect([self.response()], before_goto=stop)
        page.goto.assert_awaited_once()
        self.assertEqual(summary['stop_reason'], 'STOP')
        saved = vov.saved_records(self.out, {r['id']: r for r in self.items})
        self.assertEqual(set(saved), {2})
        # Simulate losing the final CSV line after a durable shard rename.
        (self.out / 'status.csv').write_text(','.join(vov.STATUS_COLUMNS) + '\n', encoding='utf-8')
        self.items = []
        summary, page = self.collect([], recover=saved)
        page.goto.assert_not_awaited()
        self.assertEqual(vov.last_status(self.out / 'status.csv')[2]['status'], 'ok')

    def test_404_continues_but_empty_200_and_network_error_stop(self):
        summary, _ = self.collect([self.response(404, b'Missing'), self.response()])
        self.assertEqual((summary['attempted'], summary['ok']), (2, 1))
        local = vov.last_status(self.out / 'status.csv')
        self.assertEqual(vov.queue_rows(self.items, local, {3: {}}), [])
        for response in [self.response(200, b''), TimeoutError('timeout')]:
            with self.subTest(response=response):
                summary, page = self.collect([response])
                self.assertEqual((summary['attempted'], summary['ok']), (1, 0))
                page.goto.assert_awaited_once()
        self.assertEqual((self.out / 'inspection/2.html').read_bytes(), b'')

    def test_new_layout_keeps_all_html_bytes_and_original_asset_links(self):
        raw = (b'<html><head><meta name="custom" content="keep">'
               b'<style>body{background:url(/background.jpg)}</style></head>'
               b'<body><section><p>Brief</p><table><tr><td>123</td></tr></table>'
               b'<img src="https://media.vov.vn/infographic.jpg">'
               b'<video src="/video.mp4"></video><script>custom()</script>'
               b'</section></body></html>')
        self.args.probe = True
        summary, _ = self.collect([self.response(raw=raw)])
        self.assertEqual(summary['ok'], 1)
        shard = next((self.out / 'html').glob('*.jsonl.zst'))
        record = json.loads(vov.zstd.ZstdDecompressor().decompress(shard.read_bytes()))
        self.assertEqual(base64.b64decode(record['body_b64']), raw)

    def test_article_discussing_captcha_is_not_a_challenge_but_real_gate_is(self):
        raw = b'<article><h1>Explaining access denied</h1><p>Verify you are human is a prompt.</p></article>'
        self.assertEqual(vov.classify(raw, 200, {}, URL)[0], 'ok')
        gate = b'<html><title>Just a moment...</title><body><form id="challenge-form"></form></body></html>'
        self.assertEqual(vov.classify(gate, 200, {}, URL)[0], 'blocked')

    def test_domain_restriction_and_lock(self):
        for url in ['https://vov2.vov.vn/a', 'https://vov.vn.evil/a', 'https://vov.vn:1234/a', 'https://a@vov.vn/a']:
            self.assertFalse(vov.is_vov(url))
        with vov.exclusive_lock(self.root / 'locks/vov.lock'):
            with self.assertRaisesRegex(RuntimeError, 'VOV job'):
                with vov.exclusive_lock(self.root / 'locks/vov.lock'):
                    pass
        self.items[0]['url'] = 'https://other.invalid/a'
        with self.assertRaisesRegex(ValueError, 'non-VOV'):
            self.collect([])

    def test_cli_vov_routes_without_loading_notebook_or_requiring_parquet(self):
        with patch('subprocess.call', return_value=0) as call, patch.object(run_local, 'check_environment', side_effect=AssertionError('not needed')):
            result = run_local.main(['vov', '--scan', '--max-urls', '3', '--out', str(self.out)])
        self.assertEqual(result, 0)
        command = call.call_args.args[0]
        self.assertIn(str(run_local.REPO / 'crawl/vov.py'), command)
        self.assertIn('--scan', command)
        self.assertEqual(command[command.index('--limit') + 1], '3')
        self.assertEqual(command[command.index('--delay') + 1], '2.0')

    def test_shared_cli_preserves_longchau_delay_and_vov_stop(self):
        with patch('subprocess.call', return_value=0) as call:
            run_local.main(['longchau-odd'])
        command = call.call_args.args[0]
        self.assertEqual(command[command.index('--delay') + 1], '30.0')
        (self.root / 'crawl/vov').mkdir(parents=True)
        run_local.main(['stop', '--data-root', str(self.root)])
        self.assertTrue((self.root / 'crawl/vov/STOP').is_file())


if __name__ == '__main__':
    unittest.main()
