import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))

import crawl_browser_probe as p


def test_host_of_strips_www():
    assert p.host_of("https://www.qdnd.vn/x") == "qdnd.vn"
    assert p.host_of("http://ask.familydoctor.com.cn/q/1.html") == "ask.familydoctor.com.cn"


def test_parse_html_extracts_text_not_raw(tmp_path):
    html = ("<html><head><title>T</title></head><body><article><p>" +
            "nội dung bài viết " * 40 + "</p></article></body></html>").encode()
    ex = p.parse_main_html(html, "https://baotayninh.vn/x.html")
    assert "<html" not in (ex.get("text") or "")
    assert len(ex.get("text") or "") >= 200


def test_pick_main_response_excludes_subresources():
    ev = [
        {"url": "https://a.vn/x.css", "status": 200, "type": "Stylesheet", "frame_main": True},
        {"url": "https://a.vn/x", "status": 200, "type": "Document", "frame_main": True},
        {"url": "https://a.vn/img.png", "status": 200, "type": "Image", "frame_main": True},
    ]
    assert p.pick_main_response(ev, "https://a.vn/x") == (200, "https://a.vn/x")
    assert p.pick_main_response([], "https://a.vn/x") == (None, "https://a.vn/x")


class _FakePage:
    def __init__(self, html, cdp_status=200, cdp_url=None):
        self._html = html
        self._st = cdp_status
        self._url = cdp_url

    async def sleep(self, s):
        return None

    async def get_content(self):
        return self._html

    async def get_cdp_session(self):
        raise RuntimeError("no cdp in fake")


class _FakeBrowser:
    def __init__(self, pages):
        self.pages = list(pages)
        self.stopped = False

    async def get(self, url):
        spec = self.pages.pop(0)
        if isinstance(spec, Exception):
            raise spec
        return spec

    def stop(self):
        self.stopped = True


def _ok_html(n=60):
    return "<html><body><article><p>" + "chữ nội dung " * n + "</p></article></body></html>"


def test_run_event_fake_blocked_403_captured_no_record(tmp_path):
    items = [{"id": 1, "url": "https://familydoctor.com.cn/x"}]
    async def go():
        fb = _FakeBrowser([_FakePage("<html><body>forbidden</body></html>", 403)])
        # monkeypatch CDP wiring to inject a 403 main-document response
        async def fake_wire(page, store):
            store.append({"url": items[0]["url"], "status": 403,
                          "type": "Document", "frame_main": True})
            return None
        orig = p._wire_cdp
        p._wire_cdp = fake_wire
        try:
            return await p.run(items, 9, 3.0, tmp_path / "o",
                               crawl_root=tmp_path, browser_factory=lambda: fb)
        finally:
            p._wire_cdp = orig
    rc = asyncio.run(go())
    assert rc == 0
    assert (tmp_path / "o.audit.jsonl").exists()
    aud = [json.loads(x) for x in (tmp_path / "o.audit.jsonl").read_text().splitlines()]
    assert aud[0]["status"] == 403 and aud[0]["ok"] is False
    assert not (tmp_path / "o.records.jsonl").read_bytes().strip()


def test_run_navigation_redirect_final_url_kept(tmp_path):
    items = [{"id": 7, "url": "https://www.qdnd.vn/a"}]
    async def go():
        fb = _FakeBrowser([_FakePage(_ok_html(80))])
        async def fake_wire(page, store):
            store.append({"url": "https://www.qdnd.vn/a-final", "status": 200,
                          "type": "Document", "frame_main": True})
            return None
        orig = p._wire_cdp
        p._wire_cdp = fake_wire
        try:
            return await p.run(items, 9, 3.0, tmp_path / "o",
                               crawl_root=tmp_path, browser_factory=lambda: fb)
        finally:
            p._wire_cdp = orig
    assert asyncio.run(go()) == 0
    recs = [json.loads(x) for x in (tmp_path / "o.records.jsonl").read_text().splitlines()]
    assert recs and recs[0]["final_url"] == "https://www.qdnd.vn/a-final"
    assert recs[0]["status"] == 200 and "<html" not in recs[0]["text"]


def test_run_refuses_overwrite_and_stop_lock(tmp_path):
    (tmp_path / "o.records.jsonl").write_text("x")
    rc = asyncio.run(p.run([{"id": 1, "url": "https://a.vn/"}],
                           1, 3.0, tmp_path / "o", crawl_root=tmp_path,
                           browser_factory=lambda: _FakeBrowser([])))
    assert rc == 4
    (tmp_path / "STOP").write_text("")
    rc = asyncio.run(p.run([{"id": 1, "url": "https://a.vn/"}],
                           1, 3.0, tmp_path / "o2", crawl_root=tmp_path,
                           browser_factory=lambda: _FakeBrowser([])))
    assert rc == 3
    (tmp_path / "STOP").unlink()
    (tmp_path / "heavy_job.lock").write_text("")
    rc = asyncio.run(p.run([{"id": 1, "url": "https://a.vn/"}],
                           1, 3.0, tmp_path / "o3", crawl_root=tmp_path,
                           browser_factory=lambda: _FakeBrowser([])))
    assert rc == 3


def test_run_bounds_and_browser_stop_finally(tmp_path):
    try:
        asyncio.run(p.run([{"id": 1, "url": "https://a.vn/"}], 10, 3.0,
                          tmp_path / "o", crawl_root=tmp_path,
                          browser_factory=lambda: _FakeBrowser([])))
        assert False
    except ValueError:
        pass
    try:
        asyncio.run(p.run([{"id": 1, "url": "https://a.vn/"}], 1, 1.0,
                          tmp_path / "o", crawl_root=tmp_path,
                          browser_factory=lambda: _FakeBrowser([])))
        assert False
    except ValueError:
        pass
    try:
        p.refuse_production_without_traffic_cap()
        assert False
    except SystemExit:
        pass
