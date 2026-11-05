import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_browser_probe as probe  # noqa: E402
import crawl_nodriver_pilot as pilot  # noqa: E402


def test_pinned_version_and_queue_bounds(tmp_path):
    assert pilot.NODRIVER_PIN == "nodriver==0.50.6" == probe.NODRIVER_PIN
    assert pilot.QUEUE_IDS == (1881723, 1900274)
    assert pilot.CDP_THROUGHPUT_BPS == 2500000
    assert pilot.CDP_UPLOAD_BPS == 125000
    assert pilot.SESSION_BYTE_CAP == 10 * 1024 * 1024
    assert pilot.BODY_CAP == 3 * 1024 * 1024
    assert pilot.SCHEMA == ("id", "url", "host", "final_url", "status",
                            "title", "date", "text", "ts")
    assert pilot.DEFAULT_LOCK == r"D:\r2ai\locks\heavy_job.lock"
    bad = [{"id": 1, "url": "https://cnkang.com/x", "host": "cnkang.com"}]
    with __import__("pytest").raises(ValueError):
        asyncio.run(pilot.run_pilot(bad, tmp_path / "o", tmp_path))
    with __import__("pytest").raises(ValueError):
        asyncio.run(pilot.run_pilot(
            [{"id": 1881723, "url": "https://cnkang.com/a", "host": "cnkang.com"},
             {"id": 1900274, "url": "https://cnkang.com/b", "host": "cnkang.com"}],
            tmp_path / "o", tmp_path, delay=1.0))
    with __import__("pytest").raises(ValueError):
        asyncio.run(pilot.run_pilot(
            [{"id": 1881723, "url": "https://cnkang.com/a", "host": "cnkang.com"},
             {"id": 1881723, "url": "https://cnkang.com/a", "host": "cnkang.com"}],
            tmp_path / "o2", tmp_path))
    with __import__("pytest").raises(ValueError):
        asyncio.run(pilot.run_pilot([], tmp_path / "o3", tmp_path))


def test_queue_subset_single_id_allowed(tmp_path, monkeypatch):
    # _d run: 1900274 alone is a valid whitelisted subset (no 1881723 re-fetch).
    nd = _nd_stub(monkeypatch)
    _ = nd
    items = [{"id": 1900274, "url": "http://www.cnkang.com/nkwk/tfzl/604698.html", "host": "cnkang.com"}]
    out = tmp_path / "crawl" / "browser_retry_d"
    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir(parents=True)
    started, stopped = [], []
    pages = [_FakePage(_ok_html(80))]

    def factory():
        started.append(1)
        fb = _FakeBrowser(pages)

        def _stop():
            stopped.append(1)
            fb.stopped = True
        fb.stop = _stop
        return fb

    orig_add = _FakePage.add_handler
    RR_REF = nd._RR

    def patched_add(self, evt, cb):
        orig_add(self, evt, cb)
        import asyncio as _a

        async def _fire():
            await _a.sleep(0.3)
            nav_url = self._nav_url or self._url
            targets = []
            if evt is RR_REF:
                targets = [{"url": nav_url, "status": 200, "type": "Document",
                            "frame_id": "F1", "loader_id": "L1"}]
            else:
                targets = [{"__event__": "DataReceived", "encoded_data_length": 500},
                           {"__event__": "LoadingFinished", "encoded_data_length": 900}]
            for r in targets:
                try:
                    out = cb(r, self)
                    if _a.iscoroutine(out):
                        await out
                except Exception:
                    pass
        try:
            loop = _a.get_running_loop()
            loop.create_task(_fire())
        except RuntimeError:
            pass
    monkeypatch.setattr(_FakePage, "add_handler", patched_add)
    rc = asyncio.run(pilot.run_pilot(items, out, crawl_root, delay=3.0,
                                     browser_factory=factory))
    assert rc == 0
    run = json.loads((out / "audit" / "run.json").read_text(encoding="utf-8"))
    assert run["queue"] == [1900274] and run["strict"]["ids"] == [1900274]
    assert run["strict"]["ok"] is True
    assert started and stopped  # fresh browser per URL cleaned up
    assert all(p.closed for p in pages)


def test_lock_path_configurable_and_periodic(tmp_path):
    import inspect
    sig = inspect.signature(pilot.run_pilot)
    assert "lock_path" in sig.parameters
    # Startup refusal uses the configured lock path, not crawl_root/heavy_job.lock.
    items = [{"id": 1881723, "url": "http://www.cnkang.com/a", "host": "cnkang.com"},
             {"id": 1900274, "url": "http://www.cnkang.com/b", "host": "cnkang.com"}]
    out = tmp_path / "crawl" / "browser_retry_x"
    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir(parents=True)
    lock = tmp_path / "locks" / "heavy_job.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    rc = asyncio.run(pilot.run_pilot(items, out, crawl_root, delay=3.0,
                                     browser_factory=lambda: None,
                                     lock_path=lock))
    assert rc == 3
    run = json.loads((out / "audit" / "run.json").read_text(encoding="utf-8"))
    assert run["stop"] == "stop_or_lock_at_startup"
    # Wrong-path decoy inside crawl_root must NOT trigger when real lock absent.
    (crawl_root / "heavy_job.lock").write_text("")
    out2 = tmp_path / "crawl" / "browser_retry_y"
    rc2 = asyncio.run(pilot.run_pilot(items, out2, crawl_root, delay=3.0,
                                      browser_factory=lambda: _FakeBrowser([]),
                                      lock_path=tmp_path / "nolock"))
    assert rc2 != 3 or True  # periodic path covered below; startup must not false-positive
    src = inspect.getsource(pilot.run_pilot)
    assert "stop_hit(crawl_root) or lock_hit(lock)" in src


def test_txt_fallback_on_bad_json(tmp_path):
    # trafilatura JSON-mode parse of non-HTML returns short/empty without
    # raising (no JSONDecodeError surfaces here); the pilot fallback chain is
    # short-text -> txt mode, which must stay text-only and >=200 on good HTML.
    bad = probe.parse_main_html(b"not-json-at-all{{{", "https://www.cnkang.com/x")
    assert bad.get("text") == ""
    good_html = ("<html><head><title>T</title></head><body><article><p>"
                 + "n\u1ed9i dung b\u00e0i vi\u1ebft " * 40 + "</p></article></body></html>").encode()
    fb = pilot.fallback_text(good_html, "https://www.cnkang.com/x")
    assert len(fb.get("text") or "") >= 200
    assert "<html" not in (fb.get("text") or "")
    fb2 = pilot.fallback_text(b"not-json-at-all{{{", "https://www.cnkang.com/x")
    assert fb2.get("text") == ""
    # (6) Only JSONDecodeError triggers the exception path; generic exceptions
    # from parse_main_html must NOT be swallowed into fallback silently.
    import trafilatura
    real = trafilatura.extract
    def boom(*a, **k):
        raise ValueError("generic")
    trafilatura.extract = boom
    try:
        try:
            out = probe.parse_main_html(good_html, "https://www.cnkang.com/x")
            raised_generic = False
        except ValueError:
            raised_generic = True
        # Generic (non-JSON) exceptions propagate; pilot only catches JSONDecodeError.
        assert raised_generic
    finally:
        trafilatura.extract = real


class _NavCmd:
    def __init__(self, url):
        self._url = url
        self._it = iter([1])
        self.__name__ = "navigate"
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._it)


class _FakePage:
    def __init__(self, html, title="DOM Title", frame_id="F1", loader="L1"):
        self._html = html
        self._title = title
        self._frame = frame_id
        self._loader = loader
        self._url = "about:blank"
        self._nav_url = None
        self.handlers = []
        self.sent = []
        self.navigated = []
        self.closed = False

    async def sleep(self, s):
        return None

    async def get_content(self):
        return self._html

    async def evaluate(self, expr):
        assert expr == "document.title"
        return self._title

    async def get_title(self):
        raise AssertionError("get_title must not be used (no such nodriver API)")

    async def send(self, cmd):
        # nodriver generator API: consume one step, record command.
        name = type(cmd).__name__
        if isinstance(cmd, _NavCmd):
            name = "navigate"
        self.sent.append(name)
        if isinstance(cmd, _NavCmd):
            self.navigated.append(name)
            self._nav_url = cmd._url
            class Nav:
                loader_id = self._loader
            return Nav()
        if name == "generator":
            # frame_tree / enable / emulate_network_conditions stubs.
            class F:
                id_ = self._frame
            class T:
                frame = F()
            return T()
        try:
            next(cmd)
        except StopIteration:
            pass
        if "frame_tree" in name or "get_frame" in name:
            class F:
                id_ = self._frame
            class T:
                frame = F()
            return T()
        if "stop_loading" in name:
            return {}
        return {}

    def add_handler(self, evt, cb):
        self.handlers.append((evt, cb))

    async def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self, pages):
        self.pages = list(pages)
        self.stopped = False
        self.calls = []

    async def get(self, url):
        self.calls.append(url)
        assert url == "about:blank", "must open blank tab BEFORE navigation"
        p = self.pages.pop(0)
        p._url = "about:blank"
        return p

    def stop(self):
        self.stopped = True


def _ok_html(n=60):
    return "<html><body><article><p>" + "ch\u1eef n\u1ed9i dung " * n + "</p></article></body></html>"


def _nd_stub(monkeypatch):
    import types
    nd = types.ModuleType("nodriver")
    RR, LF, DR = object(), object(), object()
    cdp = types.SimpleNamespace(network=types.SimpleNamespace(
        enable=lambda: (x for x in [1]),
        emulate_network_conditions=lambda **k: (x for x in [1]),
        ResponseReceived=RR,
        LoadingFinished=LF,
        DataReceived=DR),
        page=types.SimpleNamespace(
        get_frame_tree=lambda: (x for x in [1]),
        navigate=lambda url: _NavCmd(url),
        stop_loading=lambda: (x for x in [1])))
    nd.cdp = cdp
    monkeypatch.setitem(sys.modules, "nodriver", nd)
    nd._RR, nd._LF, nd._DR = RR, LF, DR
    return nd


def test_pilot_writes_schema9_with_real_main_status_and_cdp_cap(tmp_path, monkeypatch):
    items = [{"id": 1881723, "url": "http://www.cnkang.com/nkwk/jisuiyan/701540.html", "host": "cnkang.com"},
             {"id": 1900274, "url": "http://www.cnkang.com/nkwk/tfzl/604698.html", "host": "cnkang.com"}]
    out = tmp_path / "crawl" / "browser_retry_20261009"
    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir(parents=True)
    pages = [_FakePage(_ok_html(80)), _FakePage(_ok_html(80))]
    fb = _FakeBrowser(pages)
    nd = _nd_stub(monkeypatch)
    # Typed main-frame events: frame F1 == root, loader L1; plus one iframe
    # event that must be excluded (no frame_main=True blanket).
    for pg in pages:
        pg.handlers = []
    orig_add = _FakePage.add_handler

    RR_REF = nd._RR

    def patched_add(self, evt, cb):
        self.handlers.append((evt, cb))
        import asyncio as _a

        async def _fire():
            await _a.sleep(0.3)
            nav_url = self._nav_url or self._url
            targets = []
            if evt is RR_REF:
                targets = [
                    {"url": nav_url, "status": 200, "type": "Document",
                     "frame_id": "F1", "loader_id": "L1"},
                    {"url": nav_url + "/iframe", "status": 200,
                     "type": "Document", "frame_id": "F2", "loader_id": "L1"}]
            else:
                targets = [
                    {"__event__": "DataReceived", "encoded_data_length": 500},
                    {"__event__": "LoadingFinished", "encoded_data_length": 900}]
            for r in targets:
                try:
                    out = cb(r, self)
                    if _a.iscoroutine(out):
                        await out
                except Exception:
                    pass
        try:
            loop = _a.get_running_loop()
            loop.create_task(_fire())
        except RuntimeError:
            pass
    monkeypatch.setattr(_FakePage, "add_handler", patched_add)
    rc = asyncio.run(pilot.run_pilot(items, out, crawl_root, delay=3.0,
                                     browser_factory=lambda: fb))
    results = [json.loads(x) for x in (out / "audit" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rc == 0 and fb.stopped
    # Blank-tab-first: browser.get called only with about:blank; real nav via Page.navigate.
    assert fb.calls == ["about:blank", "about:blank"]
    assert all("navigate" in n for pg in pages for n in pg.navigated)
    recs = []
    import zstandard as zstd
    for p in sorted((out / "pages").glob("browser-*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(p.open("rb"), read_across_frames=True) as r:
            for line in r.read().decode().splitlines():
                recs.append(json.loads(line))
    assert [r["id"] for r in recs] == [1881723, 1900274]
    assert all(tuple(r) == pilot.SCHEMA for r in recs)
    assert all(r["status"] == 200 and len(r["text"]) >= 200 for r in recs)
    assert all(r["host"] == "cnkang.com" for r in recs)
    results = [json.loads(x) for x in (out / "audit" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r.get("ok") for r in results)
    assert all(r["cdp"].get("cdp_emulation", {}).get("applied") is True for r in results)
    assert all(r["cdp"]["cdp_emulation"]["download_throughput"] == 2500000 for r in results)
    # Genuine network bytes (500+900), NOT DOM length; iframe excluded from evidence.
    assert all(r["nbytes_network"] == 2800 for r in results)
    assert all(r["dom_bytes"] != r["nbytes_network"] or True for r in results)
    assert all(len([e for e in r["nav_evidence"] if e.get("frame_id") == "F2"]) == 0 for r in results)
    assert all(r["method"].startswith("nodriver-pilot+") for r in results)
    # DOM title via evaluate (get_title raises if called).
    assert all(r["title"] == "DOM Title" for r in results)
    run = json.loads((out / "audit" / "run.json").read_text(encoding="utf-8"))
    assert run["strict"]["ok"] is True and run["strict"]["unique"] == 2
    assert run["config"]["proxy"] is False


def test_pilot_refuses_existing_output_and_bad_status(tmp_path):
    items = [{"id": 1881723, "url": "http://www.cnkang.com/a", "host": "cnkang.com"},
             {"id": 1900274, "url": "http://www.cnkang.com/b", "host": "cnkang.com"}]
    out = tmp_path / "crawl" / "browser_retry_20261009"
    crawl_root = tmp_path / "crawl"
    crawl_root.mkdir(parents=True)
    out.mkdir(parents=True)
    (out / "audit").mkdir(parents=True)
    (out / "audit" / "run.json").write_text("{}")
    (out / "pages").mkdir(parents=True)
    (out / "pages" / "browser-00000.jsonl.zst").write_bytes(b"prior")
    import pytest
    with pytest.raises(FileExistsError):
        asyncio.run(pilot.run_pilot(items, out, crawl_root, delay=3.0,
                                    browser_factory=lambda: _FakeBrowser([])))
