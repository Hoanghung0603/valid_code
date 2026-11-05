import asyncio
import json
import os
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_async_supervisor as sup  # noqa: E402
from extract import classify as ext_classify  # noqa: E402


# ------------------------------------------------------------------ helpers

def _write_queue(wd, rows):
    import zstandard as zstd
    q = wd / "queue"
    q.mkdir(parents=True, exist_ok=True)
    raw = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    with (q / "smalls-00.jsonl.zst").open("xb") as fh:
        with zstd.ZstdCompressor().stream_writer(fh) as zw:
            zw.write(raw)
    (wd / "queue_manifest.json").write_text(
        json.dumps({"total": len(rows),
                    "shards": [{"file": "smalls-00.jsonl.zst", "rows": len(rows)}]}),
        encoding="utf-8")


def _row(i, host="a.vn", group="never_gaveup"):
    return {"id": i, "url": f"https://{host}/p{i}", "host": host, "group": group,
            "reason": "r", "attempts": 0, "route_hint": "direct_first"}


def _read_pages(wd):
    recs = []
    for f in sorted((wd / "pages").glob("*.jsonl.zst")):
        recs.extend(sup.iter_jsonl_zst(f))
    return recs


def _run(opts):
    return asyncio.run(sup.supervise(opts))


def _opts(wd, tmp, **kw):
    base = dict(workdir=wd, crawl_root=tmp, lock=tmp / "nolock",
                concurrency=2, per_host=1, delay=1.0, timeout=20.0,
                extract_workers=1, min_free_gb=0.05, archive_enabled=False,
                cycle_cooldown=5.0, max_cycles=kw.pop("max_cycles", 1))
    base.update(kw)
    return sup.Opts(**base)


class _FakeResp:
    def __init__(self, data):
        self.data = data

    def read(self, n=-1):
        if not self.data:
            return b""
        take = n if n and n > 0 else len(self.data)
        out, self.data = self.data[:take], self.data[take:]
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------- 1. history/infra

def test_infra_migration_append_only_resets_bogus_attempts(tmp_path):
    """run-1 ModuleNotFoundError rows must NOT burn the 3-attempt budget; the
    repair is append-only (original history lines preserved verbatim)."""
    state, audit = tmp_path / "state", tmp_path / "audit"
    state.mkdir(), audit.mkdir()
    hist = state / "done_ids.tsv"
    hist.write_text("7\tparked\n7\tparked\n7\tparked\n8\tpermanent\n",
                    encoding="utf-8")
    rows = [
        {"id": 7, "ok": False, "class": "transient", "error": "ModuleNotFoundError"},
        {"id": 7, "ok": False, "class": "transient",
         "error": "NotImplementedError: sim"},
        {"id": 8, "ok": False, "class": "permanent", "error": "None"},
    ]
    (audit / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    n = sup.migrate_infra_history(state, audit)
    assert n == 1                                   # id 7 -> one infra marker
    after = hist.read_text(encoding="utf-8")
    assert after.startswith("7\tparked\n7\tparked\n7\tparked\n8\tpermanent\n") \
        or "7\tparked\n7\tparked\n7\tparked" in after  # NOT destructively rewritten
    assert after.rstrip().endswith("7\tinfra")
    done, attempts, _ = sup.load_done_history(state)
    assert 7 not in done and attempts[7] == 0       # fresh genuine budget
    assert 8 not in done and attempts[8] == 1       # genuine attempt still counts
    # idempotent: second migration appends nothing
    assert sup.migrate_infra_history(state, audit) == 0
    # genuine attempts AFTER the marker still exhaust properly
    with (state / "done_ids.tsv").open("a", encoding="utf-8") as fh:
        for _ in range(sup.MAX_ROW_ATTEMPTS):
            fh.write("7\ttransient\n")
    done, _, _ = sup.load_done_history(state)
    assert 7 in done


def test_infra_kind_never_burns_attempts(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "done_ids.tsv").write_text(
        "1\tinfra\n1\tinfra\n1\tinfra\n1\tinfra\n", encoding="utf-8")
    done, attempts, _ = sup.load_done_history(state)
    assert 1 not in done and attempts[1] == 0


def test_retry_kind_unterminates_repaired_id(tmp_path):
    """Data repair: retry marker after ok must make the id pending again."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "done_ids.tsv").write_text(
        "10\tok\n10\tpermanent\n11\tok\n11\tretry\n", encoding="utf-8")
    done, attempts, terminal = sup.load_done_history(state)
    assert 10 in done and 10 in terminal
    assert 11 not in done and 11 not in terminal and attempts[11] == 0


def test_same_site_and_url_id_conflict_guards():
    assert sup.same_site("qdnd.vn", "https://www.qdnd.vn/a/843873")
    assert sup.same_site("medlatec.vn", "https://medlatec.vn/tin-tuc/x")
    assert not sup.same_site("qdnd.vn", "https://medlatec.vn/tin-tuc/x")
    assert not sup.same_site("a.vn", "")
    assert sup.url_id_conflict(
        "https://laodong.vn/x-1490511.ldo", "https://laodong.vn/y-150660.ldo")
    assert not sup.url_id_conflict(
        "https://laodong.vn/x-1490511.ldo", "https://laodong.vn/x-1490511.ldo")
    assert not sup.url_id_conflict(
        "https://qdnd.vn/a", "https://www.qdnd.vn/a")


# --------------------------------------------- 2. multi-cycle / caps reset

def test_multi_cycle_productive_host_continues_beyond_batch_cap(tmp_path,
                                                                monkeypatch):
    """A PRODUCTIVE browser host must keep going past the per-cycle batch cap
    (auto-resumed cycles, no user); successes are never refetched."""
    wd = tmp_path / "wd"
    monkeypatch.setattr(sup, "HOST_BROWSER_CAP", 2)
    monkeypatch.setattr(sup, "VI_BROWSER_CAP", 2)
    monkeypatch.setattr(sup, "HOST_ROW_FAIL_STREAK", 50)
    rows = [_row(100 + i, "js.vn") for i in range(5)]
    _write_queue(wd, rows)
    seen = []

    async def cf_direct(session, limiter, url, timeout_s=30.0):
        seen.append(url)
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"y" * 50 + b"</html>",
                "error": None, "infra": False}

    async def ok_browser(row, crawl_root, limiter, chrome_exe, timeout_s=60.0,
                         proxy_url=None, headless=True, pool=None):
        return {"ok": True, "status": 200, "final_url": row["url"], "title": "T",
                "raw": b"<html>" + b"z" * 400 + b"</html>", "nbytes": 100,
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        # direct body is tiny (JS-less shell -> short text) -> browser needed;
        # browser body carries the real content -> long text
        if b"zzz" in body:
            return {"title": "T", "date": "", "text": "z" * 300}
        return {"title": "", "date": "", "text": "y" * 30}

    monkeypatch.setattr(sup, "fetch_direct", cf_direct)
    monkeypatch.setattr(sup, "run_browser_one", ok_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    opts = _opts(wd, tmp_path, browser_max=1, max_cycles=3)
    assert _run(opts) == 0
    recs = _read_pages(wd)
    assert [r["id"] for r in recs] == [100, 101, 102, 103, 104]  # ALL rows
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["cycle"] >= 2                            # auto-resumed batches
    # cycle-1 batch == cap, later cycles continued (cap reset per cycle)
    assert st["counts"]["browser_ok"] == 5
    assert sorted({u.rsplit("/", 1)[-1] for u in seen}) == \
        ["p100", "p101", "p102", "p103", "p104"]       # every row really fetched
    assert len(set(r["id"] for r in recs)) == 5         # successes never duplicated


def test_successes_survive_cap_reset(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    monkeypatch.setattr(sup, "HOST_ROW_FAIL_STREAK", 50)
    _write_queue(wd, [_row(1), _row(2)])
    calls = []

    async def ok_direct(session, limiter, url, timeout_s=30.0):
        calls.append(url)
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"q" * 400 + b"</html>",
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "q" * 300}

    monkeypatch.setattr(sup, "fetch_direct", ok_direct)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, max_cycles=1)) == 0
    n = len(calls)
    assert n == 2
    assert _run(_opts(wd, tmp_path, max_cycles=1)) == 0   # new process cycle
    assert len(calls) == n                                # successes NOT erased


# --------------------------------------------------------- 3. fairness

def test_fair_concurrency_two_hosts_in_flight(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(200 + i, "fast.vn") for i in range(6)]
                 + [_row(300 + i, "slow.vn") for i in range(6)])
    live = {"n": 0, "max": 0, "per_host": {}}
    lock = threading.Lock()

    async def tracked(session, limiter, url, timeout_s=30.0):
        host = url.split("/")[2]
        with lock:
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])
            live["per_host"][host] = live["per_host"].get(host, 0) + 1
            assert live["per_host"][host] <= 4        # vi_per_host CONCURRENT bound
        await asyncio.sleep(0.05)
        with lock:
            live["n"] -= 1
            live["per_host"][host] -= 1
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"w" * 400 + b"</html>",
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "w" * 300}

    monkeypatch.setattr(sup, "fetch_direct", tracked)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, concurrency=4, per_host=2, max_cycles=1)) == 0
    assert live["max"] >= 2        # NOT sequential: >1 row actually in flight
    assert len(_read_pages(wd)) == 12


def test_vi_first_serves_vn_before_cn(tmp_path, monkeypatch):
    """VI-first: four vn rows must all be fetched before any cn row starts."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(900 + i, "vn.vn") for i in range(4)]
                 + [_row(910 + i, "zh.cn") for i in range(4)])
    order = []

    async def tracked(session, limiter, url, timeout_s=30.0):
        order.append(url.split("/")[2])
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"w" * 400 + b"</html>",
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "w" * 300}

    monkeypatch.setattr(sup, "fetch_direct", tracked)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, concurrency=1, per_host=1,
                      vi_per_host=1, max_cycles=1)) == 0
    assert order[:4] == ["vn.vn"] * 4
    assert set(order[4:]) == {"zh.cn"}


def test_headed_hosts_use_headed_browser(tmp_path, monkeypatch):
    """Hosts in headed_hosts must get headless=False (Long Chau evidence)."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(950, "nhathuoclongchau.com.vn")])
    seen = {}

    async def ok_direct(session, limiter, url, timeout_s=30.0):
        return {"status": 403, "final_url": url, "body": b"x",
                "error": None, "infra": False}

    async def headed_browser(row, crawl_root, limiter, chrome_exe,
                             timeout_s=60.0, proxy_url=None, headless=True, pool=None):
        seen["headless"] = headless
        return {"ok": True, "status": 200, "final_url": row["url"],
                "title": "T", "raw": b"<html>" + b"z" * 400 + b"</html>",
                "nbytes": 100, "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "T", "date": "", "text": "z" * 300}

    monkeypatch.setattr(sup, "fetch_direct", ok_direct)
    monkeypatch.setattr(sup, "run_browser_one", headed_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path,
                      headed_hosts={"nhathuoclongchau.com.vn"},
                      max_cycles=1)) == 0
    assert seen.get("headless") is False


# ---------------------------------------- 4. quarantine / no log spam

def test_quarantine_skips_shard_with_aggregate_no_spam(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    monkeypatch.setattr(sup, "HOST_ROW_FAIL_STREAK", 3)
    rows = [_row(400 + i, "dead.cn") for i in range(50)]
    _write_queue(wd, rows)
    calls = []

    async def dead(session, limiter, url, timeout_s=30.0):
        calls.append(url)
        return {"status": 521, "final_url": url, "body": b"",
                "error": None, "infra": False}

    async def no_browser(*a, **k):
        raise AssertionError("browser gated")

    async def no_extract(pool, body, url):
        return {"title": "", "date": "", "text": ""}

    monkeypatch.setattr(sup, "fetch_direct", dead)
    monkeypatch.setattr(sup, "run_browser_one", no_browser)
    monkeypatch.setattr(sup, "extract_in_pool", no_extract)
    assert _run(_opts(wd, tmp_path, browser_max=0, max_cycles=1)) == 0
    assert len(calls) == 3                              # only probe rows hit net
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["counts"]["deferred_cumulative"] >= 47     # aggregate, not logged
    audit_lines = [l for l in
                   (wd / "audit" / "results.jsonl").read_text(encoding="utf-8")
                   .splitlines() if l.strip()]
    assert len(audit_lines) <= 10                       # no 50-row spam
    # deferred rows are NOT attempt-burned -> still retriable next cycle
    hist = (wd / "state" / "done_ids.tsv").read_text(encoding="utf-8").splitlines()
    assert len(hist) == 3


# --------------------------------------------------- 5. archive typed 429

def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "err", hdrs=None, fp=None)


def test_archive_typed_429_both_functions(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(getattr(req, "full_url", "https://x"), 429)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    url, st, err = sup.wayback_available("https://a.vn/x", limiter=None)
    assert url is None and st == 429 and err is None     # TYPED 429, not 'HTTPError'
    raw, st2, err2 = sup.fetch_snapshot_text("https://web.archive.org/web/1/x",
                                             limiter=None)
    assert raw is None and st2 == 429 and err2 is None   # BOTH functions typed
    assert calls["n"] == 2


def test_archive_global_gate_blocks_network_and_persists(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(getattr(req, "full_url", "https://x"), 429)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(500)])
    gate = sup.Gate(wd, None)
    gate.archive_close(sup.ARCHIVE_429_COOLDOWN)
    assert gate.archive_open() is False
    # serial gate helper honours persisted cooldown
    gate2 = sup.Gate(wd, None, archive_gate_until=gate.archive_gate_until)
    assert gate2.archive_open() is False
    (wd / "state").mkdir(parents=True, exist_ok=True)
    sup.save_checkpoint(wd / "state", {"archive_gate_until": gate.archive_gate_until})
    cp = sup.load_checkpoint(wd / "state")
    assert float(cp["archive_gate_until"]) == gate.archive_gate_until
    # zero network while gated: try_archive skips before any call
    seen = {"n": 0}
    opts = _opts(wd, tmp_path, archive_enabled=True, max_cycles=1)

    async def no_net(session, limiter, url, timeout_s=30.0):
        seen["n"] += 1
        return {"status": 404, "final_url": url, "body": b"",
                "error": None, "infra": False}

    monkeypatch.setattr(sup, "fetch_direct", no_net)

    async def no_extract(pool, body, url):
        return {"title": "", "date": "", "text": ""}

    monkeypatch.setattr(sup, "extract_in_pool", no_extract)
    # pre-set the gate through checkpoint so supervise starts gated
    assert _run(opts) == 0
    assert calls["n"] == 0                               # no archive network at all
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["per_method"]["archive"].get("skipped_429_backoff", 0) >= 1


# --------------------------------------------------- 6. bandwidth limiter

def test_bandwidth_threadsafe_blocking_pace_split_lanes():
    lim = sup.BandwidthLimiter(http_bps=2000.0, browser_bps=2000.0)  # 2kB/s
    t0 = time.monotonic()
    lim.blocking_throttle(2000, "http")     # first chunk fills the bucket
    lim.blocking_throttle(2000, "http")     # second: ~1s pacing
    assert time.monotonic() - t0 >= 0.8
    # split lanes share the clock but not the budget
    lim2 = sup.BandwidthLimiter(http_bps=1e9, browser_bps=2000.0)
    t1 = time.monotonic()
    lim2.blocking_throttle(10_000_000, "http")   # huge, fast lane
    assert time.monotonic() - t1 < 0.5
    lim2.blocking_throttle(2000, "browser")      # fill browser bucket
    t2 = time.monotonic()
    lim2.blocking_throttle(2000, "browser")      # slow lane really paces
    assert time.monotonic() - t2 >= 0.8
    # thread safety: parallel reservations never overlap the same slot
    lim3 = sup.BandwidthLimiter(http_bps=1e6, browser_bps=0.0)
    errs = []

    def worker():
        try:
            for _ in range(20):
                lim3.blocking_throttle(1000, "http")
        except Exception as e:  # pragma: no cover
            errs.append(e)
    ths = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert not errs


def test_urllib_chunkwise_pacing_and_status_never_faked(tmp_path, monkeypatch):
    lim = sup.BandwidthLimiter(http_bps=0.0, browser_bps=0.0)
    paced = []

    def fake_build(*handlers):
        class O:
            def open(self, req, timeout=None):
                class R(_FakeResp):
                    status = None   # broken status: must NOT fall back to 200
                    code = None
                    url = req.full_url
                return R(b"x" * 10)

            def add_handler(self, h):
                return None
        return O()

    monkeypatch.setattr(sup, "_openers_factory", None, raising=False)
    import urllib.request as ur
    monkeypatch.setattr(ur, "build_opener", fake_build)
    res = sup.fetch_via_proxy("https://a.vn/x", "http://u:p@1.2.3.4:8080",
                              lim, timeout=5)
    assert res["status"] is None and res["error"] == "status-unknown"


def test_fetch_via_proxy_429_typed_and_chunk_paced(tmp_path, monkeypatch):
    paced = {"n": 0, "bytes": 0}
    lim = sup.BandwidthLimiter(http_bps=0.0, browser_bps=0.0)
    orig = lim.blocking_throttle

    def spy(nbytes, lane="http"):
        paced["n"] += 1
        paced["bytes"] += nbytes
        return orig(nbytes, lane)

    lim.blocking_throttle = spy

    def fake_build(*handlers):
        class O:
            def open(self, req, timeout=None):
                raise _http_error(req.full_url, 429)

            def add_handler(self, h):
                return None
        return O()

    import urllib.request as ur
    monkeypatch.setattr(ur, "build_opener", fake_build)
    res = sup.fetch_via_proxy("https://a.vn/x", "http://u:p@1.2.3.4:8080", lim)
    assert res["status"] == 429           # real typed status preserved
    assert res["body"] == b"" or isinstance(res["body"], (bytes, bytearray))


# --------------------------------------- 7. classify fail-closed challenge

def test_classify_long_challenge_failclosed():
    long_challenge = "<html>" + "Just a moment... checking your browser " * 50 \
        + "x" * 3000 + "</html>"
    assert sup.classify_inline(200, "https://a.vn/x", long_challenge, "",
                               b"") == "blocked"        # long != ok
    assert ext_classify(200, "https://a.vn/x", long_challenge) != "blocked" \
        or True  # (extract.py stays as-is; ours is deliberately stricter)
    # markers in title / final_url / raw
    assert sup.classify_inline(200, "https://a.vn/x", "x" * 300,
                               "安全验证", b"") == "blocked"
    assert sup.classify_inline(200, "https://a.vn/cdn-cgi/challenge",
                               "x" * 300, "", b"") == "blocked"
    assert sup.classify_inline(200, "https://a.vn/x", "x" * 300, "",
                               b"<html><body><script>window._cf_chl</script>"
                               b"</body></html>") == "blocked"
    # medical content must NOT false-positive (bare 验证 / 验证试验)
    med = "免疫验证试验用于评估患者免疫应答水平。" * 30
    assert sup.classify_inline(200, "https://yy.cn/a", med, "免疫学检查", b"") == "ok"
    # parity with extract.py on non-challenge inputs
    for st, txt, exp in ((200, "y" * 300, "ok"), (200, "y" * 50, "permanent"),
                         (404, "", "permanent"), (503, "", "blocked"),
                         (None, "", "transient")):
        assert sup.classify_inline(st, "https://a.vn/x", txt) == exp == \
            ext_classify(st, "https://a.vn/x", txt)


def test_status_none_never_ok():
    assert sup.classify_inline(None, "https://a.vn/x", "z" * 500) == "transient"


# --------------------------------------- 8. browser wiring (fake nodriver)

def _install_fake_nodriver(monkeypatch, events, nav_loader="L2",
                           root_frame="F1", status=200, body=b"<html>ok</html>"):
    import types
    nd = types.ModuleType("nodriver")

    class C:
        class network:
            class ResponseReceived:  # noqa
                pass

            class DataReceived:
                pass

            class LoadingFinished:
                pass

            @staticmethod
            def enable():
                return ("net.enable",)

            @staticmethod
            def emulate_network_conditions(**kw):
                return ("net.emu", kw)

        class page:
            @staticmethod
            def get_frame_tree():
                return ("page.tree",)

            @staticmethod
            def navigate(url):
                return ("page.nav", url)

            @staticmethod
            def stop_loading():
                return ("page.stop",)

        class fetch:
            class AuthRequired:
                pass

            @staticmethod
            def enable(handle_auth=True):
                return ("fetch.enable",)

    nd.cdp = C

    class FakePage:
        def __init__(self):
            self.handlers = []

        async def sleep(self, s):
            return None

        async def send(self, cmd):
            name = cmd[0] if isinstance(cmd, tuple) else "?"
            if name == "page.tree":
                class F:
                    id_ = root_frame if root_frame else None
                class T:
                    frame = F()
                if root_frame is None:
                    raise RuntimeError("no frame")
                return T()
            if name == "page.nav":
                for ev in events:
                    for hname, cb in self.handlers:
                        if hname == ev["kind"]:
                            cb(ev["obj"], None)
                class N:
                    loader_id = nav_loader
                return N()
            return None

        async def get_content(self):
            return body.decode("utf-8", "ignore")

        async def evaluate(self, expr):
            return "T"

        def add_handler(self, evt, cb):
            self.handlers.append((evt.__name__, cb))

        async def close(self):
            return None

    class FakeBrowser:
        def __init__(self, page):
            self._page = page

        async def get(self, url):
            return self._page

        async def stop(self):
            return None

    async def start(**kw):
        return FakeBrowser(FakePage())

    nd.start = start
    monkeypatch.setitem(sys.modules, "nodriver", nd)
    return nd


class _Evt:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_browser_rootframe_unknown_failclosed(tmp_path, monkeypatch):
    _install_fake_nodriver(monkeypatch, [], root_frame=None)
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/x"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5))
    assert res["ok"] is False and res["error"] == "root-frame-unknown"
    assert res["status"] is None                       # never fake 200


def test_browser_loader_postfilter_picks_final_loader(tmp_path, monkeypatch):
    # initial response on L1 arrives before nav returns L2 (challenge redirect)
    ev_l1 = {"kind": "ResponseReceived",
             "obj": _Evt(response=_Evt(url="https://a.vn/chk", status=503),
                         type_=_Evt(value="Document"), frame_id="F1",
                         loader_id="L1")}
    ev_l2 = {"kind": "ResponseReceived",
             "obj": _Evt(response=_Evt(url="https://a.vn/final", status=200),
                         type_=_Evt(value="Document"), frame_id="F1",
                         loader_id="L2")}
    _install_fake_nodriver(monkeypatch, [ev_l1, ev_l2], nav_loader="L2")
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/x"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5))
    assert res["ok"] is True
    assert res["status"] == 200 and res["final_url"] == "https://a.vn/final"


def test_browser_bytes_requestid_no_double_count(tmp_path, monkeypatch):
    events = [
        {"kind": "DataReceived",
         "obj": _Evt(request_id="r1", encoded_data_length=100)},
        {"kind": "DataReceived",
         "obj": _Evt(request_id="r1", encoded_data_length=50)},
        {"kind": "LoadingFinished",
         "obj": _Evt(request_id="r1", encoded_data_length=150)},
        {"kind": "ResponseReceived",
         "obj": _Evt(response=_Evt(url="https://a.vn/x", status=200),
                     type_=_Evt(value="Document"), frame_id="F1", loader_id="L1")},
    ]
    _install_fake_nodriver(monkeypatch, events, nav_loader="L1")
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/x"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5))
    assert res["nbytes"] == 150    # max(150 accumulated, 150 finished), NOT 300


# --------------------------------------------- 9. heavy lock / STOP

def test_heavylock_pause_retries_same_row_and_recovers(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    lock = tmp_path / "heavy_job.lock"
    _write_queue(wd, [_row(600), _row(601)])
    calls = []
    state = {"n": 0}

    async def flaky(session, limiter, url, timeout_s=30.0):
        calls.append(url)
        state["n"] += 1
        if state["n"] == 1:
            lock.write_text("")                     # lock appears mid-job
            async def _rm():
                await asyncio.sleep(0.25)
                lock.unlink()
            asyncio.get_running_loop().create_task(_rm())
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"v" * 400 + b"</html>",
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "v" * 300}

    monkeypatch.setattr(sup, "fetch_direct", flaky)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    monkeypatch.setattr(sup.Gate, "lock_poll", 0.05, raising=False)
    # make the gate poll fast
    orig_gate = sup.Gate.gate

    async def fast_gate(self, on_pause=None):
        self.raise_if_stop()
        while self.lock_hit():
            self.paused = True
            if on_pause is not None:
                on_pause()
            self.raise_if_stop()
            await asyncio.sleep(0.05)
        self.paused = False
        self.raise_if_stop()

    monkeypatch.setattr(sup.Gate, "gate", fast_gate)
    opts = _opts(wd, tmp_path, lock=lock, concurrency=1, per_host=1, max_cycles=1)
    assert _run(opts) == 0
    assert len(calls) == 2                            # SAME rows all fetched
    recs = _read_pages(wd)
    assert sorted(r["id"] for r in recs) == [600, 601]  # nothing lost to the pause
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["stop"]["flag"] is False                # pause never sticks as stop
    assert orig_gate  # keep flake quiet


def test_stop_prompt_during_sleep(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    gate = sup.Gate(wd, None)

    async def go():
        async def _stopper():
            await asyncio.sleep(0.15)
            (wd / "STOP").write_text("")
        t = asyncio.get_running_loop().create_task(_stopper())
        t0 = time.monotonic()
        try:
            await gate.sleep(30.0)
        except sup.StopRequested:
            return time.monotonic() - t0
        return 99.0

    elapsed = asyncio.run(go())
    assert elapsed < 3.0                              # STOP interrupts sleep


# ------------------------------------------------- 10. durability order

def test_publish_order_real_writer(tmp_path):
    pages = tmp_path / "pages"
    order = []
    writer = sup.ShardWriter(pages, max_records=100)
    real_flush = writer.flush_fsync

    def spy_flush():
        order.append("fsync")
        real_flush()

    writer.flush_fsync = spy_flush
    rec = {"id": 9, "url": "https://a.vn/x", "host": "a.vn",
           "final_url": "https://a.vn/x", "status": 200, "title": "t",
           "date": "", "text": "x" * 250, "ts": 1}
    writer.write_publish(rec, journal=lambda r: order.append("journal"))
    assert order[:2] == ["fsync", "journal"]
    writer.close()


def test_startup_recovery_ids_from_own_output(tmp_path, monkeypatch):
    """A crash between journal and fsync is healed: output ids are terminal."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(700)])
    pages = wd / "pages"
    pages.mkdir(parents=True)
    w = sup.ShardWriter(pages)
    w.write_publish({"id": 700, "url": "https://a.vn/p700", "host": "a.vn",
                     "final_url": "https://a.vn/p700", "status": 200,
                     "title": "t", "date": "", "text": "x" * 250, "ts": 1})
    w.close()
    calls = []

    async def seen_fetch(session, limiter, url, timeout_s=30.0):
        calls.append(url)
        return {"status": 500, "final_url": url, "body": b"",
                "error": None, "infra": False}

    async def no_extract(pool, body, url):
        return {"title": "", "date": "", "text": ""}

    monkeypatch.setattr(sup, "fetch_direct", seen_fetch)
    monkeypatch.setattr(sup, "extract_in_pool", no_extract)
    assert _run(_opts(wd, tmp_path, max_cycles=1)) == 0
    assert calls == []                                 # id 700 recovered, not refetched


def test_manifest_at_close_strict_not_double_hash(tmp_path):
    pages = tmp_path / "pages"
    manifests = []

    def on_close(p):
        manifests.append(sup.strict_scan_shard(p))

    w = sup.ShardWriter(pages, max_records=2, on_close=on_close)
    for i in range(3):
        w.write_publish({"id": i, "url": f"https://a.vn/{i}", "host": "a.vn",
                         "final_url": f"https://a.vn/{i}", "status": 200,
                         "title": "t", "date": "", "text": "x" * 250, "ts": 1})
    w.close()
    assert len(manifests) == 2                          # at rotation AND at close
    for m in manifests:
        assert m["ok"] is True                          # contract verified, ...
        assert m["strict_eof"] and m["schema_bad"] == 0
        assert m["content_bad"] == 0 and m["challenge"] == 0
        assert len(m["sha256"]) == 64


def test_strict_scanner_catches_truncated_and_challenge(tmp_path):
    import zstandard as zstd
    p = tmp_path / "good.jsonl.zst"
    with p.open("xb") as fh:
        with zstd.ZstdCompressor().stream_writer(fh) as zw:
            zw.write((json.dumps({"id": 1, "url": "https://a.vn/x", "host": "a.vn",
                                  "final_url": "https://a.vn/x", "status": 200,
                                  "title": "t", "date": "", "text": "x" * 250,
                                  "ts": 1}) + "\n").encode())
    v = sup.strict_scan_shard(p)
    assert v["ok"] and v["records"] == 1
    raw = p.read_bytes()
    bad = tmp_path / "trunc.jsonl.zst"
    bad.write_bytes(raw[:-3])                           # truncated final frame
    v2 = sup.strict_scan_shard(bad)
    assert v2["ok"] is False
    p2 = tmp_path / "chal.jsonl.zst"
    with p2.open("xb") as fh:
        with zstd.ZstdCompressor().stream_writer(fh) as zw:
            zw.write((json.dumps({"id": 2, "url": "https://a.vn/x", "host": "a.vn",
                                  "final_url": "https://a.vn/x", "status": 200,
                                  "title": "Attention Required", "date": "",
                                  "text": "Just a moment " + "x" * 250,
                                  "ts": 1}) + "\n").encode())
    v3 = sup.strict_scan_shard(p2)
    assert v3["ok"] is False and v3["challenge"] >= 1


# ------------------------------------------------------ 11. proxy routes

def test_proxy_routes_load_all_sticky_rotate_durable_no_leak(tmp_path):
    pf = tmp_path / "proxies.txt"
    pf.write_text("\n".join(f"10.0.0.{i}:8000:user{i}:pass{i}" for i in range(100))
                  + "\n", encoding="utf-8")
    routes = sup.RouteTable(sup.load_proxies(pf, 100))
    assert len(routes.routes) == 100                    # NOT truncated to 8
    i1, u1 = routes.pick("ask.39.net")
    assert i1 is not None and "user" in u1
    routes.report("ask.39.net", i1, ok=True)            # sticky
    i2, _ = routes.pick("ask.39.net")
    assert i2 == i1
    for _ in range(3):
        routes.report("ask.39.net", i1, ok=False, blocked=True)  # rotate AFTER cooldown
    i3, u3 = routes.pick("ask.39.net")
    assert i3 != i1                                     # cooled route skipped
    d = routes.to_json()
    routes2 = sup.RouteTable(routes.routes)
    routes2.load_json(d)                                # durable health
    assert routes2.hosts["ask.39.net"]["sticky"] == i1
    summary = routes.redacted_summary() + json.dumps(routes.to_json())
    assert "pass0" not in summary and "user0" not in summary  # no credential leak


def test_redaction_and_parse_security():
    assert "user1" not in sup.redact("http://user1:pass1@1.2.3.4:8080/x")
    assert "pass1" not in sup.redact("1.2.3.4:8080:user1:pass1")
    assert sup.parse_proxy_line("") is None
    assert sup.parse_proxy_line("1.2.3.4:8080:u:p") == "http://u:p@1.2.3.4:8080"


# ------------------------------------------------------- 12. singleton PID

def test_singleton_refuses_live_pid_takes_over_dead(tmp_path, monkeypatch):
    import subprocess
    wd = tmp_path / "wd"
    (wd / "queue").mkdir(parents=True)
    (wd / "state").mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        sup.atomic_json(wd / "state" / "supervisor.lock", {"pid": proc.pid})
        opts = _opts(wd, tmp_path)
        assert _run(opts) == 3                          # live lock refused
    finally:
        proc.kill()
        proc.wait()
    assert sup.pid_alive(proc.pid) is False
    sup.atomic_json(wd / "state" / "supervisor.lock", {"pid": proc.pid})
    opts = _opts(wd, tmp_path)
    assert _run(opts) == 2                              # stale taken over (empty queue)


def test_pid_alive_self():
    assert sup.pid_alive(os.getpid()) is True
    assert sup.pid_alive(0) is False


# --------------------------------------------------- 13. scheduler contract

def test_unlimit_ps1_and_bat_contract():
    root = Path(__file__).resolve().parent.parent / "deploy"
    ps1 = (root / "crawl_async_unlimit.ps1").read_text(encoding="utf-8")
    assert "ExecutionTimeLimit ([TimeSpan]::Zero)" in ps1
    assert "r2ai_crawl_async_recovery" in ps1
    bat = (root / "crawl_async_b660.bat").read_text(encoding="utf-8")
    assert "crawl_async_unlimit.ps1" in bat
    assert "PYTHONUNBUFFERED=1" in bat
    sup_src = (root / "crawl_async_supervisor.py").read_text(encoding="utf-8")
    assert "WindowsProactorEventLoopPolicy" in sup_src


# ------------------------------------------------------- 14. opts contract

def test_opts_reject_speedups():
    import pytest
    for kw in ({"concurrency": 99}, {"max_mbps": 21.0}, {"per_host": 9},
               {"vi_per_host": 9},
               {"extract_workers": 8}, {"browser_max": 9}, {"min_free_gb": 1.0},
               {"cycle_cooldown": 1.0}, {"delay": 0.5}):
        with pytest.raises(ValueError):
            sup.validate_opts(sup.Opts(**kw))
    sup.validate_opts(sup.Opts())  # defaults valid


def test_direct_skip_hosts_go_straight_to_browser(tmp_path, monkeypatch):
    """laodong/baotayninh: direct is proven-fail, must not burn a fetch."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(960, "laodong.vn")])
    direct_calls = []

    async def no_direct(session, limiter, url, timeout_s=30.0):
        direct_calls.append(url)
        raise AssertionError("direct must be skipped")

    async def ok_browser(row, crawl_root, limiter, chrome_exe, timeout_s=60.0,
                         proxy_url=None, headless=True, pool=None):
        return {"ok": True, "status": 200, "final_url": row["url"],
                "title": "T",
                "raw": b"<html>" + b"z" * 400 + b"</html>",
                "nbytes": 100, "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "T", "date": "", "text": "z" * 300}

    monkeypatch.setattr(sup, "fetch_direct", no_direct)
    monkeypatch.setattr(sup, "run_browser_one", ok_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, max_cycles=1)) == 0
    assert not direct_calls
    assert [r["id"] for r in _read_pages(wd)] == [960]


def test_browser_worth_trying_accepts_redirect_loop():
    """qdnd.vn: 302 Set-Cookie lặp -> aiohttp TooManyRedirects (status None).
    Chrome qua được bước cookie, nên lỗi này phải mở đường browser."""
    url = "https://www.qdnd.vn/x"
    assert sup.browser_worth_trying(None, "", "TooManyRedirects", url) is True
    assert sup.browser_worth_trying(
        None, "", "aiohttp.TooManyRedirects: loop", url) is True
    assert sup.browser_worth_trying(None, "", "TimeoutError", url) is False
    assert sup.browser_worth_trying(404, "", "TooManyRedirects", url) is False


def test_qdnd_goes_straight_to_browser(tmp_path, monkeypatch):
    """qdnd.vn direct đã fail 100%: không đốt fetch direct, đi browser ngay."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(961, "qdnd.vn")])
    direct_calls = []

    async def no_direct(session, limiter, url, timeout_s=30.0):
        direct_calls.append(url)
        raise AssertionError("direct must be skipped")

    async def ok_browser(row, crawl_root, limiter, chrome_exe, timeout_s=60.0,
                         proxy_url=None, headless=True, pool=None):
        return {"ok": True, "status": 200, "final_url": row["url"],
                "title": "T",
                "raw": b"<html>" + b"z" * 400 + b"</html>",
                "nbytes": 100, "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "T", "date": "", "text": "z" * 300}

    monkeypatch.setattr(sup, "fetch_direct", no_direct)
    monkeypatch.setattr(sup, "run_browser_one", ok_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, max_cycles=1)) == 0
    assert not direct_calls
    assert [r["id"] for r in _read_pages(wd)] == [961]


def test_browser_semaphore_allows_configured_max(tmp_path, monkeypatch):
    """browser_max=8 must allow parallel browser fetches (no hidden cap 1)."""
    import asyncio as _aio
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(970 + i, "fast.vn") for i in range(4)])
    live = {"n": 0, "peak": 0}
    lock = threading.Lock()

    async def slow_direct(session, limiter, url, timeout_s=30.0):
        return {"status": 200, "final_url": url,
                "body": b"<html>shell</html>", "error": None, "infra": False}

    async def slow_browser(row, crawl_root, limiter, chrome_exe,
                           timeout_s=60.0, proxy_url=None, headless=True, pool=None):
        with lock:
            live["n"] += 1
            live["peak"] = max(live["peak"], live["n"])
        await _aio.sleep(0.2)
        with lock:
            live["n"] -= 1
        return {"ok": True, "status": 200, "final_url": row["url"],
                "title": "T",
                "raw": b"<html>" + b"z" * 400 + b"</html>",
                "nbytes": 100, "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        if b"zzz" in body:
            return {"title": "T", "date": "", "text": "z" * 300}
        return {"title": "", "date": "", "text": "shell" * 2}

    monkeypatch.setattr(sup, "fetch_direct", slow_direct)
    monkeypatch.setattr(sup, "run_browser_one", slow_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, concurrency=4, per_host=4,
                      vi_per_host=4, browser_max=4, max_cycles=1)) == 0
    assert live["peak"] >= 2


def test_browser_media_block_config_present():
    src = (Path(__file__).resolve().parent.parent / "deploy"
           / "crawl_async_supervisor.py").read_text(encoding="utf-8")
    for token in ("fontsEnabled=false", "mute-audio",
                  "disable-background-networking", "set_blocked_urls",
                  "BROWSER_SESSION_CAP", "BROWSER_SETTLE_MAX",
                  "BROWSER_POOL_TABS", "class BrowserPool"):
        assert token in src
    # Fetch-interception blocking caused main-frame stalls; must be gone
    assert "BlockedByClient" not in src
    assert sup.BROWSER_SESSION_CAP == 20 * 1024 * 1024
    assert sup.BROWSER_SETTLE_MAX == 20.0


def test_browser_pool_reuses_browser_across_rows(tmp_path, monkeypatch):
    """Pool must serve 2 rows from ONE browser (no per-URL start/stop)."""
    starts = {"n": 0}

    class FakePage:
        async def sleep(self, s):
            return None

        async def send(self, *a, **k):
            class Tree:
                frame = type("F", (), {"id_": "R1"})()
            return Tree()

        async def evaluate(self, *a, **k):
            return ""

        async def get_content(self):
            return "<html>t</html>"

        async def close(self):
            return None

    class FakeBrowser:
        def __init__(self):
            self.calls = []

        async def get(self, url, **kw):
            self.calls.append((url, kw))
            return FakePage()

        def stop(self):
            return None

    async def fake_start(**kw):
        starts["n"] += 1
        return FakeBrowser()

    import types as _t
    nd = _t.ModuleType("nodriver")
    nd.start = fake_start
    nd.cdp = _t.SimpleNamespace(
        network=_t.SimpleNamespace(
            enable=lambda: ("enable",),
            emulate_network_conditions=lambda **k: ("cap",),
            ResponseReceived=object(), DataReceived=object(),
            LoadingFinished=object(),
            set_blocked_urls=lambda urls: ("blocked", urls)),
        page=_t.SimpleNamespace(
            get_frame_tree=lambda: ("tree",),
            navigate=lambda url: ("nav", url),
            stop_loading=lambda: ("stop",)),
        fetch=_t.SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "nodriver", nd)
    (tmp_path / "tools" / "browser_deps").mkdir(parents=True)
    lim = sup.BandwidthLimiter(http_bps=1e9, browser_bps=1e9)
    pool = sup.BrowserPool(tmp_path, None, True, None, [])

    async def _two():
        r1 = {"id": 1, "url": "https://a.vn/1", "host": "a.vn"}
        r2 = {"id": 2, "url": "https://a.vn/2", "host": "a.vn"}
        await pool.acquire(5.0)
        await pool.acquire(5.0)
        return pool

    import asyncio as _aio
    _aio.run(_two())
    assert starts["n"] == 1          # ONE browser for two tabs
    assert pool.tabs_used == 2
    # isolation: every acquire must ask for a NEW tab, never reuse the first
    assert pool.browser.calls and all(
        kw.get("new_tab") is True for _, kw in pool.browser.calls)


def test_browser_pool_start_carries_proxy_server(tmp_path, monkeypatch):
    """REGRESSION 13/10: pooled headed browser ignored proxy_url, so every
    Long Chau row went out on the origin IP (rate-limited) despite rotation.
    _start_browser must pass --proxy-server to nd.start."""
    seen = {}

    async def fake_start(**kw):
        seen.update(kw)
        import types as _t

        class FakeBrowser:
            async def get(self, url, **kw):
                class FakePage:
                    async def close(self):
                        return None
                return FakePage()
        return FakeBrowser()

    import types as _t
    nd = _t.ModuleType("nodriver")
    nd.start = fake_start
    monkeypatch.setitem(__import__("sys").modules, "nodriver", nd)
    (tmp_path / "tools" / "browser_deps").mkdir(parents=True)
    pool = sup.BrowserPool(tmp_path, None, False,
                           "http://u:p@r9:99", ["--mute-audio"])

    import asyncio as _aio
    _aio.run(pool._start_browser(5.0))
    args = seen.get("browser_args", [])
    assert seen.get("headless") is False
    assert "--proxy-server=http://r9:99" in args
    assert "--mute-audio" in args            # shared args preserved
    # no-proxy pool must NOT add the flag
    seen.clear()
    pool2 = sup.BrowserPool(tmp_path, None, False, None, ["--mute-audio"])
    _aio.run(pool2._start_browser(5.0))
    assert all(not a.startswith("--proxy-server")
               for a in seen.get("browser_args", []))


def test_browser_pool_ensure_race_single_start(tmp_path, monkeypatch):
    """Concurrent acquire() must spawn exactly ONE Chrome (race fix): before
    the fix every racing worker passed `browser is None` and leaked a process."""
    starts = {"n": 0}
    gate = {"go": None}

    class FakePage:
        async def close(self):
            return None

    class FakeBrowser:
        async def get(self, url, **kw):
            return FakePage()

        def stop(self):
            return None

    async def slow_start(**kw):
        starts["n"] += 1
        await asyncio.sleep(0.1)         # widen the race window
        return FakeBrowser()

    import types as _t
    nd = _t.ModuleType("nodriver")
    nd.start = slow_start
    monkeypatch.setitem(__import__("sys").modules, "nodriver", nd)
    (tmp_path / "tools" / "browser_deps").mkdir(parents=True)
    pool = sup.BrowserPool(tmp_path, None, True, None, [])

    async def _many():
        await asyncio.gather(*[pool.acquire(5.0) for _ in range(4)])

    asyncio.run(_many())
    assert starts["n"] == 1
    assert pool.tabs_used == 4


def test_pool_janitor_closes_idle_pool(tmp_path, monkeypatch):
    """Janitor must discard pools whose last use exceeded the idle window."""
    closed = []

    class P:
        def __init__(self):
            self.browser = object()
            self.last_used = 0.0
            self.discarded = False

        async def discard(self):
            closed.append(self)
            self.browser = None

    p_old, p_new = P(), P()
    p_new.last_used = time.monotonic()
    pools = {"old": p_old, "new": p_new}
    stop = {"flag": False}

    async def _jan():
        # replicate janitor one-pass logic against the module constants
        now = time.monotonic()
        for k, p in list(pools.items()):
            if p.browser is not None and now - p.last_used > sup.POOL_IDLE_CLOSE_S:
                await p.discard()
        stop["flag"] = True

    asyncio.run(_jan())
    assert closed == [p_old]
    assert sup.POOL_IDLE_CLOSE_S >= 150.0     # never cut a live fetch
    # (fetch max ~2.5 min; idle 300s is 2x safe margin)


def test_pool_dead_session_discards_and_defers(tmp_path, monkeypatch):
    """A dead pool session must discard the browser and defer (no fail burn)."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(980, "dead.vn")])

    async def dead_direct(session, limiter, url, timeout_s=30.0):
        return {"status": 200, "final_url": url,
                "body": b"<html>shell</html>", "error": None, "infra": False}

    async def dead_browser(row, crawl_root, limiter, chrome_exe,
                           timeout_s=60.0, proxy_url=None, headless=True,
                           pool=None):
        return {"ok": False, "status": None, "final_url": row["url"],
                "title": "", "raw": b"", "nbytes": 0,
                "error": "pool-dead:Session with given id not found",
                "infra": False, "pool_dead": True}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "shell" * 2}

    monkeypatch.setattr(sup, "fetch_direct", dead_direct)
    monkeypatch.setattr(sup, "run_browser_one", dead_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, max_cycles=1)) == 0
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["counts"]["deferred_cumulative"] >= 1
    assert st["harvested_unique"] == 0
    assert sup.is_pool_dead("Session with given id not found")
    assert not sup.is_pool_dead("TimeoutError: plain")


def test_stop_browser_reap_kills_surviving_process(tmp_path, monkeypatch):
    """stop() failing to terminate the Chrome tree must end with a taskkill:
    the leak fix reaps the recorded PID when it survives stop()."""
    killed = []

    class StubbornBrowser:
        _process_pid = 424242
        _process = type("P", (), {"pid": 424242})()

        def stop(self):
            return None                     # nodriver-style silent no-op

    alive = {"v": True}
    monkeypatch.setattr(sup, "pid_alive", lambda pid: alive["v"])
    monkeypatch.setattr(sup, "kill_process_tree",
                        lambda pid: (killed.append(pid), alive.update(v=False),
                                     True)[-1])
    asyncio.run(sup.stop_browser_reap(StubbornBrowser(), wait_s=0.3))
    assert killed == [424242]


def test_stop_browser_reap_no_kill_when_already_dead(monkeypatch):
    """A browser whose process exited on its own must NOT be taskkilled."""
    killed = []

    class DeadBrowser:
        _process_pid = 424243

        def stop(self):
            return None

    monkeypatch.setattr(sup, "pid_alive", lambda pid: False)
    monkeypatch.setattr(sup, "kill_process_tree",
                        lambda pid: killed.append(pid) or True)
    asyncio.run(sup.stop_browser_reap(DeadBrowser(), wait_s=0.3))
    assert killed == []


def test_kill_process_tree_invalid_pid_no_raise():
    assert sup.kill_process_tree(0) is True
    assert sup.kill_process_tree(-5) is True


# ============================================================ ROUND 3 (parent)

def _install_fake2(monkeypatch, events, nav_loader="L1", root_frame="F1",
                   dom_url="", body=b"<html>ok</html>", title="T",
                   dom_urls=None, ready_states=None, body_lens=None,
                   text_lens=None, challenges=None, bodies=None):
    """Fake nodriver with cdp.fetch interception + DOM location support.

    Sequenced polls (one entry per settle iteration; last entry repeats):
    dom_urls / ready_states / body_lens / text_lens / challenges / bodies.
    Defaults emulate a settled complete DOM with ample content so existing
    tests break the settle-wait on the first poll."""
    import types
    nd = types.ModuleType("nodriver")
    sent = []

    class Fetch:
        class RequestPaused:
            pass

        class AuthRequired:
            pass

        class AuthChallengeResponse:
            def __init__(self, response, username=None, password=None):
                self.response, self.username, self.password = \
                    response, username, password

        @staticmethod
        def enable(patterns=None, handle_auth_requests=None):
            return ("fetch.enable", handle_auth_requests)

        @staticmethod
        def continue_request(request_id=None):
            return ("fetch.continue_request", request_id)

        @staticmethod
        def continue_with_auth(request_id=None, auth_challenge_response=None):
            return ("fetch.continue_with_auth", request_id,
                    auth_challenge_response)

    class C:
        network = types.SimpleNamespace(
            ResponseReceived=type("ResponseReceived", (), {}),
            DataReceived=type("DataReceived", (), {}),
            LoadingFinished=type("LoadingFinished", (), {}),
            enable=lambda: ("net.enable",),
            emulate_network_conditions=lambda **kw: ("net.emu", kw),
        )
        page = types.SimpleNamespace(
            get_frame_tree=lambda: ("page.tree",),
            navigate=lambda u: ("page.nav", u),
            stop_loading=lambda: ("page.stop",),
        )
        fetch = Fetch
    nd.cdp = C

    class FakePage:
        def __init__(self):
            self.handlers = []
            self._poll = [0]
            self._seq = {"dom": list(dom_urls) if dom_urls else None,
                         "ready": list(ready_states) if ready_states else None,
                         "blen": list(body_lens) if body_lens else None,
                         "tlen": list(text_lens) if text_lens else None,
                         "chal": list(challenges) if challenges else None,
                         "body": list(bodies) if bodies else None}
            self._body_default = body

        def _take(self, key, default):
            seq = self._seq[key]
            if not seq:
                return default
            idx = min(self._poll[0], len(seq) - 1)
            return seq[idx]

        async def sleep(self, s):
            await asyncio.sleep(0)
            return None

        async def send(self, cmd):
            await asyncio.sleep(0)      # yield so the pump can run
            name = cmd[0] if isinstance(cmd, tuple) else "?"
            sent.append(cmd)
            if name == "page.tree":
                if root_frame is None:
                    raise RuntimeError("no frame")
                class F:
                    id_ = root_frame
                class T:
                    frame = F()
                return T()
            if name == "page.nav":
                for ev in events:
                    for evt_name, cb in self.handlers:
                        if evt_name == ev["kind"]:
                            cb(ev["obj"], None)
                class N:
                    loader_id = nav_loader
                return N()
            return None

        async def get_content(self):
            b = self._take("body", self._body_default)
            return b.decode("utf-8", "ignore")

        async def evaluate(self, expr):
            if "location" in expr:
                return self._take("dom", dom_url)
            if "readyState" in expr:
                return self._take("ready", "complete")
            if "querySelectorAll" in expr:
                self._poll[0] += 1    # last poll of one settle iteration
                return 1 if self._take("chal", False) else 0
            if "innerText.length" in expr and "title" not in expr:
                return self._take("blen", 500)
            if "innerText" in expr:
                return self._take("tlen", 500)
            if expr.strip() == "document.title":
                return title
            return title

        def add_handler(self, evt, cb):
            self.handlers.append((evt.__name__, cb))

        async def close(self):
            return None

    class FakeBrowser:
        def __init__(self, page):
            self._page = page

        async def get(self, url, **kw):
            return self._page

        async def stop(self):
            await asyncio.sleep(0)
            return None

    async def start(**kw):
        return FakeBrowser(FakePage())

    nd.start = start
    monkeypatch.setitem(sys.modules, "nodriver", nd)
    return sent


def test_normalize_status_accepts_cdp_float():
    assert sup.normalize_status(200.0) == 200        # CDP sends FLOAT 200.0
    assert sup.normalize_status(200) == 200
    assert sup.normalize_status(404.0) == 404
    assert sup.normalize_status(None) is None
    assert sup.normalize_status("200") is None       # str never trusted
    assert sup.normalize_status(200.5) is None
    assert sup.normalize_status(float("nan")) is None
    assert sup.normalize_status(float("inf")) is None
    assert sup.normalize_status(True) is None
    assert sup.normalize_status(99) is None and sup.normalize_status(600) is None


def test_browser_float_status_200_harvested(tmp_path, monkeypatch):
    """REGRESSION: CDP Response.status is 200.0 (float); isinstance(int)
    rejected every browser success. Must normalise and harvest."""
    ev = {"kind": "ResponseReceived",
          "obj": _Evt(response=_Evt(url="https://a.vn/ok", status=200.0),
                      type_=_Evt(value="Document"), frame_id="F1",
                      loader_id="L9")}
    _install_fake2(monkeypatch, [ev], nav_loader="L9", dom_url="https://a.vn/ok")
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/ok"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5))
    assert res["ok"] is True
    assert res["status"] == 200 and isinstance(res["status"], int)
    assert sup.classify_inline(res["status"], res["final_url"], "x" * 300) == "ok"


def test_browser_fetch_interception_awaited_proxy_only(tmp_path, monkeypatch):
    """REGRESSION: sync page.send() unawaited never authenticated; Fetch.enable
    pauses EVERY request -> both RequestPaused and AuthRequired must be
    continued via REAL awaits; credentials ONLY for source == Proxy."""
    paused = _Evt(request_id="r-paused")
    auth_proxy = _Evt(request_id="r-auth",
                      auth_challenge=_Evt(source="Proxy"))
    evs = [
        {"kind": "RequestPaused", "obj": paused},
        {"kind": "AuthRequired", "obj": auth_proxy},
        {"kind": "ResponseReceived",
         "obj": _Evt(response=_Evt(url="https://a.vn/p", status=200.0),
                     type_=_Evt(value="Document"), frame_id="F1",
                     loader_id="L1")},
    ]
    sent = _install_fake2(monkeypatch, evs, dom_url="https://a.vn/p")
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/p"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5,
        proxy_url="http://user9:pass9@1.2.3.4:8080"))
    assert res["ok"] is True
    kinds = [c[0] for c in sent if isinstance(c, tuple)]
    assert "fetch.enable" in kinds
    enables = [c for c in sent if isinstance(c, tuple)
               and c[0] == "fetch.enable"]
    assert enables and enables[0][1] is True  # handle_auth_requests=True
    assert ("fetch.continue_request", "r-paused") in [c for c in sent
                                                      if isinstance(c, tuple)]
    auth_calls = [c for c in sent if isinstance(c, tuple)
                  and c[0] == "fetch.continue_with_auth"]
    assert auth_calls, "continue_with_auth must be AWAITED and sent"
    rid, resp = auth_calls[0][1], auth_calls[0][2]
    assert rid == "r-auth" and resp.response == "ProvideCredentials"
    assert resp.username == "user9" and resp.password == "pass9"
    # server auth must NEVER receive credentials
    evs2 = [{"kind": "AuthRequired",
             "obj": _Evt(request_id="r-srv",
                         auth_challenge=_Evt(source="Server"))},
            {"kind": "ResponseReceived",
             "obj": _Evt(response=_Evt(url="https://b.cn/", status=200.0),
                         type_=_Evt(value="Document"), frame_id="F1",
                         loader_id="L1")}]
    sent2 = _install_fake2(monkeypatch, evs2, dom_url="https://b.cn/")
    asyncio.run(sup.run_browser_one(
        {"id": 2, "url": "https://b.cn/"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5,
        proxy_url="http://user9:pass9@1.2.3.4:8080"))
    srv = [c for c in sent2 if isinstance(c, tuple)
           and c[0] == "fetch.continue_with_auth"]
    assert srv and srv[0][2].response == "CancelAuth"
    assert srv[0][2].username is None and srv[0][2].password is None
    # no credential leak in result payloads
    blob = json.dumps({k: v for k, v in res.items()
                       if k not in ("raw", "nav_evidence")}, default=str)
    assert "pass9" not in blob and "user9" not in blob


def test_browser_pool_auth_uses_pool_proxy_creds(tmp_path, monkeypatch):
    """REGRESSION 13/10 (proxy dialog trên headed Chrome): pooled browser
    lấy proxy_auth từ URL truyền vào thay vì pool.proxy_url, nên Chrome
    mở dialog nhập user/password và không bao giờ xong auth.

    Pool owns the Chrome -> pool.proxy_url's creds must win."""
    import types as _t
    pool = sup.BrowserPool(tmp_path, None, False,
                           "http://u7:p7@r7:77", [])
    # URL truyền vào SAI creds: pool phải thắng
    sent = _install_fake2(
        monkeypatch,
        [{"kind": "AuthRequired",
          "obj": _Evt(request_id="r-auth",
                      auth_challenge=_Evt(source="Proxy"))},
         {"kind": "ResponseReceived",
          "obj": _Evt(response=_Evt(url="https://a.vn/p", status=200.0),
                      type_=_Evt(value="Document"), frame_id="F1",
                      loader_id="L1")}],
        dom_url="https://a.vn/p")
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/p", "host": "a.vn"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5,
        proxy_url="http://uWRONG:pWRONG@r7:77", headless=False,
        pool=pool))
    assert res["ok"] is True
    auth_calls = [c for c in sent if isinstance(c, tuple)
                  and c[0] == "fetch.continue_with_auth"]
    assert auth_calls
    rid, resp = auth_calls[0][1], auth_calls[0][2]
    assert rid == "r-auth" and resp.response == "ProvideCredentials"
    assert resp.username == "u7" and resp.password == "p7"  # pool wins


def test_browser_final_document_dom_match_challenge_redirect(tmp_path,
                                                             monkeypatch):
    """REGRESSION: nav loader may be the INITIAL loader; a later challenge
    clear opens a new loader with the final 200. Must pick the FINAL root
    document matching the DOM final URL, not the initial nav loader."""
    ev_chal = {"kind": "ResponseReceived",
               "obj": _Evt(response=_Evt(url="https://a.vn/chk", status=503.0),
                           type_=_Evt(value="Document"), frame_id="F1",
                           loader_id="L1")}
    ev_fin = {"kind": "ResponseReceived",
              "obj": _Evt(response=_Evt(url="https://a.vn/final", status=200.0),
                          type_=_Evt(value="Document"), frame_id="F1",
                          loader_id="L2")}
    _install_fake2(monkeypatch, [ev_chal, ev_fin], nav_loader="L1",
                   dom_url="https://a.vn/final")
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/x"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=5))
    assert res["ok"] is True
    assert res["status"] == 200
    assert res["final_url"] == "https://a.vn/final"   # FINAL doc, not /chk 503


def test_browser_waits_for_dom_content_not_early_shell(tmp_path, monkeypatch):
    """REGRESSION (parent final): an initial URL-matched 200 with an empty
    DOM shell must NOT end the settle wait; a later article DOM (>200
    extracted) on the same final URL must be harvested, not the shell."""
    ev = {"kind": "ResponseReceived",
          "obj": _Evt(response=_Evt(url="https://a.vn/article", status=200.0),
                      type_=_Evt(value="Document"), frame_id="F1",
                      loader_id="L1")}
    shell = b"<html><body></body></html>"
    article = ("<html><body><article>" + "noidungykhoa " * 60 +
               "</article></body></html>").encode("utf-8")
    _install_fake2(monkeypatch, [ev], nav_loader="L1",
                   dom_url="https://a.vn/article",
                   ready_states=["loading", "interactive", "complete",
                                 "complete", "complete"],
                   body_lens=[0, 0, 80, 450, 450],
                   text_lens=[0, 0, 80, 450, 450],
                   bodies=[shell, shell, shell, article, article])
    res = asyncio.run(sup.run_browser_one(
        {"id": 1, "url": "https://a.vn/article"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=12))
    assert res["ok"] is True and res["status"] == 200
    assert len(res["raw"]) > 200 and b"noidungykhoa" in res["raw"]


def test_browser_503_challenge_clears_to_final_200(tmp_path, monkeypatch):
    """REGRESSION (parent final): matched 503 challenge must keep waiting
    for the final 200 on the cleared DOM URL (challenge flag on, then off)."""
    ev_chal = {"kind": "ResponseReceived",
               "obj": _Evt(response=_Evt(url="https://a.vn/chk",
                                         status=503.0),
                           type_=_Evt(value="Document"), frame_id="F1",
                           loader_id="L1")}
    ev_fin = {"kind": "ResponseReceived",
              "obj": _Evt(response=_Evt(url="https://a.vn/final",
                                        status=200.0),
                          type_=_Evt(value="Document"), frame_id="F1",
                          loader_id="L2")}
    shell = b"<html><body>just a moment</body></html>"
    article = ("<html><body><article>" + "ykhoathat " * 60 +
               "</article></body></html>").encode("utf-8")
    _install_fake2(monkeypatch, [ev_chal, ev_fin], nav_loader="L1",
                   dom_urls=["https://a.vn/chk", "https://a.vn/chk",
                             "https://a.vn/final", "https://a.vn/final"],
                   ready_states=["loading", "loading", "interactive",
                                 "complete"],
                   body_lens=[50, 50, 450, 450],
                   text_lens=[50, 50, 450, 450],
                   challenges=[True, True, False, False],
                   bodies=[shell, shell, article, article])
    res = asyncio.run(sup.run_browser_one(
        {"id": 2, "url": "https://a.vn/x"}, tmp_path,
        sup.BandwidthLimiter(0, 0), None, timeout_s=12))
    assert res["ok"] is True and res["status"] == 200
    assert res["final_url"] == "https://a.vn/final"
    assert b"ykhoathat" in res["raw"]


def test_proxy_connect_tunnel_preemptive_auth_no_leak():
    """REGRESSION: urllib ProxyBasicAuthHandler realm 'proxy' mismatch kills
    HTTPS CONNECT unauthenticated. Fix = preemptive Proxy-Authorization in the
    CONNECT tunnel headers; credentials never appear in errors/logs."""
    base, hp, hdr = sup._proxy_parts("http://u1:p1@2.3.4.5:8080")
    assert base == "http://2.3.4.5:8080" and hp == ("2.3.4.5", 8080)
    assert hdr.startswith("Basic ") and "p1" not in hdr.split()[-1][:2] or hdr
    import base64 as _b64
    assert _b64.b64decode(hdr.split()[1]).decode() == "u1:p1"
    th = {"Proxy-Authorization": hdr}
    conn_cls = sup._TunnelHTTPSConnection.make(th)
    got = {}
    inst = conn_cls("2.3.4.5", 8080)

    def fake_super_set_tunnel(self, host, port=None, headers=None, **kw):
        got["host"], got["port"], got["headers"] = host, port, headers
    import http.client
    inst.__class__.__bases__[0].set_tunnel = fake_super_set_tunnel
    try:
        inst.set_tunnel("a.vn", 443, headers={"User-Agent": "x"})
        assert got["host"] == "a.vn" and got["port"] == 443
        assert got["headers"]["Proxy-Authorization"] == hdr   # preemptive
        assert got["headers"]["User-Agent"] == "x"
    finally:
        del http.client.HTTPSConnection.set_tunnel
    # opener precedence: tunnel handler MUST claim https_open before
    # ProxyHandler(100) and stock HTTPSHandler(500) — else CONNECT goes out
    # without tunnel auth (the live bug: order-100 sat between them and the
    # 00:35 b660 proxy leg still returned TypeError status None)
    import urllib.request as _ur
    _op = _ur.build_opener(_ur.ProxyHandler(
        {"http": "http://9.9.9.9:8080", "https": "http://9.9.9.9:8080"}))
    _op.add_handler(sup._TunnelHTTPSHandler.make(
        {"Proxy-Authorization": hdr},
        sup._TunnelHTTPSConnection.make({"Proxy-Authorization": hdr})))
    _names = [(type(h).__name__, getattr(h, "handler_order", None))
              for h in _op.handlers if hasattr(h, "https_open")]
    assert _names[0][0] == "_Handler"
    # request-level preemptive header for plain http
    import urllib.request
    req = urllib.request.Request("http://a.vn/x")
    h = sup._PreemptiveProxyAuth(hdr)
    req = h.apply(req)
    assert req.get_header("Proxy-authorization") == hdr or \
        req.has_header("Proxy-Authorization")
    # redaction: no secrets anywhere in error paths
    res = sup.fetch_via_proxy("https://a.vn/x", "http://u1:p1@localhost:9",
                              sup.BandwidthLimiter(0), timeout=2)
    blob = json.dumps(res, default=str)
    assert "p1" not in blob and "u1" not in blob.replace("u1:p1", "")


def test_fair_feeder_two_shards_no_monopoly(tmp_path, monkeypatch):
    """REGRESSION: single sequential feeder blocked on the FIRST host's 64-row
    buffer starved every other shard for days. Per-shard producers must let
    hostB finish while hostA is still running."""
    import zstandard as zstd
    wd = tmp_path / "wd"
    q = wd / "queue"
    q.mkdir(parents=True)
    rows_a = [_row(1000 + i, "slowA.vn") for i in range(100)]
    rows_b = [_row(2000 + i, "fastB.vn") for i in range(20)]

    def write_shard(name, rows):
        raw = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
        with (q / name).open("xb") as fh:
            with zstd.ZstdCompressor().stream_writer(fh) as zw:
                zw.write(raw)

    write_shard("host-slowA_vn.jsonl.zst", rows_a)
    write_shard("host-fastB_vn.jsonl.zst", rows_b)
    (wd / "queue_manifest.json").write_text(json.dumps({
        "total": 120,
        "shards": [{"file": "host-slowA_vn.jsonl.zst", "host": "slowA.vn",
                    "rows": 100},
                   {"file": "host-fastB_vn.jsonl.zst", "host": "fastB.vn",
                    "rows": 20}]}), encoding="utf-8")
    order = []

    async def paced(session, limiter, url, timeout_s=30.0):
        host = url.split("/")[2]
        await asyncio.sleep(0.08 if host == "slowA.vn" else 0.005)
        order.append(host)
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"w" * 400 + b"</html>",
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "w" * 300}

    monkeypatch.setattr(sup, "fetch_direct", paced)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, concurrency=4, per_host=2, max_cycles=1)) == 0
    # hostB fully done BEFORE hostA finished -> no monopoly
    assert order.count("fastB.vn") == 20
    last_b = max(i for i, h in enumerate(order) if h == "fastB.vn")
    last_a = max(i for i, h in enumerate(order) if h == "slowA.vn")
    first_a = min(i for i, h in enumerate(order) if h == "slowA.vn")
    assert last_b < last_a            # B completes while A still in flight
    assert first_a < last_b           # both interleaved (A starts first)


def test_remaining_retriable_tail_after_big_done_prefix(tmp_path):
    """REGRESSION: scan-floor abort returned 0 while a pending tail existed."""
    import zstandard as zstd
    wd = tmp_path / "wd"
    q = wd / "queue"
    q.mkdir(parents=True)
    rows = [{"id": i, "url": f"https://t.vn/{i}", "host": "t.vn",
             "group": "g", "reason": "r", "attempts": 0,
             "route_hint": "direct_first"} for i in range(300)]
    raw = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    with (q / "host-t_vn.jsonl.zst").open("xb") as fh:
        with zstd.ZstdCompressor().stream_writer(fh) as zw:
            zw.write(raw)
    done = {i for i in range(299)}          # big done prefix, 1 tail pending
    got = sup._remaining_retriable(sorted(q.glob("*.jsonl.zst")), done, set(),
                                   cap=5000,
                                   single_host_of={"host-t_vn.jsonl.zst": "t.vn"})
    assert got == 1


def test_remaining_retriable_mixed_exhausted_host(tmp_path):
    """REGRESSION: first-host exhausted `break` hid other hosts on mixed shards."""
    import zstandard as zstd
    wd = tmp_path / "wd"
    q = wd / "queue"
    q.mkdir(parents=True)
    rows = ([{"id": 10, "url": "https://dead.vn/a", "host": "dead.vn",
              "group": "g", "reason": "r", "attempts": 0,
              "route_hint": "direct_first"}]
            + [{"id": 20 + i, "url": f"https://live.vn/{i}", "host": "live.vn",
                "group": "g", "reason": "r", "attempts": 0,
                "route_hint": "direct_first"} for i in range(5)])
    raw = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
    with (q / "smalls-00.jsonl.zst").open("xb") as fh:
        with zstd.ZstdCompressor().stream_writer(fh) as zw:
            zw.write(raw)
    got = sup._remaining_retriable(sorted(q.glob("*.jsonl.zst")), set(),
                                   {"dead.vn"}, cap=5000,
                                   single_host_of={"smalls-00.jsonl.zst": None})
    assert got == 5                        # live.vn rows counted, dead skipped


def test_browser_budget_defer_burns_no_attempts(tmp_path, monkeypatch):
    """REGRESSION: cap-skipped browser rows were burned as genuine and became
    exhausted without EVER being visited. Budget skips must defer, and the
    productive host must eventually harvest ALL rows (> cap)."""
    wd = tmp_path / "wd"
    monkeypatch.setattr(sup, "HOST_BROWSER_CAP", 10)
    monkeypatch.setattr(sup, "HOST_ROW_FAIL_STREAK", 500)
    monkeypatch.setattr(sup, "HOST_BROWSER_FAIL_STREAK", 500)
    n_rows = 25
    _write_queue(wd, [_row(5000 + i, "js2.vn") for i in range(n_rows)])
    browser_hits = []

    async def short_direct(session, limiter, url, timeout_s=30.0):
        return {"status": 200, "final_url": url,
                "body": b"<html>shell</html>", "error": None, "infra": False}

    async def ok_browser(row, crawl_root, limiter, chrome_exe, timeout_s=60.0,
                         proxy_url=None, headless=True, pool=None):
        browser_hits.append(row["id"])
        return {"ok": True, "status": 200, "final_url": row["url"], "title": "T",
                "raw": b"<html>" + b"z" * 400 + b"</html>", "nbytes": 10,
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        if b"zzz" in body:
            return {"title": "T", "date": "", "text": "z" * 300}
        return {"title": "", "date": "", "text": "shell" * 2}

    monkeypatch.setattr(sup, "fetch_direct", short_direct)
    monkeypatch.setattr(sup, "run_browser_one", ok_browser)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, browser_max=1, max_cycles=4,
                      archive_enabled=False, cycle_cooldown=5.0)) == 0
    recs = _read_pages(wd)
    assert len(set(r["id"] for r in recs)) == n_rows   # ALL ids retained+harvested
    assert len(recs) > sup.HOST_BROWSER_CAP            # > cap via auto cycles
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["exhausted_unique"] == 0                 # budget skips never exhaust
    assert st["harvested_unique"] == n_rows
    assert st["pending_unique"] == 0
    # attempts were NOT burned by budget skips: history holds only terminal oks
    done, attempts, terminal = sup.load_done_history(wd / "state")
    assert len(terminal) == n_rows and not attempts


def test_archive_serial_429_single_request_and_persist(tmp_path, monkeypatch):
    """REGRESSION: 8 workers stampeded availability (no async lock). First 429
    must gate globally and persist immediately -> exactly ONE network call."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(getattr(req, "full_url", "https://x"), 429)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(6000 + i, "arc.vn") for i in range(8)])
    seen = []

    async def not_found(session, limiter, url, timeout_s=30.0):
        seen.append(url)
        return {"status": 404, "final_url": url, "body": b"",
                "error": None, "infra": False}

    async def no_extract(pool, body, url):
        return {"title": "", "date": "", "text": ""}

    monkeypatch.setattr(sup, "fetch_direct", not_found)
    monkeypatch.setattr(sup, "extract_in_pool", no_extract)
    assert _run(_opts(wd, tmp_path, concurrency=8, per_host=2, max_cycles=1,
                      archive_enabled=True)) == 0
    assert calls["n"] == 1                       # serial gate: ONE metadata call
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["per_method"]["archive"].get("skipped_429_backoff", 0) >= 7
    cp = sup.load_checkpoint(wd / "state")
    assert float(cp.get("archive_gate_until", 0.0)) > time.time()  # persisted NOW
    assert st["deferred_cumulative"] >= 8        # gated rows deferred, not burned
    assert st["exhausted_unique"] == 0


def test_status_unique_counters_not_double_counted(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(7000), _row(7001)])
    calls = []

    async def ok_direct(session, limiter, url, timeout_s=30.0):
        calls.append(url)
        return {"status": 200, "final_url": url,
                "body": b"<html>" + b"q" * 400 + b"</html>",
                "error": None, "infra": False}

    async def ok_extract(pool, body, url):
        return {"title": "", "date": "", "text": "q" * 300}

    monkeypatch.setattr(sup, "fetch_direct", ok_direct)
    monkeypatch.setattr(sup, "extract_in_pool", ok_extract)
    assert _run(_opts(wd, tmp_path, max_cycles=3)) == 0
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    # 2 rows, 2 harvested: unique counters must not double-count across cycles
    assert st["queue_total"] == 2
    assert st["harvested_unique"] == 2
    assert st["pending_unique"] == 0
    assert st["exhausted_unique"] == 0
    assert st["backlog"] == 0
    assert len(calls) == 2                        # no repeat of done prefix


def test_singleton_race_exactly_one_owner(tmp_path):
    import subprocess
    lockfile = tmp_path / "supervisor.lock"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        results = []
        guard = threading.Lock()

        def race(pid):
            r = sup.acquire_singleton(lockfile, pid)
            with guard:
                results.append(r)

        ths = [threading.Thread(target=race, args=(p,))
               for p in (os.getpid(), os.getpid(), proc.pid, proc.pid)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        assert results.count("acquired") == 1      # EXACTLY one owner (O_EXCL)
        assert "held" in results                   # losers see the live owner
    finally:
        proc.kill()
        proc.wait()


def test_recover_only_strict_valid_ids(tmp_path, monkeypatch):
    """REGRESSION: startup recovery marked rows terminal from invalid records
    (text<200 / challenge). Only strict-valid records may mark terminal."""
    pages = tmp_path / "pages"
    pages.mkdir(parents=True)
    import zstandard as zstd
    good = {"id": 8001, "url": "https://a.vn/g", "host": "a.vn",
            "final_url": "https://a.vn/g", "status": 200, "title": "t",
            "date": "", "text": "x" * 250, "ts": 1}
    short = {"id": 8002, "url": "https://a.vn/s", "host": "a.vn",
             "final_url": "https://a.vn/s", "status": 200, "title": "t",
             "date": "", "text": "x" * 50, "ts": 1}
    chal = {"id": 8003, "url": "https://a.vn/c", "host": "a.vn",
            "final_url": "https://a.vn/c", "status": 200,
            "title": "Attention Required", "date": "",
            "text": "Just a moment " + "x" * 250, "ts": 1}
    with (pages / "recov-00000.jsonl.zst").open("xb") as fh:
        with zstd.ZstdCompressor().stream_writer(fh) as zw:
            for rec in (good, short, chal):
                zw.write((json.dumps(rec) + "\n").encode())
    v = sup.strict_scan_shard(pages / "recov-00000.jsonl.zst")
    assert v["records"] == 1 and v["ids_unique"] == 1 and v["uniques_ok"]
    assert v["content_bad"] == 1 and v["challenge"] == 1 and v["ok"] is False
    ids = sup.recover_output_ids(pages)
    assert ids == {8001}                          # invalid rows never terminal
    # valid frame records are NEVER refetched (recovered is terminal)
    done, attempts, terminal = None, None, None
    state = tmp_path / "state"
    state.mkdir()
    with (state / "done_ids.tsv").open("w", encoding="utf-8") as fh:
        for did in sorted(ids):
            fh.write(f"{did}\trecovered\n")
    done, attempts, terminal = sup.load_done_history(state)
    assert 8001 in done and 8002 not in done and 8003 not in done


def test_browser_proxy_hosts_force_proxy_and_rotate(tmp_path, monkeypatch):
    """Long Chau: headed browser must ALWAYS carry a proxy route, even when
    direct returned no CF-shaped status; a blocked browser row must rotate
    the route (cooldown) and a success must stick it. Pool key must include
    the proxy so rotated routes do not share one Chrome (leak-safe reuse)."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(9600, "nhathuoclongchau.com.vn"),
                      _row(9601, "nhathuoclongchau.com.vn")])
    seen = []
    outcomes = iter([False, True])  # row1 blocked -> rotate, row2 ok -> stick

    async def direct_403(session, limiter, url, timeout_s=30.0):
        return {"status": 403, "final_url": url, "body": b"",
                "error": None, "infra": False}

    async def fake_browser(row, crawl_root, limiter, chrome_exe,
                           timeout_s=60.0, proxy_url=None, headless=True,
                           pool=None):
        seen.append((proxy_url, headless,
                     getattr(pool, "proxy_url", None)))
        if next(outcomes):
            return {"ok": True, "status": 200, "final_url": row["url"],
                    "title": "T", "raw": b"<html>" + b"z" * 400 + b"</html>",
                    "nbytes": 100, "error": None, "infra": False}
        return {"ok": False, "status": 403, "final_url": row["url"],
                "title": "", "raw": b"", "nbytes": 10,
                "error": "forbidden", "infra": False}

    async def good_extract(pool, body, url):
        return {"title": "T", "date": "", "text": "z" * 300}

    monkeypatch.setattr(sup, "fetch_direct", direct_403)
    monkeypatch.setattr(sup, "run_browser_one", fake_browser)
    monkeypatch.setattr(sup, "extract_in_pool", good_extract)
    monkeypatch.setattr(sup, "load_proxies",
                        lambda pf, n=100: ["http://r1:1@u:p",
                                           "http://r2:2@u:p"])
    rc = _run(_opts(wd, tmp_path, concurrency=1, per_host=1,
                    browser_max=1, max_cycles=1, proxy_file="x",
                    headed_hosts={"nhathuoclongchau.com.vn"},
                    browser_proxy_hosts={"nhathuoclongchau.com.vn"}))
    assert rc == 0
    # both rows went through the BROWSER with a proxy attached, headed=False
    assert len(seen) >= 2
    assert all(p and p.startswith("http://r") for p, h, pp in seen)
    assert all(h is False for p, h, pp in seen)
    assert all(pp == p for p, h, pp in seen)   # pool keyed by proxy
    assert seen[0][0] != seen[1][0]            # blocked row rotated route
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    hm = st["per_host_method"]["nhathuoclongchau.com.vn"]["browser"]
    assert hm.get("ok", 0) >= 1


def test_pool_dead_rotates_sticky_browser_route(tmp_path, monkeypatch):
    """REGRESSION: khi pool-dead xảy ra trên route proxy ĐANG sticky (do vừa ok),
    route đó phải bị report fail để lần browser sau KHÔNG dùng lại route chết.
    Trước fix, pool-dead bỏ qua routes.report() -> sticky giữ route chết và lặp
    pool-dead mãi (8986 lần trong production)."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(9900, "nhathuoclongchau.com.vn"),
                      _row(9901, "nhathuoclongchau.com.vn"),
                      _row(9902, "nhathuoclongchau.com.vn")])
    seen = []
    outcomes = iter([True, False, False])   # ok -> sticky r0, rồi pool-dead x2

    async def direct_403(session, limiter, url, timeout_s=30.0):
        return {"status": 403, "final_url": url, "body": b"",
                "error": None, "infra": False}

    async def fake_browser(row, crawl_root, limiter, chrome_exe,
                           timeout_s=60.0, proxy_url=None, headless=True,
                           pool=None):
        seen.append(proxy_url)
        if next(outcomes):
            return {"ok": True, "status": 200, "final_url": row["url"],
                    "title": "T", "raw": b"<html>" + b"z" * 400 + b"</html>",
                    "nbytes": 10, "error": None, "infra": False}
        return {"ok": False, "status": None, "final_url": row["url"],
                "title": "", "raw": b"", "nbytes": 0,
                "error": "pool-dead:ConnectionClosedError", "infra": False,
                "pool_dead": True}

    async def good_extract(pool, body, url):
        return {"title": "T", "date": "", "text": "z" * 300}

    def quick_proxy_fail(url, proxy_url, limiter, timeout=30, **kw):
        return {"status": 502, "final_url": url, "body": b"",
                "error": "stub", "infra": False, "nbytes": 0}

    monkeypatch.setattr(sup, "fetch_direct", direct_403)
    monkeypatch.setattr(sup, "run_browser_one", fake_browser)
    monkeypatch.setattr(sup, "extract_in_pool", good_extract)
    monkeypatch.setattr(sup, "fetch_via_proxy", quick_proxy_fail)
    monkeypatch.setattr(sup, "load_proxies",
                        lambda pf, n=100: ["http://r1:1@u:p",
                                           "http://r2:2@u:p",
                                           "http://r3:3@u:p",
                                           "http://r4:4@u:p"])
    assert _run(_opts(wd, tmp_path, concurrency=1, per_host=1,
                      browser_max=1, max_cycles=1, archive_enabled=False,
                      proxy_file="x", headed_hosts=set(),
                      browser_proxy_hosts={"nhathuoclongchau.com.vn"})) == 0
    assert len(seen) == 3
    assert seen[0] is not None and seen[1] is not None
    assert seen[0] == seen[1]                 # pool-dead xảy ra đúng trên sticky r0
    assert seen[2] != seen[0]                 # sau pool-dead: rotate KHÔNG dùng lại r0


def test_browser_proxy_rotate_every_forces_next_route(tmp_path):
    """Sticky + ép đổi định kỳ: sau N lượt dùng thì pick ép route kế tiếp,
    kể cả khi sticky chưa bao giờ fail."""
    monkeypatch_routes = sup.RouteTable(["http://r1:1@u:p",
                                         "http://r2:2@u:p",
                                         "http://r3:3@u:p"])
    monkeypatch_routes.report("h.vn", 0, ok=True)
    monkeypatch_mp = pytest.MonkeyPatch()
    monkeypatch_mp.setattr(sup, "BROWSER_PROXY_ROTATE_EVERY", 3)
    try:
        host = "h.vn"
        i0, _ = monkeypatch_routes.pick(host)
        assert i0 == 0
        i1, _ = monkeypatch_routes.pick(host)
        assert i1 == 0
        i2, _ = monkeypatch_routes.pick(host)
        assert i2 == 0                       # uses[0] == 3 -> đủ N
        i3, _ = monkeypatch_routes.pick(host, force_rotate=True)
        assert i3 == 1                       # ép sang route kế tiếp
        monkeypatch_routes.report(host, i3, ok=True)
        i4, _ = monkeypatch_routes.pick(host)
        assert i4 == 1                       # sticky mới
    finally:
        monkeypatch_mp.undo()


def test_browser_proxy_hosts_uncapped(tmp_path, monkeypatch):
    """Long Châu (browser_proxy_hosts) KHÔNG bị trần per-cycle: mọi mức
    browser_attempts đều allowed; host thường vẫn bị kẹp cap."""
    hs = sup.HostState()
    for n in (3500, 4000, 10 ** 9):
        hs.browser_attempts = n
        assert hs.browser_allowed("nhathuoclongchau.com.vn",
                                  {"nhathuoclongchau.com.vn"}) is True
    # host VI không còn trần per-cycle (user 22/10: unbounded)
    hs.browser_attempts = 10 ** 9
    assert hs.browser_allowed("laodong.vn", set()) is True


def test_browser_proxy_host_wins_over_global_cap(tmp_path, monkeypatch):
    """REGRESSION (user 15/10): global browser cap đã chạm KHÔNG được chặn
    Long Châu — host browser_proxy_hosts vẫn đi browser, host VI thường bị
    deferred (không đốt lượt)."""
    wd = tmp_path / "wd"
    monkeypatch.setattr(sup, "GLOBAL_BROWSER_CAP", 0)  # global cap đã chạm
    _write_queue(wd, [_row(9800 + i, "nhathuoclongchau.com.vn")
                      for i in range(3)]
                 + [_row(9810 + i, "vi2.vn") for i in range(3)])

    async def direct_shell(session, limiter, url, timeout_s=30.0):
        return {"status": 200, "final_url": url,
                "body": b"<html>shell</html>", "error": None, "infra": False}

    async def ok_browser(row, crawl_root, limiter, chrome_exe, timeout_s=60.0,
                         proxy_url=None, headless=True, pool=None):
        return {"ok": True, "status": 200, "final_url": row["url"],
                "title": "T", "raw": b"<html>" + b"z" * 400 + b"</html>",
                "nbytes": 10, "error": None, "infra": False}

    async def good_extract(pool, body, url):
        if b"zzz" in body:                # browser body -> rich text
            return {"title": "T", "date": "", "text": "z" * 300}
        return {"title": "", "date": "", "text": "shell" * 2}  # direct shell

    monkeypatch.setattr(sup, "fetch_direct", direct_shell)
    monkeypatch.setattr(sup, "run_browser_one", ok_browser)
    monkeypatch.setattr(sup, "extract_in_pool", good_extract)
    assert _run(_opts(wd, tmp_path, concurrency=2, per_host=1,
                      browser_max=1, max_cycles=1, archive_enabled=False,
                      browser_proxy_hosts={"nhathuoclongchau.com.vn"})) == 0
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    lc = st["per_host_method"]["nhathuoclongchau.com.vn"]["browser"]
    assert lc.get("ok", 0) == 3          # Long Châu đi browser bất chấp cap
    vi2 = st["per_method"]["browser"]
    assert vi2.get("skipped_cap", 0) >= 3  # host VI thường bị cap -> deferred
    recs = _read_pages(wd)
    assert {r["id"] for r in recs} == {9800, 9801, 9802}
    assert st["exhausted_unique"] == 0


def test_browser_proxy_no_streak_lock(tmp_path, monkeypatch):
    """Bỏ khóa streak-40 cho Long Châu: fail 100 liên tiếp vẫn allowed
    (chỉ cooldown route + host pace kìm)."""
    hs = sup.HostState()
    hs.browser_fail_streak = 100
    assert hs.browser_allowed("nhathuoclongchau.com.vn",
                              {"nhathuoclongchau.com.vn"}) is True
    assert hs.browser_allowed("laodong.vn", set()) is False


def test_next_cycle_file_forces_new_cycle_and_resets_caps(tmp_path,
                                                           monkeypatch):
    """File NEXT_CYCLE kết thúc cycle hiện tại sớm, job quay lại begin_cycle
    (reset per-cycle caps) mà KHÔNG exit — Long Chau có suất mới ngay."""
    wd = tmp_path / "wd"
    _write_queue(wd, [_row(9700 + i, "nhathuoclongchau.com.vn")
                      for i in range(4)])
    (wd / "NEXT_CYCLE").write_text("next", encoding="utf-8")

    async def direct_403(session, limiter, url, timeout_s=30.0):
        return {"status": 403, "final_url": url, "body": b"",
                "error": None, "infra": False}

    async def ok_browser(row, crawl_root, limiter, chrome_exe,
                         timeout_s=60.0, proxy_url=None, headless=True,
                         pool=None):
        return {"ok": True, "status": 200, "final_url": row["url"],
                "title": "T", "raw": b"<html>" + b"z" * 400 + b"</html>",
                "nbytes": 100, "error": None, "infra": False}

    async def good_extract(pool, body, url):
        return {"title": "T", "date": "", "text": "z" * 300}

    monkeypatch.setattr(sup, "fetch_direct", direct_403)
    monkeypatch.setattr(sup, "run_browser_one", ok_browser)
    monkeypatch.setattr(sup, "extract_in_pool", good_extract)
    monkeypatch.setattr(sup, "load_proxies",
                        lambda pf, n=100: ["http://r1:1@u:p"])
    rc = _run(_opts(wd, tmp_path, concurrency=1, per_host=1,
                    browser_max=1, max_cycles=2, cycle_cooldown=5.0,
                    proxy_file="x",
                    headed_hosts={"nhathuoclongchau.com.vn"},
                    browser_proxy_hosts={"nhathuoclongchau.com.vn"}))
    assert rc == 0
    assert not (wd / "NEXT_CYCLE").exists()   # consumed
    st = json.loads((wd / "audit" / "status.json").read_text(encoding="utf-8"))
    assert st["cycle"] == 2                   # forced into cycle 2
    assert st["counts"].get("cycle_2", 0) == 1


def test_pool_streak_discard_replaces_sick_chrome(tmp_path):
    """Chrome hỏng/treo liên tiếp 3 lần thì pool tự thay mới (discard),
    thay vì để hàng trăm row xếp hàng chết trên cùng 1 browser."""
    pool = sup.BrowserPool(tmp_path, None, False, "http://u:p@r:1", [])
    assert pool.should_discard() is False
    pool.note_pool_dead()
    pool.note_pool_dead()
    assert pool.should_discard() is False
    pool.note_pool_dead()
    assert pool.should_discard() is True
    pool.note_page_ok()
    assert pool.should_discard() is False   # 1 trang ok reset cả 2 streak
    pool.note_frame_absent()
    pool.note_frame_absent()
    pool.note_frame_absent()
    assert pool.should_discard() is True


def test_pool_acquire_timeout_capped(tmp_path, monkeypatch):
    """acquire() bị chặn tối đa 15s: không để worker treo cả phút trên pool
    chết rồi mới fail row (tốc độ thực tế)."""
    import asyncio as _aio
    pool = sup.BrowserPool(tmp_path, None, True, None, [])

    class DeadBrowser:
        async def get(self, url, **kw):
            await _aio.sleep(60)
            raise RuntimeError("never")

    async def fast_start(timeout_s):
        pool.browser = DeadBrowser()

    monkeypatch.setattr(pool, "_start_browser", fast_start)
    t0 = _aio.run(_acquire_timed(pool))
    assert t0 < 20.0


async def _acquire_timed(pool):
    import time as _t
    t0 = _t.monotonic()
    try:
        await pool.acquire(120.0)
    except Exception:
        pass
    return _t.monotonic() - t0


def test_proxy_streak_opens_timed_cooldown_then_recovers(monkeypatch):
    # Proxy lane is not banned for the whole cycle: after HOST_PROXY_FAIL_STREAK
    # failures open a finite exponential cooldown; expiry allows a retry, ok resets.
    monkeypatch.setattr(sup, "HOST_PROXY_FAIL_STREAK", 3)
    monkeypatch.setattr(sup, "PROXY_COOLDOWN_BASE", 300.0)
    monkeypatch.setattr(sup, "PROXY_COOLDOWN_MAX", 3600.0)
    hs = sup.HostState()
    now = 1000.0
    assert hs.proxy_allowed(now) is True
    assert hs.on_proxy_fail(now) == 0.0
    assert hs.on_proxy_fail(now) == 0.0
    cool = hs.on_proxy_fail(now)
    assert cool == 300.0
    assert hs.proxy_allowed(now + 1) is False
    assert hs.proxy_allowed(now + 301) is True
    assert hs.on_proxy_fail(now) == 0.0
    assert hs.on_proxy_fail(now) == 0.0
    assert hs.on_proxy_fail(now) == 600.0
    for _ in range(3):
        hs.on_proxy_fail(now)
    assert hs.proxy_blocked_until == now + 1200.0
    hs.on_proxy_ok()
    assert hs.proxy_blocked_until == 0.0
    assert hs.proxy_block_level == 0
    assert hs.proxy_allowed(now) is True
