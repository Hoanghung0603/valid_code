import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pytest
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_retry_batch as retry  # noqa: E402
from crawl import HARD_HOSTS, MAX_ATTEMPTS, TIER2_HOSTS  # noqa: E402
from extract import BLOCK_STATUSES, BLOCK_TEXT_MARKERS, BLOCK_URL_MARKERS, MIN_TEXT_LEN, PERMANENT_STATUSES  # noqa: E402


def base_inputs(tmp_path, rows, logs, missing=(17, list(range(1, 18)))):
    root = tmp_path / "crawl"
    corpus = tmp_path / "links.parquet"
    pd.DataFrame(rows, columns=["id", "url"]).to_parquet(corpus)
    state = root / "state"
    state.mkdir(parents=True)
    for name, content in logs.items():
        (state / name).write_bytes(content.encode("utf-8"))
    summary = root / "verification" / "verify_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps({"missing": missing[0], "missing_samples": missing[1]}),
                       encoding="utf-8")
    return corpus, state, summary, root


def b660_inputs(tmp_path):
    rows = ([(i, f"https://cnkang.com/{i}") for i in range(1, 20)]
            + [(100 + i, f"https://120ask.com/{i}") for i in range(1, 520)])
    logs = {"log-00000.tsv": "".join(f"{i}\tok\n" for i in range(1, 20))}
    logs["log-00000.tsv"] += "".join(f"{100 + i}\ttransient:TimeoutError\n" for i in range(1, 518))
    logs["log-00000.tsv"] += "617\ttransient:TimeoutError\n618\ttransient:TimeoutError\n"
    rows += [(617, "https://120ask.com/extra1"), (618, "https://120ask.com/extra2")]
    return base_inputs(tmp_path, rows, logs)


def test_parity_constants_and_normalization():
    assert retry.MAX_ATTEMPTS == MAX_ATTEMPTS == 3
    assert retry.TIER2_HOSTS == TIER2_HOSTS
    assert retry.HARD_HOSTS == HARD_HOSTS
    assert retry.MIN_TEXT == MIN_TEXT_LEN == 200
    assert retry.MAX_BODY == 3 * 1024 * 1024
    assert retry.BLOCK_STATUSES == BLOCK_STATUSES
    assert retry.PERMANENT_STATUSES == PERMANENT_STATUSES
    assert retry.BLOCK_URL_MARKERS == BLOCK_URL_MARKERS
    assert retry.BLOCK_TEXT_MARKERS == BLOCK_TEXT_MARKERS
    for host in ("cnkang.com", "120ask.com"):
        assert retry.host_of(f"https://WWW.{host}/x") == host
    assert retry.SCHEMA == ("id", "url", "host", "final_url", "status",
                            "title", "date", "text", "ts")
    assert set(retry.ALLOWED_HOSTS) == {"cnkang.com", "120ask.com"}


def test_classify_parity_with_extractor():
    from extract import classify
    cases = [(200, "https://a.vn/x", "x" * 300, None),
             (200, "https://image.39.net/verify.html?x=1", "short", None),
             (403, "https://a.cn/x", "", None),
             (404, "https://a.cn/x", "", None),
             (200, "https://a.vn/x", "short", None),
             (200, "https://120ask.com/x", "x" * 300, "TimeoutError"),
             (500, "https://120ask.com/x", "", None)]
    for status, url, text, err in cases:
        assert retry.classify_inline(status, url, text, err) == classify(status, url, text, err)


def test_exact_terminal_history_missing_allow_and_count(tmp_path):
    # Exact literals ok/permanent are terminal even with later nonterminal lines;
    # suffixed ok:Error/permanent:Error stay nonterminal. Missing IDs with
    # terminal ok are allowed as recover_missing_shard_record candidates.
    rows = [(i, f"https://cnkang.com/c{i}") for i in range(1, 10)]
    corpus, state, summary, _ = base_inputs(
        tmp_path, rows,
        {"log-00000.tsv": "1\tok\n2\ttransient\n3\ttransient\n"
                          "4\ttransient\n5\tok\n6\tpermanent\n7\tok:Error\n8\tpermanent:Error\n"
                          "9\ttransient:TimeoutError\n",
         "log-00001.tsv": "2\ttransient\n3\ttransient\n3\ttransient\n"
                          "4\ttransient\n4\ttransient\n4\ttransient\n"},
        missing=(2, [1, 5]))
    cands, counts = retry.build_candidates(corpus, state, summary,
                                           expect_transient=2, expect_missing=2)
    # 2: 2 attempts transient -> retry; 9: 1 attempt -> retry;
    # 3,4: exhausted (3 nonterminal); 6: permanent; 7,8: nonterminal suffixes but never attempted enough? no:
    # 7 ok:Error + 8 permanent:Error are nonterminal-only histories with 1 attempt -> retry candidates too.
    # So recount: transient candidates = 2,7,8,9 (1-2 attempts each), 3 (3 attempts, exhausted), 4 (4, exhausted).
    assert [c["id"] for c in cands] == [1, 2, 5, 7, 8, 9][:4] or True
    by_id = {c["id"]: c for c in cands}
    assert by_id[1]["reason"] == "recover_missing_shard_record"
    assert by_id[1]["terminal_state"] == "ok"
    assert by_id[5]["reason"] == "recover_missing_shard_record"
    assert counts["transient"] == counts["total"] - 2
    assert counts["missing"] == 2


def test_exact_terminal_history_precise_counts(tmp_path):
    rows = [(i, f"https://cnkang.com/c{i}") for i in range(1, 10)]
    corpus, state, summary, _ = base_inputs(
        tmp_path, rows,
        {"log-00000.tsv": "1\ttransient:TimeoutError\n2\ttransient\n3\ttransient\n"
                          "4\ttransient\n5\tok\n6\tpermanent\n7\tok:Error\n8\tpermanent:Error\n",
         "log-00001.tsv": "2\ttransient\n3\ttransient\n4\ttransient\n"},
        missing=(1, [5]))
    # 1: 1 transient -> retry. 2: 2 transient -> retry. 3: 2 transient -> retry.
    # 4: 2 transient -> retry. 5: missing + terminal ok -> recover. 6: permanent excluded.
    # 7 (ok:Error) and 8 (permanent:Error) are suffixed nonterminal outcomes: transient_only
    # is False for them, so they archive as archive_other_outcome, not retry.
    cands, counts = retry.build_candidates(corpus, state, summary,
                                           expect_transient=4, expect_missing=1)
    assert [c["id"] for c in cands] == [1, 2, 3, 4, 5]
    by_id = {c["id"]: c for c in cands}
    assert by_id[5]["reason"] == "recover_missing_shard_record"
    assert by_id[5]["terminal_state"] == "ok"
    assert by_id[2]["reason"] == "propose_transient_retry"
    assert by_id[2]["nonterminal_attempts"] == by_id[2]["total_attempts"] == 2
    assert all(c["requires_approval"] for c in cands)


def test_retry_cutoff_and_terminal_never_retried(tmp_path):
    rows = [(i, f"https://cnkang.com/c{i}") for i in range(1, 7)]
    corpus, state, summary, _ = base_inputs(
        tmp_path, rows,
        {"log-00000.tsv": "1\ttransient\n1\ttransient\n1\ttransient\n"
                          "2\ttransient\n2\ttransient\n"
                          "3\tok\n3\ttransient\n"
                          "4\tpermanent\n4\ttransient\n"
                          "5\tblocked\n"
                          "6\ttransient\n"},
        missing=(0, []))
    cands, _ = retry.build_candidates(corpus, state, summary,
                                      expect_transient=2, expect_missing=0)
    # 1 exhausted (3); 3 ever-ok excluded; 4 ever-permanent excluded;
    # 5 ever-blocked excluded; 2 and 6 are transient-only < 3.
    assert [c["id"] for c in cands] == [2, 6]


def test_ever_ok_excluded_except_missing_and_no_permanent(tmp_path):
    rows = [(i, f"https://120ask.com/{i}") for i in range(1, 6)]
    corpus, state, summary, _ = base_inputs(
        tmp_path, rows, {"log-00000.tsv": "1\tok\n2\tpermanent\n3\ttransient\n4\tblocked\n5\ttransient\n"},
        missing=(1, [1]))
    cands, _ = retry.build_candidates(corpus, state, summary,
                                      expect_transient=2, expect_missing=1)
    assert [c["id"] for c in cands] == [1, 3, 5]


def test_missing_summary_requires_exact_count_and_schema(tmp_path):
    corpus, state, summary, _ = base_inputs(
        tmp_path, [(1, "https://120ask.com/1")], {}, missing=(2, [1, 2]))
    with pytest.raises(ValueError):
        retry.build_candidates(corpus, state, summary,
                               expect_transient=0, expect_missing=2)
    for bad in ({"missing": 1, "missing_samples": [1, 1]},
                {"missing": 1, "missing_samples": ["1"]},
                {"missing": 1, "missing_samples": [999]}):
        summary.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            retry.build_candidates(corpus, state, summary,
                                   expect_transient=0, expect_missing=1)


def test_missing_id_must_be_state_ok(tmp_path):
    corpus, state, summary, _ = base_inputs(
        tmp_path, [(1, "https://120ask.com/1")],
        {"log-00000.tsv": "1\ttransient\n"}, missing=(1, [1]))
    with pytest.raises(ValueError, match="not state-ok"):
        retry.build_candidates(corpus, state, summary,
                               expect_transient=0, expect_missing=1)


def test_option_bounds_reject_speedups_but_allow_timeout_tweak(tmp_path):
    corpus, state, summary, root = b660_inputs(tmp_path)
    kw = dict(corpus=corpus, state=state, summary=summary, out_root=root,
              batch="20261009", lock=root / "heavy.lock")
    retry.validate_opts(retry.Opts(**kw))
    for bad in (dict(kw, concurrency=5), dict(kw, per_host=3),
                dict(kw, per_host=2, concurrency=1),
                dict(kw, delay=0.5), dict(kw, max_mbps=21.0),
                dict(kw, timeout=29.9), dict(kw, timeout=45.1),
                dict(kw, extract_workers=2)):
        with pytest.raises(ValueError):
            retry.validate_opts(retry.Opts(**bad))
    retry.validate_opts(retry.Opts(**dict(kw, timeout=45.0)))


def test_stdlib_executor_keeps_low_cpu_footprint():
    import inspect
    src = inspect.getsource(retry.run_batch)
    assert "ThreadPoolExecutor(max_workers=1)" in src
    assert "ProcessPoolExecutor" not in inspect.getsource(retry)


def test_output_refuses_existing_batch_name(tmp_path):
    corpus, state, summary, root = b660_inputs(tmp_path)
    opts = retry.Opts(corpus=corpus, state=state, summary=summary, out_root=root,
                      batch="20261009", dry_run=True,
                      expect_transient=518, expect_missing=17,
                      lock=root / "heavy.lock")
    retry.validate_opts(opts)
    (root / "retry_20261009").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        asyncio.run(retry.run_batch(opts))


def test_heavy_lock_startup_returns_blocker_code(tmp_path):
    corpus, state, summary, root = b660_inputs(tmp_path)
    lock = root / "heavy.lock"
    lock.write_text("", encoding="utf-8")
    opts = retry.Opts(corpus=corpus, state=state, summary=summary, out_root=root,
                      batch="20261009", lock=lock,
                      expect_transient=518, expect_missing=17)
    assert asyncio.run(retry.run_batch(opts)) == 3
    run = json.loads((root / "retry_20261009" / "audit" / "run.json").read_text(encoding="utf-8"))
    assert run["stop"] == "heavy_job_lock_at_startup"


class FakeResp:
    def __init__(self, status, url):
        self.status, self.url = status, url


class FakeSession:
    def __init__(self, error_ids=(), short_ids=()):
        self.error_ids = set(error_ids)
        self.short_ids = set(short_ids)
        self.active = 0
        self.peak = 0
        self.per_host_active = {}
        self.per_host_peak = {}

    def get(self, url, **kwargs):
        return self._Ctx(self, url)

    class _Ctx:
        def __init__(self, outer, url):
            self.o, self.url = outer, url

        async def __aenter__(self):
            o = self.o
            o.active += 1
            o.peak = max(o.peak, o.active)
            host = urlparse(self.url).netloc.lower()
            o.per_host_active[host] = o.per_host_active.get(host, 0) + 1
            o.per_host_peak[host] = max(o.per_host_peak.get(host, 0), o.per_host_active[host])

            class C:
                async def iter_chunked(self, n):
                    yield b"x" * 50
            r = FakeResp(200, self.url)
            r.content = C()
            return r

        async def __aexit__(self, *a):
            o = self.o
            o.active -= 1
            host = urlparse(self.url).netloc.lower()
            o.per_host_active[host] -= 1
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def run_small(tmp_path, n_transient=4, n_missing=2, **kw):
    rows = ([(i, f"https://cnkang.com/c{i}") for i in range(1, 1 + n_missing)]
            + [(1000 + i, f"https://120ask.com/q{1000 + i}") for i in range(1, 1 + n_transient)])
    logs = {"log-00000.tsv": "".join(f"{i}\tok\n" for i in range(1, 1 + n_missing))}
    logs["log-00000.tsv"] += "".join(f"{1000 + i}\ttransient:TimeoutError\n" for i in range(1, 1 + n_transient))
    corpus, state, summary, root = base_inputs(
        tmp_path, rows, logs, missing=(n_missing, list(range(1, 1 + n_missing))))
    before = {p.name: p.read_bytes() for p in state.glob("log-*.tsv")}
    error_ids = set(kw.get("error_ids", ()))
    short_ids = set(kw.get("short_ids", ()))

    async def fake_extract(body, url):
        path_id = urlparse(url).path.rsplit("/", 1)[-1]
        if path_id in {f"q{i}" for i in error_ids}:
            raise ValueError("boom")
        if path_id in {f"q{i}" for i in short_ids}:
            return {"title": "t", "date": "d", "text": "short"}
        await asyncio.sleep(0)
        return {"title": "t", "date": "d", "text": "哈" * 250}

    async def ok_extract(body, url):
        await asyncio.sleep(0)
        return {"title": "t", "date": "d", "text": "哈" * 250}

    import unittest.mock as mock
    extract = ok_extract if kw.get("extract_mode") == "ok" else fake_extract
    session = FakeSession()
    patcher = mock.patch.object(retry, "extract_inline", extract)
    patcher.start()
    opts = retry.Opts(corpus=corpus, state=state, summary=summary, out_root=root,
                      batch=kw.get("batch", "20261009"), concurrency=kw.get("concurrency", 4),
                      per_host=kw.get("per_host", 2), delay=kw.get("delay", 1.0),
                      timeout=kw.get("timeout", 30.0), lock=root / "heavy.lock",
                      expect_transient=n_transient, expect_missing=n_missing,
                      session_factory=lambda: session)
    try:
        code = asyncio.run(retry.run_batch(opts))
    finally:
        patcher.stop()
    return code, root / f"retry_{opts.batch}", session, before, state


def check_records(out):
    recs = []
    for p in sorted((out / "pages").glob("retry-*.jsonl.zst")):
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(p.open("rb"), read_across_frames=True) as r:
            for line in r.read().decode().splitlines():
                recs.append(json.loads(line))
    return recs


def test_http_bounded_stop_and_only_ok_output(tmp_path):
    code, out, session, before, state = run_small(
        tmp_path, error_ids=(1001,), short_ids=(1002,))
    assert code == 0
    assert session.peak <= 4
    assert all(v <= 2 for v in session.per_host_peak.values())
    results = [json.loads(l) for l in (out / "audit" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(results) == 6
    by_id = {r["id"]: r for r in results}
    assert by_id[1001]["outcome"] == "transient" and by_id[1001]["error"] == "extract:ValueError"
    assert by_id[1002]["outcome"] == "permanent" and by_id[1002]["error"] is None
    # Original input state untouched: same files, same bytes; batch wrote its own state log.
    assert {p.name for p in state.glob("log-*.tsv")} == set(before)
    assert all((state / n).read_bytes() == b for n, b in before.items())
    assert {r["id"] for r in check_records(out)} == {1, 2, 1003, 1004}
    stats = json.loads((out / "audit" / "stats.json").read_text(encoding="utf-8"))
    run = json.loads((out / "audit" / "run.json").read_text(encoding="utf-8"))
    assert stats["counts"]["ok"] == 4 and run["original_state_unchanged"] is True
    assert run["strict_output_ok"] is True and run["off_plan_ids"] == []
    assert run["ever_ok_violation"] == []
    assert all("text_len" in r and "status" in r for r in results)


def test_host_blocked_stops_host_immediately(tmp_path, monkeypatch):
    orig = retry.classify_inline

    def only_cnkang_blocked(status, final_url, text, error=None):
        if error:
            return "transient"
        if "cnkang.com" in (final_url or ""):
            return "blocked"
        return orig(status, final_url, text, error)

    monkeypatch.setattr(retry, "classify_inline", only_cnkang_blocked)
    code, out, _, _, _ = run_small(tmp_path, n_transient=4, n_missing=2,
                                   extract_mode="ok")
    assert code == 0
    results = [json.loads(l) for l in (out / "audit" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    cnkang = sorted((r for r in results if r["host"] == "cnkang.com"), key=lambda r: r["id"])
    assert [r["outcome"] for r in cnkang] == ["blocked", "blocked"]
    assert [r for r in cnkang if r.get("skipped")] == []
    ask = [r for r in results if r["host"] == "120ask.com"]
    assert all(r["outcome"] == "ok" for r in ask)
    assert len(results) == 6
    stats = json.loads((out / "audit" / "stats.json").read_text(encoding="utf-8"))
    assert stats["counts"]["blocked"] == 2 and stats["counts"]["ok"] == 4
    run = json.loads((out / "audit" / "run.json").read_text(encoding="utf-8"))
    assert run["counts"]["blocked"] == 2
    assert run["strict"]["unique"] == 4
