"""Tests for DbSync in longchau_odd_fast — verifies DB write-back while crawling."""
from __future__ import annotations

import pathlib, sys, types, tempfile, json, asyncio, shutil
import pytest

# ── load module under test ──────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'crawl'))
import longchau_odd_fast as lof
import longchau_crawl4ai_fast as eng


# ── helpers ─────────────────────────────────────────────────────────────────

def _tmp():
    """Return a temporary directory that works on Windows (no AppData/Temp restriction)."""
    d = pathlib.Path(tempfile.mkdtemp(dir=ROOT / 'tests' / '_tmp_db_sync'))
    return d


def _make_db(tmp: pathlib.Path):
    """Create a minimal DuckDB with crawl_status table containing 2 odd IDs."""
    import duckdb
    db = tmp / 'test.duckdb'
    con = duckdb.connect(str(db))
    con.execute("""
        CREATE TABLE crawl_status (
            id           BIGINT PRIMARY KEY,
            url          VARCHAR,
            final_url    VARCHAR,
            lang_expected VARCHAR,
            status       VARCHAR DEFAULT 'pending',
            http_code    INTEGER,
            content_type VARCHAR,
            n_bytes      BIGINT,
            attempts     INTEGER DEFAULT 0,
            fetched_at   TIMESTAMP,
            shard        INTEGER,
            sha1_raw     VARCHAR,
            error_msg    VARCHAR
        )
    """)
    con.execute("INSERT INTO crawl_status (id, url, status) VALUES (101, 'https://nhathuoclongchau.com.vn/x', 'pending')")
    con.execute("INSERT INTO crawl_status (id, url, status) VALUES (103, 'https://nhathuoclongchau.com.vn/y', 'pending')")
    con.close()
    return db


def _status(db: pathlib.Path, id_: int):
    import duckdb
    con = duckdb.connect(str(db))
    row = con.execute("SELECT status, http_code, sha1_raw, n_bytes FROM crawl_status WHERE id=?", [id_]).fetchone()
    con.close()
    return row


# ── tests ────────────────────────────────────────────────────────────────────

class TestDbSync:

    @pytest.fixture(autouse=True)
    def _tmpdir(self):
        """Use a local temp dir that avoids Windows AppData/Temp permission issues."""
        base = ROOT / 'tests' / '_tmp_db_sync'
        base.mkdir(exist_ok=True)
        d = pathlib.Path(tempfile.mkdtemp(dir=base))
        self._tmp = d
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_push_and_flush_writes_ok_to_db(self):
        db = _make_db(self._tmp)
        sync = lof.DbSync(db, batch_size=10)
        row = dict(id=101, status='ok', final_url='https://nhathuoclongchau.com.vn/x',
                   http_code=200, content_type='text/html', n_bytes=1234,
                   fetched_at='2026-10-22T07:00:00+00:00', sha1_raw='abc123', error_msg=None)
        sync.push(row)
        sync.close()  # flush on close

        result = _status(db, 101)
        assert result is not None
        assert result[0] == 'ok'
        assert result[1] == 200
        assert result[2] == 'abc123'
        assert result[3] == 1234

    def test_batch_waits_for_staging_flush(self):
        db = _make_db(self._tmp)
        sync = lof.DbSync(db, batch_size=1)
        row = dict(id=101, status='ok', final_url='https://nhathuoclongchau.com.vn/x',
                   http_code=200, content_type='text/html', n_bytes=999,
                   fetched_at='2026-10-22T07:00:00+00:00', sha1_raw='xyz', error_msg=None)
        sync.push(row)
        # The collector flushes raw shards before calling DbSync.flush().
        result = _status(db, 101)
        assert result[0] == 'pending'
        sync.flush()
        result = _status(db, 101)
        assert result[0] == 'ok'
        sync.close()

    def test_does_not_overwrite_already_ok_rows(self):
        """UPDATE has WHERE status NOT IN ('ok','non_html','empty') — already-ok rows must be protected."""
        import duckdb
        tmp = self._tmp
        db = _make_db(tmp)
        # Pre-set row 101 to ok
        con = duckdb.connect(str(db))
        con.execute("UPDATE crawl_status SET status='ok', http_code=200, sha1_raw='original' WHERE id=101")
        con.close()

        sync = lof.DbSync(db, batch_size=1)
        row = dict(id=101, status='blocked', final_url='https://nhathuoclongchau.com.vn/x',
                   http_code=403, content_type='text/html', n_bytes=0,
                   fetched_at='2026-10-22T08:00:00+00:00', sha1_raw='new_hash', error_msg='blocked')
        sync.push(row)
        sync.close()

        result = _status(db, 101)
        # Should still be 'ok' from before — not overwritten
        assert result[0] == 'ok'
        assert result[2] == 'original'

    def test_collect_calls_db_sync_push(self):
        """collect() must call db_sync.push() for each successful fetch."""
        pushed = []

        class FakeSync:
            def push(self, row): pushed.append(dict(row))
            def flush(self): pass
            def close(self): pass

        # Minimal fake base module
        base = types.SimpleNamespace(
            STATUS_COLUMNS=lof.STATUS_COLUMNS,
            CHALLENGE_MARKERS=lof.CHALLENGE_MARKERS,
            RawOutput=lof.RawOutput,
            is_longchau=lof.is_longchau,
            timestamp=lof.timestamp,
            now=lof.now,
            atomic_json=lof.atomic_json,
            wait_for_rest=lambda p, s: True,
            encoding=lof.encoding,
            COOLDOWN_MINUTES=10,
        )

        # Fake crawler result
        FakeResult = types.SimpleNamespace(
            url='https://nhathuoclongchau.com.vn/bai-viet/test',
            status_code=200, redirected_status_code=None,
            response_headers={}, html='<html>ok</html>',
            error_message=None, success=True,
        )

        class FakeCrawler:
            async def arun(self, url, config): return FakeResult
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            crawler_strategy = None

        args = types.SimpleNamespace(
            probe=False, scan=True, batch_size=50, delay=0.0,
            jitter=False, trial_urls=0, sample_first=0,
            block_cooldown_minutes=10, continuous=True,
        )
        rows = [{'id': 101, 'url': 'https://nhathuoclongchau.com.vn/bai-viet/test'}]
        state = {'kind': 'scan_ready', 'next_probe_at': None}
        out = self._tmp / 'out'
        out.mkdir()

        asyncio.run(eng.collect(
            rows, out, state, args, base,
            crawler_factory=lambda: FakeCrawler(),
            run_config=object(),
            db_sync=FakeSync(),
        ))

        assert len(pushed) == 1
        assert pushed[0]['status'] == 'ok'
        assert pushed[0]['id'] == 101
