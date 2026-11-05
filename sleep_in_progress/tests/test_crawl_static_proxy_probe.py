import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_static_proxy_probe as p  # noqa: E402


def _corpus(tmp_path, rows):
    cp = tmp_path / "links.parquet"
    pd.DataFrame(rows, columns=["id", "url"]).to_parquet(cp)
    return cp


def test_parse_proxy_line_both_formats_no_secret_echo():
    a = p.parse_proxy_line("203.0.113.10:8080:alice:s3cret")
    assert a and "s3cret" in a and "alice" in a  # internal use only
    # redaction must strip it for any log surface
    assert "s3cret" not in p.redact(a)
    assert "alice" not in p.redact("user=alice pass=s3cret")
    b = p.parse_proxy_line("http://bob:pw123@203.0.113.13:8080")
    assert b and p.redact(b) == "http://***@203.0.113.13:8080"
    assert p.parse_proxy_line("not a proxy") is None
    assert p.parse_proxy_line("# comment") is None


def test_load_proxy_index0_fixed_counts_only(tmp_path, capsys):
    pf = tmp_path / "prox.txt"
    pf.write_text("203.0.113.11:1111:u0:p0\n203.0.113.12:2222:u1:p1\n", encoding="utf-8")
    first, st = p.load_proxy_index0(pf)
    assert first and ":2222" not in first  # index 0 fixed, no rotation
    assert st["lines"] == 2 and st["valid"] == 2 and st["scheme"] == "http"
    out = capsys.readouterr().out
    assert "u0" not in out and "p0" not in out  # loader prints nothing secret


def test_no_valid_proxy_reports_counts_no_examples(tmp_path, capsys):
    pf = tmp_path / "prox.txt"
    pf.write_text("garbage line\n", encoding="utf-8")
    cp = _corpus(tmp_path, [(1, "https://zysjonline.com/a")])
    rc, audit = p.run(cp, tmp_path / "o", pf, delay=3.0,
                      session_get=lambda u, pr: (_ for _ in ()).throw(AssertionError("no fetch")))
    assert rc == 1
    assert audit["error"] == "no valid proxy"
    out = capsys.readouterr().out
    assert "garbage" not in out and "u0" not in out


def test_candidate_scope_max3_and_host_only(tmp_path):
    rows = [(i, f"https://zysjonline.com/x{i}") for i in range(1, 10)]
    rows += [(50, "https://cnkang.com/other")]
    cp = _corpus(tmp_path, rows)
    cands = p.select_candidates(cp, 3)
    assert len(cands) == 3
    assert all(c["url"].startswith("https://zysjonline.com/") for c in cands)
    with pytest.raises(ValueError):
        p.select_candidates(cp, 4)  # helper enforces caller cap via run path too
    # run() itself caps at 3 regardless
    pf = tmp_path / "prox.txt"
    pf.write_text("203.0.113.11:1111:u0:p0\n", encoding="utf-8")
    seen = []

    def fake(u, pr):
        seen.append(u)
        return {"status": 500, "final_url": u, "body": b"", "nbytes": 0,
                "elapsed": 0.01, "error": None}

    rc, audit = p.run(cp, tmp_path / "o2", pf, delay=3.0, session_get=fake)
    assert rc == 0 and audit["attempted"] == 3 and len(seen) == 3


def _ok_body():
    html = ("<html><head><title>T</title></head><body><article><p>" +
            "nội dung bài viết " * 60 + "</p></article></body></html>").encode()
    return html


def test_407_stops_batch_no_secret_logs(tmp_path, capsys):
    pf = tmp_path / "prox.txt"
    pf.write_text("203.0.113.11:1111:u0:p0\n", encoding="utf-8")
    cp = _corpus(tmp_path, [(1, "https://zysjonline.com/a"), (2, "https://zysjonline.com/b")])
    calls = []

    def fake(u, pr):
        calls.append(pr)
        return {"status": 407, "final_url": u, "body": b"", "nbytes": 0,
                "elapsed": 0.01, "error": None}

    rc, audit = p.run(cp, tmp_path / "o", pf, delay=3.0, session_get=fake)
    assert rc == 3 and len(calls) == 1  # first proxy-auth endpoint fail -> stop
    out = capsys.readouterr().out
    assert "u0" not in out and "p0" not in out and "203.0.113.11" not in out


def test_429_stops_host_no_fallback(tmp_path, capsys):
    pf = tmp_path / "prox.txt"
    pf.write_text("203.0.113.11:1111:u0:p0\n", encoding="utf-8")
    cp = _corpus(tmp_path, [(1, "https://zysjonline.com/a"), (2, "https://zysjonline.com/b")])
    calls = []

    def fake(u, pr):
        calls.append(u)
        return {"status": 429, "final_url": u, "body": b"", "nbytes": 0,
                "elapsed": 0.01, "error": None}

    rc, audit = p.run(cp, tmp_path / "o", pf, delay=3.0, session_get=fake)
    assert rc == 4 and len(calls) == 1  # stop host, no try of remaining


def test_no_url_with_credentials_and_no_raw_html(tmp_path):
    pf = tmp_path / "prox.txt"
    pf.write_text("203.0.113.11:1111:u0:p0\n", encoding="utf-8")
    cp = _corpus(tmp_path, [(1, "https://zysjonline.com/a")])

    def fake(u, pr):
        assert "u0" not in u and "p0" not in u  # error: no URL with credential
        return {"status": 200, "final_url": u, "body": _ok_body(),
                "nbytes": len(_ok_body()), "elapsed": 0.05, "error": None}

    rc, audit = p.run(cp, tmp_path / "o", pf, delay=3.0, session_get=fake)
    assert rc == 0 and audit["ok"] == 1
    recs = [json.loads(x) for x in (tmp_path / "o.records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 1 and set(recs[0]) == set(p.SCHEMA)
    assert "<html" not in recs[0]["text"]
    assert "u0" not in recs[0]["text"] and "p0" not in (tmp_path / "o.audit.jsonl").read_text()
    # exception surfaces redacted
    assert "u0" not in p.redact("Traceback ip 203.0.113.11:1111:u0:p0 boom")


def test_extract_uses_local_scripts_extract():
    import inspect
    import crawl_static_proxy_probe as q
    src = inspect.getsource(q)
    assert "from extract import" in src and "scripts" in src
