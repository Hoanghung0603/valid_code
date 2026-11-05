import hashlib
import json
import sys
from pathlib import Path

import pytest
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
from crawl_salvage_shard import export_missing, salvage_file  # noqa: E402


def _rec(i, host="cnkang.com", text="哈" * 250, status=200):
    return {"id": i, "url": f"http://{host}/{i}", "host": host, "final_url": f"http://{host}/{i}",
            "status": status, "title": "t", "date": "2026-01-01", "text": text, "ts": 1791078028}


def _blob(recs):
    return b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in recs)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _complete_lines_via_stream_reader(path):
    dctx = zstd.ZstdDecompressor()
    buf = bytearray()
    n = 0
    try:
        with path.open("rb") as fh, dctx.stream_reader(fh, read_across_frames=True) as r:
            while True:
                c = r.read(1 << 20)
                if not c:
                    break
                buf += c
                while True:
                    i = buf.find(b"\n")
                    if i < 0:
                        break
                    del buf[:i + 1]
                    n += 1
    except zstd.ZstdError:
        pass
    return n


def test_truncated_final_frame_salvages_complete_lines_only(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    recs = [_rec(i) for i in range(1, 301)]
    data = comp.compress(_blob(recs))
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(data[:-50])
    sha0 = _sha(src)
    expected = _complete_lines_via_stream_reader(src)

    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    rep = salvage_file(src, out, expect_source_sha=None, expected_records=None)

    assert rep["source_truncated_final_frame"] is True
    assert 0 < expected < 300
    assert rep["accepted"] == expected
    assert rep["out_records"] == expected
    assert rep["ids_match"] is True
    assert rep["out_duplicates"] == 0
    assert rep["out_eof_ok"] is True
    assert rep["out_reparse_zstd_error"] is None
    assert rep["source_unchanged"] is True
    assert _sha(src) == sha0
    assert rep["passed"] is True
    assert rep["hosts"] == {"cnkang.com": expected}
    assert rep["noncjk_sample"] == 0


def test_incomplete_final_line_excluded_without_malformed_count(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    body = _blob([_rec(i) for i in range(1, 4)])
    partial = json.dumps(_rec(4), ensure_ascii=False).encode()[:40]
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(comp.compress(body + partial))
    sha0 = _sha(src)

    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    rep = salvage_file(src, out, expect_source_sha=None, expected_records=3)

    assert rep["source_tail_bytes"] > 0
    assert rep["source_truncated_final_frame"] is False
    assert rep["accepted"] == 3
    assert rep["rejected"].get("malformed", 0) == 0
    assert rep["ids_match"] is True
    assert rep["passed"] is True
    assert _sha(src) == sha0


def test_malformed_line_between_frames_counted_and_excluded(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(comp.compress(_blob([_rec(1)]) + b'{"id": broken\n' + _blob([_rec(2)])))
    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    rep = salvage_file(src, out, expect_source_sha=None, expected_records=2)
    assert rep["accepted"] == 2
    assert rep["rejected"].get("malformed") == 1
    assert rep["passed"] is True


def test_refuses_overwrite_collision(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(comp.compress(_blob([_rec(1)])))
    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        salvage_file(src, out, expect_source_sha=None, expected_records=1)
    assert out.read_bytes() == b"existing"


def test_source_sha_mismatch_raises(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(comp.compress(_blob([_rec(1)])))
    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    with pytest.raises(ValueError):
        salvage_file(src, out, expect_source_sha="0" * 64, expected_records=1)
    assert not out.exists()


def test_dedup_and_content_rejects(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    recs = [_rec(1), _rec(1), _rec(2, text="ngắn"), _rec(3, status=404)]
    bad = _rec(4)
    del bad["final_url"]
    recs.append(bad)
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(comp.compress(_blob(recs)))
    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    rep = salvage_file(src, out, expect_source_sha=None, expected_records=1)
    assert rep["accepted"] == 1
    assert rep["rejected"]["dup_id"] == 1
    assert rep["rejected"]["content"] == 2
    assert rep["rejected"]["schema"] == 1
    assert rep["out_duplicates"] == 0
    assert rep["passed"] is True


def test_mixed_language_sampling(tmp_path):
    comp = zstd.ZstdCompressor(level=3)
    recs = [_rec(1), _rec(2, host="120ask.com", text="x" * 300)]
    src = tmp_path / "crawl-00009.jsonl.zst"
    src.write_bytes(comp.compress(_blob(recs)))
    out = tmp_path / "salvage" / "crawl-salvaged-00009.jsonl.zst"
    rep = salvage_file(src, out, expect_source_sha=None, expected_records=2)
    assert rep["hosts"] == {"cnkang.com": 1, "120ask.com": 1}
    assert rep["noncjk_sample"] == 1
    assert rep["language"] == "ZH theo host"
    assert rep["passed"] is True


def test_export_missing_records(tmp_path):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"id": [1, 2, 3, 99], "url": ["http://cnkang.com/a", "http://120ask.com/b",
                                                    "http://cnkang.com/c", "http://x.com/d"]})
    parquet = tmp_path / "links.parquet"
    df.to_parquet(parquet)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"missing_samples": [2, 3, 7]}), encoding="utf-8")
    out = tmp_path / "missing_records.jsonl"
    rep = export_missing(summary, parquet, out)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rep["requested"] == 3
    assert rep["found"] == 2
    assert rep["not_found"] == [7]
    assert rep["passed"] is False
    assert {r["id"] for r in rows} == {2, 3}
    assert all(r["reason"] == "state_ok_missing_data" for r in rows)
