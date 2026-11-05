"""Cứu shard crawl bị cụt frame cuối (offline, chỉ đọc nguồn, không crawl).

- Giữ các dòng JSON hoàn chỉnh đạt schema hiện hành (9 trường), status == 200, text >= 200;
  dữ liệu từ frame cuối bị cụt vẫn được cứu nếu dòng đã hoàn chỉnh; dòng dở bị loại.
- Output shard mới mở bằng 'xb' (không ghi đè); schema giữ nguyên; không đụng state/nguồn.
- Đọc lại output bằng scanner strict (decompressobj + .eof) và so id-set nguồn vs output; yêu cầu khớp, 0 trùng.
- Chế độ missing: xuất missing_records.jsonl (id/url/host/reason) từ parquet read-only theo verify_summary.json.

Mã thoát: 0 = đạt mọi bất biến; 1 = report ghi nhưng bất biến sai; 2 = lỗi (output tồn tại, sha nguồn lệch, ...).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import zstandard as zstd

SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
MIN_TEXT = 200
EXPECT_SHA = "7b88fb4b6e143e4b057735a10bb028a6f9ff1c6fbea34235fe8c4dd1a82b38f9"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def lower_priority():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
    except Exception as e:
        print(f"[salvage] WARN lower_priority: {e}", flush=True)


def _new_stats() -> dict:
    return {"frames": 0, "lines": 0, "out_bytes": 0, "tail_bytes": 0, "eof_ok": False,
            "truncated_final_frame": False, "zstd_error": None}


def scan_strict(path: Path, stats: dict, line_cb):
    dctx = zstd.ZstdDecompressor()
    obj = None
    buf = bytearray()
    pos = 0

    def feed(out: bytes):
        nonlocal pos
        if not out:
            return
        buf.extend(out)
        while True:
            nl = buf.find(10, pos)
            if nl < 0:
                break
            stats["lines"] += 1
            line_cb(bytes(buf[pos:nl]))
            pos = nl + 1
        if pos:
            del buf[:pos]
            pos = 0

    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                data = chunk
                while data:
                    if obj is None:
                        obj = dctx.decompressobj()
                    out = obj.decompress(data)
                    stats["out_bytes"] += len(out)
                    feed(out)
                    if obj.eof:
                        stats["frames"] += 1
                        data = obj.unused_data
                        obj = None
                    else:
                        data = b""
        if obj is not None:
            out = obj.flush()
            stats["out_bytes"] += len(out)
            feed(out)
            stats["eof_ok"] = bool(obj.eof)
            if not obj.eof:
                stats["truncated_final_frame"] = True
        else:
            stats["eof_ok"] = True
    except zstd.ZstdError as e:
        stats["zstd_error"] = f"{type(e).__name__}: {e}"[:200]
        stats["eof_ok"] = False
    if buf:
        stats["tail_bytes"] = len(buf)


def salvage_file(src: Path, out_path: Path, expect_source_sha: str = None,
                 expected_records: int = None) -> dict:
    stats = _new_stats()
    sha_before = sha256_of(src)
    if expect_source_sha and sha_before != expect_source_sha:
        raise ValueError(f"sha256 nguồn lệch: {sha_before} != {expect_source_sha}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected = Counter()
    ids = set()
    hosts = Counter()
    noncjk = 0
    accepted = 0
    dup = 0
    fh = out_path.open("xb")
    try:
        zw = zstd.ZstdCompressor(level=6).stream_writer(fh)

        def cb(line: bytes):
            nonlocal accepted, dup, noncjk
            try:
                rec = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                rejected["malformed"] += 1
                return
            if not isinstance(rec, dict):
                rejected["not_object"] += 1
                return
            missing = [f for f in SCHEMA if f not in rec]
            extra = [k for k in rec if k not in SCHEMA]
            if missing or extra:
                rejected["schema"] += 1
                return
            rid = rec["id"]
            if not isinstance(rid, int) or isinstance(rid, bool):
                rejected["id"] += 1
                return
            text = rec["text"]
            if rec["status"] != 200 or not isinstance(text, str) or len(text) < MIN_TEXT:
                rejected["content"] += 1
                return
            if rid in ids:
                dup += 1
                rejected["dup_id"] += 1
                return
            ids.add(rid)
            accepted += 1
            host = rec["host"] if isinstance(rec["host"], str) else ""
            hosts[host] += 1
            head = text[:200]
            cjk = sum(1 for c in head if "\u4e00" <= c <= "\u9fff")
            if not (head and cjk > len(head) * 0.3):
                noncjk += 1
            zw.write(line + b"\n")

        scan_strict(src, stats, cb)
        zw.flush(zstd.FLUSH_FRAME)
        zw.close()
    except Exception:
        fh.close()
        raise
    sha_after = sha256_of(src)
    out_sha = sha256_of(out_path)

    stats2 = _new_stats()
    out_ids = set()
    out_dups = 0

    def cb2(line: bytes):
        nonlocal out_dups
        rec = json.loads(line)
        rid = rec["id"]
        if rid in out_ids:
            out_dups += 1
        else:
            out_ids.add(rid)

    scan_strict(out_path, stats2, cb2)

    records_match = expected_records is None or accepted == expected_records
    passed = (sha_before == sha_after and ids == out_ids and out_dups == 0
              and stats2["eof_ok"] and not stats2["zstd_error"] and records_match)
    return {
        "mode": "salvage", "src": str(src), "out": str(out_path),
        "source_sha256_before": sha_before, "source_sha256_after": sha_after,
        "source_unchanged": sha_before == sha_after,
        "source_complete_lines": stats["lines"], "source_frames": stats["frames"],
        "source_truncated_final_frame": stats["truncated_final_frame"],
        "source_zstd_error": stats["zstd_error"], "source_tail_bytes": stats["tail_bytes"],
        "accepted": accepted, "rejected": dict(rejected),
        "out_records": stats2["lines"], "out_sha256": out_sha, "out_size": out_path.stat().st_size,
        "out_frames": stats2["frames"], "out_eof_ok": stats2["eof_ok"],
        "out_reparse_zstd_error": stats2["zstd_error"],
        "ids_match": ids == out_ids, "out_duplicates": out_dups,
        "expected_records": expected_records, "records_match_expected": records_match,
        "hosts": dict(hosts), "noncjk_sample": noncjk, "language": "ZH theo host",
        "passed": passed,
    }


def export_missing(summary_path: Path, parquet_path: Path, out_path: Path,
                   reason: str = "state_ok_missing_data") -> dict:
    import pandas as pd
    ids = json.loads(Path(summary_path).read_text(encoding="utf-8"))["missing_samples"]
    df = pd.read_parquet(parquet_path, columns=["id", "url"])
    sub = df[df["id"].isin(set(ids))]
    lines = []
    found = set()
    for row in sub.itertuples(index=False):
        host = urlparse(str(row.url)).netloc.lower().removeprefix("www.")
        found.add(int(row.id))
        lines.append(json.dumps({"id": int(row.id), "url": str(row.url), "host": host, "reason": reason},
                                ensure_ascii=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    rep = {"mode": "missing", "summary": str(summary_path), "parquet": str(parquet_path),
           "requested": len(ids), "found": len(lines), "not_found": sorted(set(ids) - found),
           "out": str(out_path), "out_sha256": sha256_of(out_path), "out_size": out_path.stat().st_size,
           "passed": len(lines) == len(ids)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("salvage", "missing"), default="salvage")
    ap.add_argument("--src", default=r"D:\r2ai\data\crawl\pages\crawl-00009.jsonl.zst")
    ap.add_argument("--out", default=r"D:\r2ai\data\crawl\salvage\crawl-salvaged-00009.jsonl.zst")
    ap.add_argument("--report", default=r"D:\r2ai\data\crawl\verification\salvage_00009_report.json")
    ap.add_argument("--expect-source-sha", default=EXPECT_SHA)
    ap.add_argument("--expected-records", type=int, default=15930)
    ap.add_argument("--summary", default=r"D:\r2ai\data\crawl\verification\verify_summary.json")
    ap.add_argument("--parquet", default=r"D:\r2ai\data\links_corpus.parquet")
    ap.add_argument("--out-missing", default=r"D:\r2ai\data\crawl\verification\missing_records.jsonl")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    lower_priority()
    if a.mode == "salvage":
        rep = salvage_file(Path(a.src), Path(a.out),
                           expect_source_sha=a.expect_source_sha or None,
                           expected_records=a.expected_records if a.expected_records > 0 else None)
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False))
        return 0 if rep["passed"] else 1
    rep = export_missing(Path(a.summary), Path(a.parquet), Path(a.out_missing))
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
