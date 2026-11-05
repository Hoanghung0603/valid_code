"""Kiểm tra read-only shard crawl đã đóng (b660 runtime) + đối chiếu state.

Chạy qua tác vụ r2ai_crawl_verify -> crawl_verify_closed.bat (remote: D:\\r2ai\\data\\crawl\\tools).

- Giải nén zstd NGHIÊM theo từng frame bằng decompressobj + kiểm .eof sau mỗi frame
  (không dùng stream_reader, vốn có thể nhận im lặng frame cụt ở cuối file).
- Tự kiểm âm (negative test): zstd 3 frame sạch, bản cắt cụt 7 byte, bản hỏng header;
  cùng hàm scan phải phát hiện -> zstd_selftest.json.
- Stream từng dòng, không giữ text trong RAM; kiểm schema 9 trường, status 200 + text >= 200 ký tự,
  id duy nhất toàn bộ shard, ts hợp lệ; đối chiếu state ok (missing/extra), theo host, ước lượng zh/vi.
- Chỉ ghi vào --out; không sửa/xoá dữ liệu crawl, không đụng tác vụ r2ai_crawl.

Mã thoát: 0 = xong; 1 = file không ổn định/lỗi IO; 2 = lỗi không mong đợi; 3 = heavy_job.lock tồn tại.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import zstandard as zstd

SCHEMA = ("id", "url", "host", "final_url", "status", "title", "date", "text", "ts")
TS_MIN = 1767225600
SAMPLE_CAP = 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tm_str(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def detect_lang(text: str, sample: int = 200) -> str:
    head = text[:sample]
    if not head:
        return "empty"
    cjk = sum(1 for c in head if "\u4e00" <= c <= "\u9fff")
    return "zh" if cjk > len(head) * 0.3 else "vi"


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
        print(f"[verify] WARN lower_priority: {e}", flush=True)


def scan_zstd(path: Path, on_line):
    """Đọc giải nén NGHIÊM; trả {'frames','truncated','zstd_error','out_bytes'}."""
    dctx = zstd.ZstdDecompressor()
    obj = None
    frames = 0
    out_bytes = 0
    zstd_err = None
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
            on_line(bytes(buf[pos:nl]), False)
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
                    out_bytes += len(out)
                    feed(out)
                    if obj.eof:
                        frames += 1
                        data = obj.unused_data
                        obj = None
                    else:
                        data = b""
        if obj is not None:
            out = obj.flush()
            out_bytes += len(out)
            feed(out)
    except zstd.ZstdError as e:
        zstd_err = f"{type(e).__name__}: {e}"[:200]
    if zstd_err is None and obj is not None and not obj.eof:
        zstd_err = "empty_or_no_frame" if (frames == 0 and out_bytes == 0) else "truncated_final_frame"
    if zstd_err is None and frames == 0:
        zstd_err = "empty_or_no_frame"
    if buf:
        on_line(bytes(buf), True)
    return {"frames": frames, "truncated": zstd_err == "truncated_final_frame",
            "zstd_error": zstd_err, "out_bytes": out_bytes}


def zstd_selftest(out_dir: Path) -> dict:
    tmp = out_dir / "_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = [b"".join(json.dumps({"i": i, "frame": k}, separators=(",", ":")).encode() + b"\n" for i in range(300))
             for k in range(3)]
    comp = zstd.ZstdCompressor(level=3)
    clean = tmp / "clean.jsonl.zst"
    with clean.open("wb") as f:
        for part in parts:
            f.write(comp.compress(part))
    data = clean.read_bytes()
    (tmp / "truncated.jsonl.zst").write_bytes(data[:-7])
    corrupt = bytearray(data)
    corrupt[0:4] = b"\x00\x00\x00\x00"
    (tmp / "corrupt.jsonl.zst").write_bytes(bytes(corrupt))

    res = {}
    for name in ("clean", "truncated", "corrupt"):
        box = {"n": 0}

        def cb(line, trunc, b=box):
            b["n"] += 1

        info = scan_zstd(tmp / f"{name}.jsonl.zst", cb)
        info["lines"] = box["n"]
        info["detected_bad"] = bool(info["zstd_error"]) or info["truncated"]
        res[name] = info
    res["passed"] = (
        res["clean"]["frames"] == 3
        and res["clean"]["zstd_error"] is None
        and res["clean"]["lines"] == 900
        and res["truncated"]["detected_bad"]
        and res["corrupt"]["detected_bad"]
    )
    (out_dir / "zstd_selftest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def dd(counter) -> dict:
    return {str(k): v for k, v in counter.items()}


class Checker:
    def __init__(self, min_text: int):
        self.min_text = min_text
        self.ids = set()
        self.g = {
            "records": 0, "malformed": 0, "blank": 0,
            "schema_missing": 0, "schema_missing_fields": Counter(),
            "schema_extra": 0, "schema_extra_fields": Counter(),
            "invalid_samples": [],
            "status": Counter(), "content_ok": 0, "content_bad": Counter(), "content_bad_samples": [],
            "id_bad": 0, "dup": 0, "dup_samples": [], "field_type_bad": Counter(),
            "ts_bad": 0, "ts_min": None, "ts_max": None, "lang": Counter(),
        }
        self.hosts = defaultdict(lambda: {"records": 0, "content_ok": 0, "content_bad": Counter(),
                                          "lang": Counter(), "status": Counter(), "ts_min": None, "ts_max": None})

    def on_line(self, raw: bytes, sh: dict, truncated: bool):
        sh["lines"] += 1
        if truncated:
            sh["tail_lines"] += 1
        if not raw.strip():
            sh["blank"] += 1
            self.g["blank"] += 1
            return
        try:
            rec = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            sh["malformed"] += 1
            self.g["malformed"] += 1
            if len(sh["malformed_samples"]) < 10:
                sh["malformed_samples"].append({"line": sh["lines"], "truncated": truncated, "error": type(e).__name__})
            return
        if not isinstance(rec, dict):
            sh["malformed"] += 1
            self.g["malformed"] += 1
            return
        self.on_record(rec, sh)

    def on_record(self, rec: dict, sh: dict):
        g = self.g
        g["records"] += 1
        sh["records"] += 1

        missing = [f for f in SCHEMA if f not in rec]
        extra = [k for k in rec if k not in SCHEMA]
        if missing or extra:
            g["schema_missing"] += 1 if missing else 0
            g["schema_extra"] += 1 if extra else 0
            for f in missing:
                g["schema_missing_fields"][f] += 1
            for k in extra:
                g["schema_extra_fields"][k] += 1
            if len(g["invalid_samples"]) < SAMPLE_CAP:
                g["invalid_samples"].append({"id": rec.get("id"), "missing": missing, "extra": extra})

        rid = rec.get("id")
        if isinstance(rid, int) and not isinstance(rid, bool):
            if rid in self.ids:
                g["dup"] += 1
                sh["dup_ids"] += 1
                if len(g["dup_samples"]) < SAMPLE_CAP:
                    g["dup_samples"].append(rid)
            else:
                self.ids.add(rid)
        else:
            g["id_bad"] += 1
            sh["id_bad"] += 1

        host = rec.get("host")
        if not isinstance(host, str):
            host = ""
        sh["hosts"][host] += 1
        h = self.hosts[host]
        h["records"] += 1

        status = rec.get("status")
        text = rec.get("text")
        g["status"][status] += 1
        sh["status"][status] += 1
        h["status"][status] += 1

        if status == 200 and isinstance(text, str) and len(text) >= self.min_text:
            g["content_ok"] += 1
            sh["content_ok"] += 1
            h["content_ok"] += 1
        else:
            if status != 200:
                bad = "status_not_200"
            elif not isinstance(text, str):
                bad = "text_not_str"
            else:
                bad = f"text_lt_{self.min_text}"
            g["content_bad"][bad] += 1
            sh["content_bad"][bad] += 1
            h["content_bad"][bad] += 1
            if len(g["content_bad_samples"]) < SAMPLE_CAP:
                g["content_bad_samples"].append({"id": rid, "status": status, "reason": bad})

        for field, typ in (("url", str), ("final_url", str), ("title", str), ("date", str),
                           ("status", int), ("ts", int)):
            val = rec.get(field)
            if not isinstance(val, typ) or isinstance(val, bool):
                g["field_type_bad"][field] += 1
                sh["field_type_bad"][field] += 1

        ts = rec.get("ts")
        if isinstance(ts, int) and not isinstance(ts, bool):
            sh["ts_min"] = ts if sh["ts_min"] is None or ts < sh["ts_min"] else sh["ts_min"]
            sh["ts_max"] = ts if sh["ts_max"] is None or ts > sh["ts_max"] else sh["ts_max"]
            g["ts_min"] = ts if g["ts_min"] is None or ts < g["ts_min"] else g["ts_min"]
            g["ts_max"] = ts if g["ts_max"] is None or ts > g["ts_max"] else g["ts_max"]
            h["ts_min"] = ts if h["ts_min"] is None or ts < h["ts_min"] else h["ts_min"]
            h["ts_max"] = ts if h["ts_max"] is None or ts > h["ts_max"] else h["ts_max"]
            if not (TS_MIN <= ts <= time.time() + 86400):
                g["ts_bad"] += 1
        else:
            g["ts_bad"] += 1

        if isinstance(text, str) and text:
            lang = detect_lang(text)
            g["lang"][lang] += 1
            sh["lang"][lang] += 1
            h["lang"][lang] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=r"D:\r2ai\data\crawl\pages")
    ap.add_argument("--state", default=r"D:\r2ai\data\crawl\state")
    ap.add_argument("--out", default=r"D:\r2ai\data\crawl\verification")
    ap.add_argument("--lock", default=r"D:\r2ai\locks\heavy_job.lock")
    ap.add_argument("--prefix", default="crawl")
    ap.add_argument("--min-text", type=int, default=200)
    ap.add_argument("--pause", type=float, default=0.02)
    a = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    lock = Path(a.lock)
    if lock.exists():
        print(f"[verify] BLOCKED: {lock} ton tai; khong chay.", flush=True)
        return 3

    lower_priority()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = Path(a.pages)
    t0 = time.time()

    selftest = zstd_selftest(out)
    print(f"[verify] selftest passed={selftest['passed']} truncated_detected={selftest['truncated']['detected_bad']}", flush=True)
    if not selftest["passed"]:
        print("[verify] selftest FAILED -> dung.", flush=True)
        return 2

    files = sorted(pages.glob(f"{a.prefix}-*.jsonl.zst"))
    snap_a, sha_a = {}, {}
    for p in files:
        st = p.stat()
        snap_a[p.name] = {"size": st.st_size, "mtime": int(st.st_mtime), "mtime_iso": tm_str(st.st_mtime)}
        sha_a[p.name] = sha256_of(p)

    chk = Checker(a.min_text)
    shards = []
    zstd_error_shards, zstd_truncated_shards = [], []
    frames_total = 0
    for i, p in enumerate(files, 1):
        sh = {"name": p.name, "size": snap_a[p.name]["size"], "mtime": snap_a[p.name]["mtime"],
              "mtime_iso": snap_a[p.name]["mtime_iso"], "sha256_before": sha_a[p.name], "sha256_after": "",
              "stable": False, "records": 0, "lines": 0, "tail_lines": 0, "malformed": 0, "blank": 0,
              "malformed_samples": [], "schema_missing": 0, "schema_extra": 0, "id_bad": 0, "dup_ids": 0,
              "content_ok": 0, "content_bad": Counter(), "field_type_bad": Counter(), "status": Counter(),
              "lang": Counter(), "hosts": Counter(), "ts_min": None, "ts_max": None,
              "frames": 0, "truncated": False, "zstd_error": None, "sha256_final": ""}

        def on_line(raw, trunc, sh=sh):
            chk.on_line(raw, sh, trunc)

        info = scan_zstd(p, on_line)
        frames_total += info["frames"]
        sh["frames"] = info["frames"]
        sh["truncated"] = info["truncated"]
        sh["zstd_error"] = info["zstd_error"]
        if info["zstd_error"]:
            zstd_error_shards.append(p.name)
        if info["truncated"]:
            zstd_truncated_shards.append(p.name)
        shards.append(sh)
        print(f"[verify] shard {i}/{len(files)} {p.name} records={sh['records']} frames={sh['frames']}"
              f" zstd={'ERR' if info['zstd_error'] else 'ok'}", flush=True)
        if a.pause:
            time.sleep(a.pause)

    snap_c = {}
    for p in pages.glob(f"{a.prefix}-*.jsonl.zst"):
        st = p.stat()
        snap_c[p.name] = {"size": st.st_size, "mtime": int(st.st_mtime)}
    names_a, names_c = set(snap_a), set(snap_c)
    added, removed = sorted(names_c - names_a), sorted(names_a - names_c)
    by_name = {s["name"]: s for s in shards}
    unstable, mtime_changed = [], []
    for name in sorted(names_a & names_c):
        final = sha256_of(pages / name)
        by_name[name]["sha256_final"] = final
        by_name[name]["sha256_after"] = final
        by_name[name]["stable"] = (snap_a[name]["size"] == snap_c[name]["size"]) and (sha_a[name] == final)
        if not by_name[name]["stable"]:
            unstable.append(name)
        if snap_a[name]["mtime"] != snap_c[name]["mtime"]:
            mtime_changed.append(name)
    stable_all = not added and not removed and not unstable

    state_ok = set()
    state_lines = state_malformed = 0
    state_outcomes = Counter()
    state_files = []
    for p in sorted(Path(a.state).glob("log-*.tsv")):
        st = p.stat()
        state_files.append({"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime),
                            "mtime_iso": tm_str(st.st_mtime), "sha256": sha256_of(p)})
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                state_lines += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    state_malformed += 1
                    continue
                try:
                    sid = int(parts[0])
                except ValueError:
                    state_malformed += 1
                    continue
                outcome = parts[1]
                state_outcomes[outcome] += 1
                if outcome == "ok":
                    state_ok.add(sid)
    missing = state_ok - chk.ids
    extra = chk.ids - state_ok

    idxs = []
    for name in names_a:
        m = re.search(r"-(\d+)\.jsonl\.zst$", name)
        if m:
            idxs.append(int(m.group(1)))
    gaps = sorted(set(range(min(idxs), max(idxs) + 1)) - set(idxs)) if idxs else []

    manifest = out / "verify_manifest.sha256"
    with manifest.open("w", encoding="utf-8") as f:
        for name in sorted(names_a):
            f.write(f"{sha_a[name]}  {name}\n")

    perhost = {}
    for host, h in sorted(chk.hosts.items()):
        perhost[host] = {"records": h["records"], "content_ok": h["content_ok"],
                         "content_bad": dd(h["content_bad"]), "lang": dd(h["lang"]),
                         "status": dd(h["status"]), "ts_min": h["ts_min"], "ts_max": h["ts_max"]}

    summary = {
        "status": "completed",
        "started_at": tm_str(t0), "finished_at": tm_str(time.time()),
        "duration_s": round(time.time() - t0, 1),
        "state_ok_unique": len(state_ok),
        "shard_records": chk.g["records"],
        "unique": len(chk.ids),
        "duplicates": chk.g["dup"], "duplicate_samples": chk.g["dup_samples"],
        "missing": len(missing), "missing_samples": sorted(missing)[:SAMPLE_CAP],
        "extra": len(extra), "extra_samples": sorted(extra)[:SAMPLE_CAP],
        "invalid_schema": chk.g["schema_missing"] + chk.g["schema_extra"],
        "invalid_schema_missing_fields": dd(chk.g["schema_missing_fields"]),
        "invalid_schema_extra_fields": dd(chk.g["schema_extra_fields"]),
        "invalid_schema_samples": chk.g["invalid_samples"],
        "id_bad": chk.g["id_bad"],
        "field_type_bad": dd(chk.g["field_type_bad"]),
        "malformed_lines": chk.g["malformed"], "blank_lines": chk.g["blank"],
        "content_ok": chk.g["content_ok"],
        "content_violations": dd(chk.g["content_bad"]),
        "content_violation_samples": chk.g["content_bad_samples"],
        "zstd": {"strict_ok": selftest["passed"] and not zstd_error_shards and not zstd_truncated_shards,
                 "selftest_passed": selftest["passed"], "selftest_file": "zstd_selftest.json",
                 "frames": frames_total, "error_shards": zstd_error_shards,
                 "truncated_shards": zstd_truncated_shards},
        "unstable": unstable, "added": added, "removed": removed, "mtime_changed": mtime_changed,
        "stable": stable_all,
        "perhost": perhost,
        "shardscount": len(files),
        "shard_bytes": sum(s["size"] for s in shards),
        "index_gaps": gaps,
        "state": {"files": state_files, "lines": state_lines, "malformed_lines": state_malformed,
                  "outcomes": dd(state_outcomes)},
        "ts_min": chk.g["ts_min"], "ts_max": chk.g["ts_max"], "ts_bad": chk.g["ts_bad"],
        "lang": dd(chk.g["lang"]),
        "shards_file": "verify_shards.json", "manifest_file": "verify_manifest.sha256",
    }
    (out / "verify_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    shards_out = {"state_files": state_files,
                  "shards": [{k: (dd(v) if isinstance(v, Counter) else v) for k, v in s.items()} for s in shards]}
    (out / "verify_shards.json").write_text(json.dumps(shards_out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[verify] DONE shards={len(files)} records={chk.g['records']} unique={len(chk.ids)}"
          f" dup={chk.g['dup']} missing={len(missing)} extra={len(extra)}"
          f" invalid_schema={summary['invalid_schema']} zstd_err={len(zstd_error_shards)}"
          f" stable={stable_all}", flush=True)
    return 0 if stable_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
