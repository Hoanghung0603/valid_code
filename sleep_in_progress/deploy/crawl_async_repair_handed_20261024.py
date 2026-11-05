"""Repair stale-tab contamination in already-handed recovery shards."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\r2ai\data\crawl\tools")
import crawl_async_supervisor as sup
import scan_contamination as sc
import zstandard as zstd

BASE = Path(r"D:\r2ai\data\crawl\async_recovery_20261009")
PAGES = BASE / "pages"
AUDIT = BASE / "audit"
STATE = BASE / "state"
REPORT = AUDIT / "contamination_all_20261024.json"
BACKUP = PAGES / "repair_backup_20261024_handed"
ALLOWED = {f"recov-{i:05d}.jsonl.zst" for i in range(0, 40)}
SKIP_FILES = {"recov-00006.jsonl.zst", "recov-00007.jsonl.zst"}


def raw_lines(path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if line.strip():
                        yield line
            if buf.strip():
                yield buf


def sha(path):
    return sup.sha256_of(path)


def shard_index(name):
    return int(name.split(".")[0].split("-")[1])


def update_manifest(v):
    mp = AUDIT / "manifest_closed.jsonl"
    rows = []
    found = False
    for line in mp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            rows.append(line)
            continue
        if r.get("file") == v["file"]:
            old = r.get("closed_at")
            r = dict(v)
            r["closed_at"] = old or time.strftime("%Y-%m-%d %H:%M:%S")
            r["repaired_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            r["repair"] = "stale-tab contamination 20261024 handed"
            found = True
        rows.append(json.dumps(r, ensure_ascii=False))
    if not found:
        v = dict(v)
        v["closed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        v["repaired_at"] = v["closed_at"]
        v["repair"] = "stale-tab contamination 20261024 handed"
        rows.append(json.dumps(v, ensure_ascii=False))
    tmp = mp.with_name(mp.name + ".tmp")
    tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(tmp, mp)


def repair(entry):
    name = entry["file"]
    if name not in ALLOWED:
        raise SystemExit(f"REFUSE {name}: not a handed shard")
    if name in SKIP_FILES:
        print(json.dumps({"file": name, "skip": "non-VI 404 false positives"}))
        return None
    ids = {int(s["id"]) for s in entry["suspects"]}
    if not ids:
        print(json.dumps({"file": name, "skip": "no suspects"}))
        return None
    path = PAGES / name
    BACKUP.mkdir(parents=True, exist_ok=True)
    bak = BACKUP / name
    if not bak.exists():
        shutil.copy2(path, bak)
    if sha(path) != sha(bak):
        raise SystemExit(f"BACKUP SHA MISMATCH {name}")
    tmp = path.with_name(path.name + ".repair_tmp")
    kept = 0
    removed_ids = set()
    comp = zstd.ZstdCompressor(level=6)
    with tmp.open("wb") as fh:
        zw = comp.stream_writer(fh)
        for line in raw_lines(path):
            rid = json.loads(line)["id"]
            if rid in ids:
                removed_ids.add(int(rid))
                continue
            zw.write(line.rstrip(b"\r") + b"\n")
            kept += 1
            if kept % 50 == 0:
                zw.flush(zstd.FLUSH_FRAME)
        zw.flush(zstd.FLUSH_FRAME)
        fh.flush()
        os.fsync(fh.fileno())
        zw.close()
    if not removed_ids:
        tmp.unlink()
        print(json.dumps({"file": name, "skip": "already clean"}))
        return None
    v = sup.strict_scan_shard(tmp)
    if not v.get("ok"):
        raise SystemExit(f"VERIFY FAILED {name}: {v}")
    if v["records"] != kept or kept + len(removed_ids) != entry["records"]:
        raise SystemExit(
            f"COUNT MISMATCH {name}: kept={kept} removed={len(removed_ids)} "
            f"before={entry['records']} scan={v['records']}")
    os.replace(tmp, path)
    v = sup.strict_scan_shard(path)
    if not v.get("ok") or v["records"] != kept:
        raise SystemExit(f"POST-VERIFY FAILED {name}: {v}")
    post = sc.scan(path)
    if post["counts"]["suspects"]:
        raise SystemExit(
            f"POST-SCAN FAILED {name}: {post['counts']}")
    update_manifest(v)
    with (STATE / "done_ids.tsv").open("a", encoding="utf-8") as fh:
        for rid in sorted(removed_ids):
            fh.write(f"{rid}\tretry\n")
        fh.flush()
        os.fsync(fh.fileno())
    out = {"file": name, "before": entry["records"],
           "removed": len(removed_ids), "after": v["records"],
           "sha": v["sha256"], "ok": v["ok"]}
    print(json.dumps(out))
    return out


def main():
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    shards = sorted(data["shards"], key=lambda s: shard_index(s["file"]))
    results = []
    for entry in shards:
        r = repair(entry)
        if r:
            results.append(r)
    print(json.dumps({"done": len(results), "results": results},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
