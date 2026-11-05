"""Per-host stats for async-recovery shards NOT yet handed off.

Bounded explicit filename list, read-only. Marks the ACTIVE shard (still being
written) separately: its record count is only what is flushed so far, so it is
NOT handoff-ready even if it reads cleanly.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crawl_async_supervisor as sup  # noqa: E402

PAGES = Path(r"D:\r2ai\data\crawl\async_recovery_20261009\pages")
# shards closed after the 17/10 handoff (recov-00000..00032 already handed off)
FILES = [("recov-00033.jsonl.zst", False)]
# active shard: read-only peek, not handoff-ready
ACTIVE = "recov-00034.jsonl.zst"


def scan(path: Path) -> dict:
    v = sup.strict_scan_shard(path)
    v.pop("ids", None)
    hosts: Counter = Counter()
    for rec in sup.iter_jsonl_zst(path):
        hosts[str(rec.get("host") or "?")] += 1
    return {"records": v["records"], "ids_unique": v["ids_unique"],
            "strict_eof": v["strict_eof"], "zstd_error": v["zstd_error"],
            "sha256": v["sha256"], "size": v["size"], "ok": v["ok"],
            "hosts": dict(hosts.most_common())}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    out = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "closed": [],
           "active": None}
    total: Counter = Counter()
    for name, _ in FILES:
        p = PAGES / name
        if not p.exists():
            out.setdefault("missing", []).append(name)
            continue
        r = scan(p)
        out["closed"].append(dict(r, file=name))
        total.update(r["hosts"])
    ap = PAGES / ACTIVE
    if ap.exists():
        try:
            r = scan(ap)
            out["active"] = dict(r, file=ACTIVE,
                                 note="ACTIVE dang ghi - KHONG handoff-ready")
            total.update(r["hosts"])
        except Exception as e:
            out["active"] = {"file": ACTIVE, "error": type(e).__name__}
    out["total_hosts"] = dict(total.most_common())
    out["total_records"] = int(sum(total.values()))
    print("STATS " + json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())