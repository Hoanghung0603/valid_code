"""Strict output-contract verifier for async_recovery pages (HANDOFF gate).

NOT a double-hash: each shard is strict-verified (decompressobj + .eof,
schema-9 exact keys, id int, status 200, text>=200, challenge fail-closed)
and hashed once. Only shards with ok=true may enter HANDOFF.

Usage: python crawl_async_verify_output.py --pages <dir> [--json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crawl_async_supervisor import strict_scan_shard  # noqa: E402


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", required=True, type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rows = []
    all_ids = set()
    dupes = 0
    for p in sorted(a.pages.glob("*.jsonl.zst")):
        v = strict_scan_shard(p)
        for did in v.pop("ids", []):
            if did in all_ids:
                dupes += 1
            all_ids.add(did)
        rows.append(v)
    out = {"shards": rows, "shards_n": len(rows),
           "records": sum(r["records"] for r in rows),
           "ids_unique_sum": sum(r["ids_unique"] for r in rows),
           "ids_unique_global": len(all_ids),
           "global_duplicate_ids": dupes,
           "all_uniques_ok": all(r["uniques_ok"] for r in rows) if rows else False,
           "all_ok": (all(r["ok"] for r in rows) if rows else False) and dupes == 0}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
