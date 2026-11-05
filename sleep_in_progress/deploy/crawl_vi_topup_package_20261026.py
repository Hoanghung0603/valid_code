"""Verify + build handoff_vi_topup_20261026 package from vi_topup shards.

Steps (READ-ONLY on sources; writes ONLY the new package dir):
 1. strict re-scan of every vi-topup-*.jsonl.zst shard (fail -> abort);
 2. scan_contamination on every shard (fail -> abort);
 3. internal dedup check by path_key(final_url) + id uniqueness;
 4. cross-package dup check vs handoff_vi_20261025_full (informational);
 5. package = copy shards + manifest.sha256 + summary.json.

Exit 0 built; 1 verification failed; 2 package dir already exists.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"D:\r2ai\data\crawl\tools")
import crawl_async_supervisor as sup  # noqa: E402
import scan_contamination as sc  # noqa: E402

TOPUP = Path(r"D:\r2ai\data\crawl\vi_topup_20261026\pages")
FULL = Path(r"D:\r2ai\data\crawl\handoff_vi_20261025_full")
PKG = Path(r"D:\r2ai\data\crawl\handoff_vi_topup_20261026")

srcs = sorted(TOPUP.glob("vi-topup-*.jsonl.zst"))
if not srcs:
    print("NO_SHARDS")
    sys.exit(1)

# 1+2+3 verify + dedup inventory
chosen = {}
ids_seen = set()
n_rec = 0
per_host = {}
verify = []
for p in srcs:
    v = sup.strict_scan_shard(p)
    v.pop("ids", None)
    scan = sc.scan(p)
    susp = scan["counts"]["suspects"]
    verify.append({"file": p.name, "records": v["records"],
                   "ids_unique": v["ids_unique"],
                   "uniques_ok": v["uniques_ok"], "frames": v["frames"],
                   "strict_eof": v["strict_eof"], "schema_bad": v["schema_bad"],
                   "content_bad": v["content_bad"], "challenge": v["challenge"],
                   "malformed": v["malformed"], "sha256": v["sha256"],
                   "size": v["size"], "ok": v["ok"],
                   "contamination_suspects": susp})
    if not v["ok"] or susp:
        print("VERIFY_FAIL", p.name, json.dumps(v), "suspects", susp)
        sys.exit(1)
    for rec in sup.iter_jsonl_zst(p):
        n_rec += 1
        rid = int(rec["id"])
        ids_seen.add(rid)
        h = str(rec.get("host") or "?")
        per_host[h] = per_host.get(h, 0) + 1
        key = sc.path_key(rec.get("final_url") or rec.get("url") or "")
        chosen.setdefault(key, rid)

internal_dup_fin = n_rec - len(chosen)
internal_dup_id = n_rec - len(ids_seen)

# 4 cross-package dup check vs full package 25/10
old = {}
for p in sorted(FULL.glob("vi-recov-*.jsonl.zst")):
    for rec in sup.iter_jsonl_zst(p):
        k = sc.path_key(rec.get("final_url") or rec.get("url") or "")
        old.setdefault(k, int(rec["id"]))
cross = [(k, chosen[k], old[k]) for k in chosen if k in old]

# 5 build
if PKG.exists() and any(PKG.glob("*.jsonl.zst")):
    print("PKG_EXISTS", str(PKG))
    sys.exit(2)
PKG.mkdir(parents=True, exist_ok=True)
manifest = []
for p in srcs:
    dst = PKG / p.name
    shutil.copyfile(p, dst)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    manifest.append(f"{h}  {p.name}")
(PKG / "manifest.sha256").write_text("\n".join(manifest) + "\n",
                                     encoding="utf-8")
summary = {
    "at": sup.time.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(sup, "time") else "",
    "package": str(PKG),
    "supersedes": [],
    "complements": "handoff_vi_20261025_full (ids disjoint; both current)",
    "source": {
        "batch": "vi_topup_20261026",
        "shards": [p.name for p in srcs],
        "selection": {"targets": 2095, "ok": 1793},
        "method": "nodriver fresh browser + rotated Webshare proxy, schema-9",
    },
    "vi": {"records": n_rec, "ids_unique": len(ids_seen),
           "final_url_unique": len(chosen),
           "dup_final_internal": internal_dup_fin,
           "dup_id_internal": internal_dup_id,
           "hosts": per_host},
    "cross_dup_vs_20261025_full": [{"key": k, "topup_id": a, "full_id": b}
                                   for k, a, b in cross],
    "verify": verify,
}
(PKG / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                             indent=1),
                                  encoding="utf-8")
print("PKG_BUILT", str(PKG))
print("RECORDS", n_rec, "UNIQ_ID", len(ids_seen),
      "UNIQ_FINAL", len(chosen))
print("INTERNAL_DUP_FIN", internal_dup_fin, "INTERNAL_DUP_ID",
      internal_dup_id)
print("CROSS_DUP_COUNT", len(cross))
print("CROSS_DUP_SAMPLE", json.dumps(cross[:30], ensure_ascii=False))
print("PER_HOST", json.dumps(per_host, ensure_ascii=False))
