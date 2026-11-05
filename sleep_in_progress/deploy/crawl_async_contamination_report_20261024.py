"""Merge stale-tab contamination scan reports into one auditable report."""
import json
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(r"D:\r2ai\data\crawl\async_recovery_20261009")
AUDIT = BASE / "audit"
REPORTS = [
    "contamination_scan_00000-00004.json",
    "contamination_scan_00005-00012.json",
    "contamination_scan_00013-00019.json",
    "contamination_scan_00020-00026.json",
    "contamination_scan_00027-00034.json",
    "contamination_scan_00035-00039.json",
    "contamination_scan_recov-00040-recov-00043.json",
]
OUT_JSON = AUDIT / "contamination_all_20261024.json"
OUT_TSV = AUDIT / "contamination_bad_ids_20261024.tsv"
OUT_SKIPPED_TSV = AUDIT / "contamination_false_positive_20261024.tsv"
SKIP_FILES = {"recov-00006.jsonl.zst", "recov-00007.jsonl.zst"}

GROUPS = {
    "handoff_20261017": range(0, 33),
    "handoff_20261021": range(33, 35),
    "handoff_20261023": range(35, 40),
    "unhanded_20261024": range(40, 44),
}


def shard_index(name):
    stem = name.split(".")[0]
    try:
        return int(stem.split("-")[1])
    except Exception:
        return None


def host_of(url):
    try:
        return (urlparse(url or "").hostname or "").lower()
    except Exception:
        return ""


def is_vi_host(host):
    return host.endswith(".vn") or host in {"vinmec.com", "www.vinmec.com"}


def main():
    entries = {}
    reports = []
    for name in REPORTS:
        path = AUDIT / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append({"file": name, "shards": len(data)})
        for entry in data:
            key = entry["file"]
            if key in entries:
                old = entries[key]
                old["suspects"].extend(entry.get("suspects") or [])
                for ck, cv in (entry.get("counts") or {}).items():
                    old["counts"][ck] = old["counts"].get(ck, 0) + cv
            else:
                entries[key] = {
                    "file": key,
                    "records": entry.get("records", 0),
                    "counts": dict(entry.get("counts") or {}),
                    "suspects": list(entry.get("suspects") or []),
                }

    shards = []
    for name in sorted(entries):
        e = entries[name]
        by_id = {}
        for s in e["suspects"]:
            rid = int(s["id"])
            if rid not in by_id:
                by_id[rid] = {
                    "id": rid,
                    "url": s.get("url") or "",
                    "final_url": s.get("final_url") or "",
                    "text_len": s.get("text_len", 0),
                    "reasons": [],
                }
            for reason in s.get("reasons") or []:
                if reason not in by_id[rid]["reasons"]:
                    by_id[rid]["reasons"].append(reason)
        suspects = [by_id[k] for k in sorted(by_id)]
        reasons = Counter(r for s in suspects for r in s["reasons"])
        hosts = Counter(host_of(s["url"]) for s in suspects)
        vi_hosts = Counter(h for h in hosts if is_vi_host(h))
        shards.append({
            "file": name,
            "index": shard_index(name),
            "records": e["records"],
            "suspects": suspects,
            "suspect_count": len(suspects),
            "reasons": dict(reasons),
            "hosts": dict(hosts.most_common()),
            "suspects_vi_host": sum(vi_hosts.values()),
            "suspects_non_vi_host": len(suspects) - sum(vi_hosts.values()),
        })

    totals = {
        "records": sum(s["records"] for s in shards),
        "suspects": sum(s["suspect_count"] for s in shards),
        "reasons": dict(Counter(
            r for s in shards for r in s["suspects"] for r in s["reasons"])),
        "suspects_vi_host": sum(s["suspects_vi_host"] for s in shards),
        "suspects_non_vi_host": sum(s["suspects_non_vi_host"] for s in shards),
        "shards_with_suspects": sum(1 for s in shards if s["suspect_count"]),
    }
    groups = {}
    for tag, idxs in GROUPS.items():
        idx_set = set(idxs)
        g = [s for s in shards if s["index"] in idx_set]
        groups[tag] = {
            "shards": len(g),
            "records": sum(s["records"] for s in g),
            "suspects": sum(s["suspect_count"] for s in g),
            "suspects_vi_host": sum(s["suspects_vi_host"] for s in g),
            "suspects_non_vi_host": sum(s["suspects_non_vi_host"] for s in g),
        }

    out = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reports": reports,
        "totals": totals,
        "groups": groups,
        "shards": shards,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    with OUT_TSV.open("w", encoding="utf-8", newline="") as fh:
        fh.write("id\tshard\treasons\turl\tfinal_url\n")
        for s in shards:
            if s["file"] in SKIP_FILES:
                continue
            for bad in s["suspects"]:
                url = bad["url"].replace("\t", " ").replace("\n", " ")
                final = bad["final_url"].replace("\t", " ").replace("\n", " ")
                fh.write(f"{bad['id']}\t{s['file']}\t"
                         f"{','.join(bad['reasons'])}\t{url}\t{final}\n")
        fh.flush()
    with OUT_SKIPPED_TSV.open("w", encoding="utf-8", newline="") as fh:
        fh.write("id\tshard\treasons\turl\tfinal_url\n")
        for s in shards:
            if s["file"] not in SKIP_FILES:
                continue
            for bad in s["suspects"]:
                url = bad["url"].replace("\t", " ").replace("\n", " ")
                final = bad["final_url"].replace("\t", " ").replace("\n", " ")
                fh.write(f"{bad['id']}\t{s['file']}\t"
                         f"{','.join(bad['reasons'])}\t{url}\t{final}\n")
        fh.flush()
    print(json.dumps({"json": str(OUT_JSON), "tsv": str(OUT_TSV),
                      "false_positive_tsv": str(OUT_SKIPPED_TSV),
                      "totals": totals, "groups": groups},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
