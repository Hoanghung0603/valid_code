"""Tạo manifest 'healthy' cho bàn giao: loại shard lỗi khỏi verify_manifest.sha256 và kiểm chéo số liệu.

Chỉ đọc verify_manifest.sha256 + verify_shards.json; ghi 1 file manifest mới trong verification/.
Không sửa/di chuyển dữ liệu shard gốc.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"D:\r2ai\data\crawl\verification\verify_manifest.sha256")
    ap.add_argument("--exclude", default="crawl-00009.jsonl.zst")
    ap.add_argument("--shards-json", default=r"D:\r2ai\data\crawl\verification\verify_shards.json")
    ap.add_argument("--out", default=r"D:\r2ai\data\crawl\verification\verify_manifest_healthy.sha256")
    a = ap.parse_args()

    ex = [x.strip() for x in a.exclude.split(",") if x.strip()]
    lines = Path(a.manifest).read_text(encoding="utf-8").splitlines()
    keep, seen_ex = [], []
    for line in lines:
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.strip()
        if name in ex:
            seen_ex.append(name)
            continue
        keep.append((digest, name))
    missing_ex = [x for x in ex if x not in seen_ex]
    if missing_ex:
        print(f"[healthy] LOI: khong thay trong manifest: {missing_ex}")
        return 2

    out = Path(a.out)
    with out.open("w", encoding="utf-8") as f:
        for digest, name in keep:
            f.write(f"{digest}  {name}\n")
    h = hashlib.sha256(out.read_bytes()).hexdigest()

    shards = json.loads(Path(a.shards_json).read_text(encoding="utf-8"))["shards"]
    by_name = {s["name"]: s for s in shards}
    keepset = {n for _, n in keep}
    diff = keepset ^ (set(by_name) - set(ex))
    if diff:
        print(f"[healthy] LOI: lech ten giua manifest va shards json: {sorted(diff)[:5]}")
        return 2
    clean = [by_name[n] for _, n in keep]
    excl = [by_name[x] for x in ex]
    rec = sum(s["records"] for s in clean)
    size = sum(s["size"] for s in clean)
    print(f"[healthy] excluded={[s['name'] for s in excl]} excluded_records={sum(s['records'] for s in excl)}")
    print(f"[healthy] clean_shards={len(clean)} clean_records_sum={rec} clean_bytes_sum={size}")
    print(f"[healthy] first={clean[0]['name']} last={clean[-1]['name']}")
    print(f"[healthy] manifest={out} lines={len(keep)} sha256={h}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
