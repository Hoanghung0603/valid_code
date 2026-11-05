"""Kiểm tra shard đã chuyển: sha256 khớp manifest, và (tùy chọn) đọc thử bằng shards.read_shard.

Dùng: python scripts/verify_shards.py <thư mục shard> <manifest.sha256> [--read] [--only-present]
manifest: mỗi dòng '<sha256>  <tên file>' (định dạng của sha256sum).
--only-present: bỏ qua file có trong manifest nhưng chưa có trong thư mục (dùng khi chuyển từng lô).
Thoát mã 1 nếu có file sai hash hoặc thiếu (khi không có --only-present).
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shards import read_shard  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir")
    ap.add_argument("manifest")
    ap.add_argument("--read", action="store_true", help="đọc thử từng shard, đếm bản ghi")
    ap.add_argument("--only-present", action="store_true")
    a = ap.parse_args()

    d = Path(a.shard_dir)
    expected = {}
    for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[name.strip()] = digest.lower()

    ok = bad = missing = records = 0
    for name, digest in sorted(expected.items()):
        p = d / name
        if not p.exists():
            missing += 1
            if not a.only_present:
                print(f"THIEU {name}")
            continue
        if sha256_of(p) != digest:
            bad += 1
            print(f"SAI_HASH {name}")
            continue
        ok += 1
        if a.read:
            records += sum(1 for _ in read_shard(p))
    print(f"ok={ok} sai_hash={bad} thieu={missing}" + (f" ban_ghi={records}" if a.read else ""))
    sys.exit(1 if bad or (missing and not a.only_present) else 0)


if __name__ == "__main__":
    main()
