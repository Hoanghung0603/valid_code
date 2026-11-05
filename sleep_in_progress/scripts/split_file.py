"""Tách file lớn thành các phần <= --part-mb MB (để vượt giới hạn SFTP 100MiB/file) + in sha256 toàn file.

Ghép lại: nối các phần theo thứ tự .part000, .part001, ... rồi so sha256 với dòng in ra.
  python scripts/split_file.py --file submission_d600k120.zip --part-mb 80
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BUF = 1 << 20  # đọc theo khối 1MiB, không nạp cả file vào RAM


def split_file(path: Path, part_bytes: int) -> tuple[list[Path], str]:
    """Ghi path.partNNN, mỗi phần <= part_bytes. Trả (danh sách phần, sha256 hex của file gốc)."""
    if part_bytes <= 0:
        raise ValueError("part_bytes phải > 0")
    h = hashlib.sha256()
    parts: list[Path] = []
    with path.open("rb") as src:
        idx = 0
        while True:
            written = 0
            out_path = path.with_name(f"{path.name}.part{idx:03d}")
            out = None
            while written < part_bytes:
                buf = src.read(min(BUF, part_bytes - written))
                if not buf:
                    break
                if out is None:
                    out = out_path.open("wb")
                out.write(buf)
                h.update(buf)
                written += len(buf)
            if out is None:
                break
            out.close()
            parts.append(out_path)
            idx += 1
            if written < part_bytes:
                break
    return parts, h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--part-mb", type=int, default=80, help="kích thước tối đa mỗi phần (MiB)")
    a = ap.parse_args()
    parts, digest = split_file(Path(a.file), a.part_mb << 20)
    for p in parts:
        print(f"{p} {p.stat().st_size}", flush=True)
    print(f"sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()
