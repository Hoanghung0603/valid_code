"""Đóng gói bài nộp vào zip (đúng 1 entry 'submission.json') + in thống kê.

Mặc định ZIP_DEFLATED level 9 như cũ. --method lzma: ZIP_LZMA (nhỏ hơn, hệ chấm #5468 đọc được;
--level là preset lzma nếu truyền, mặc định preset thư viện). Dùng:
  python scripts/zip_submission.py --json submission_d600k60.json --zip submission_d600k60.zip
  python scripts/zip_submission.py --json submission_c120.json --zip submission_c120.zip --method lzma
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

METHODS = {"deflate": zipfile.ZIP_DEFLATED, "lzma": zipfile.ZIP_LZMA,
           "bzip2": zipfile.ZIP_BZIP2, "stored": zipfile.ZIP_STORED}
ENTRY = "submission.json"


def pack(json_path, zip_path, method: str = "deflate", level: int | None = None) -> dict:
    """Nén json_path vào zip_path (1 entry submission.json). Trả thống kê."""
    if method not in METHODS:
        raise ValueError(f"method la: {method}")
    p, zp = Path(json_path), Path(zip_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    n = max(len(data), 1)
    nd = [len(s.get("relevant_docs", [])) for s in data]
    nc = [len(s.get("relevant_chunks", [])) for s in data]
    lvl = 9 if (level is None and method == "deflate") else level
    with zipfile.ZipFile(zp, "w", METHODS[method], compresslevel=lvl) as z:
        z.write(p, arcname=ENTRY)
    return {"method": method, "cau": len(data), "doc_cau": sum(nd) / n, "chunk_cau": sum(nc) / n,
            "json_bytes": p.stat().st_size, "zip_bytes": zp.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="file submission.json")
    ap.add_argument("--zip", required=True, help="zip đầu ra (chỉ chứa submission.json)")
    ap.add_argument("--method", choices=sorted(METHODS), default="deflate",
                    help="cách nén (mặc định deflate = ZIP_DEFLATED như cũ)")
    ap.add_argument("--level", type=int, default=None,
                    help="mức nén deflate 0-9 (mặc định 9) hoặc preset lzma 0-9 (mặc định preset thư viện)")
    a = ap.parse_args()
    st = pack(a.json, a.zip, a.method, a.level)
    print(f"{a.zip}: {st['cau']} câu, TB doc/câu {st['doc_cau']:.2f}, TB chunk/câu {st['chunk_cau']:.2f}, "
          f"JSON {st['json_bytes'] / 1e6:.1f}MB, zip {st['zip_bytes'] / 1e6:.1f}MB ({a.method})", flush=True)


if __name__ == "__main__":
    main()
