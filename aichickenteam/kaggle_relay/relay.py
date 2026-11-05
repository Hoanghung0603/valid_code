"""Kaggle CPU: PUT file zip bài nộp lên presigned URL của Codabench (mạng Kaggle→BTC nhanh hơn mạng local).
Nếu file chưa phải ZIP_LZMA (vd. zip deflate thô quá 100MB) thì nén lại LZMA preset 9e, từ điển 1GB (CPU Kaggle ~30GB RAM)."""
import glob
import lzma
import os
import struct
import subprocess
import zipfile

src = glob.glob("/kaggle/input/**/sub*.bin", recursive=True)
if src:
    z = src[0]
else:   # Kaggle tự giải nén zip ⇒ có thể chỉ còn submission.json
    z = None
    raw = glob.glob("/kaggle/input/**/submission.json", recursive=True)[0]
url = open(glob.glob("/kaggle/input/**/url.txt", recursive=True)[0]).read().strip()
DICT = 1024 << 20


def _init(self):
    f = {"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": DICT}
    props = lzma._encode_filter_properties(f)
    self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[f])
    return struct.pack("<BBH", 9, 4, len(props)) + props


if z is None or zipfile.ZipFile(z).infolist()[0].compress_type != zipfile.ZIP_LZMA:
    data = open(raw, "rb").read() if z is None else zipfile.ZipFile(z).read("submission.json")
    zipfile.LZMACompressor._init = _init
    z = "/kaggle/working/sub_lzma.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_LZMA) as out:
        out.writestr("submission.json", data)
    assert zipfile.ZipFile(z).read("submission.json") == data
    print("recompressed", len(data) >> 20, "MB raw ->", os.path.getsize(z) >> 20, "MB", flush=True)
if os.path.getsize(z) >= 100 * 10**6:
    open("/kaggle/working/put_result.txt", "w").write(f"TOO_BIG {os.path.getsize(z)}")
    raise SystemExit("zip >= 100MB")
r = subprocess.run(["curl", "-sS", "-T", z, "-H", "Content-Type: application/zip", "-o", "/dev/null", "-w", "%{http_code} %{speed_upload}", url],
                   capture_output=True, text=True)
open("/kaggle/working/put_result.txt", "w").write(r.stdout + " " + r.stderr[:200] + f" size={os.path.getsize(z)} src={os.path.basename(src[0]) if src else 'raw'}")
print(r.stdout, r.stderr[:200])
if os.path.exists("/kaggle/working/sub_lzma.zip"):
    os.remove("/kaggle/working/sub_lzma.zip")
