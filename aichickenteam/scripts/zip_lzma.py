"""ZIP_LZMA với từ điển lớn (mặc định zipfile chỉ preset 6 / 8MB): chunk của cùng doc lặp lại ở nhiều câu → từ điển 512MB nén được.
Header LZMA ghi props (dict size) nên mọi bộ giải nén chuẩn (Python zipfile, 7z) đều đọc được.
    python scripts/zip_lzma.py in.zip out.zip   (hoặc import write_zip)"""
import lzma
import struct
import sys
import zipfile

DICT = int(__import__("os").environ.get("R2AI_LZMA_DICT_MB", "512")) << 20


def _init(self):
    f = {"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": DICT}
    if DICT > (512 << 20) or __import__("os").environ.get("R2AI_LZMA_HC4"):   # bt4 tốn ~11.5×dict RAM; hc4 ~7.5×
        f["mf"] = lzma.MF_HC4
    props = lzma._encode_filter_properties(f)
    self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[f])
    return struct.pack("<BBH", 9, 4, len(props)) + props


zipfile.LZMACompressor._init = _init


def write_zip(out, data: bytes, name="submission.json"):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_LZMA) as z:
        z.writestr(name, data)


def reorder(raw: bytes) -> bytes:
    """Xếp các câu có chunk trùng nhau cạnh nhau (tham lam theo 40 câu gần nhất) → từ điển LZMA bắt được lặp; nội dung từng câu giữ nguyên."""
    import json
    from collections import deque
    d = json.loads(raw)
    H = [set(hash(c["chunk_text"]) for c in q["relevant_chunks"]) for q in d]
    order, left, recent = [0], set(range(1, len(d))), deque([0], maxlen=40)
    while left:
        U = set().union(*(H[i] for i in recent))
        j = max(left, key=lambda i: len(H[i] & U)); order.append(j); left.remove(j); recent.append(j)
    return json.dumps([d[i] for i in order], ensure_ascii=False).encode()


if __name__ == "__main__":
    d = zipfile.ZipFile(sys.argv[1]).read("submission.json")
    if __import__("os").environ.get("R2AI_REORDER"):
        d = reorder(d)
    write_zip(sys.argv[2], d)
    assert zipfile.ZipFile(sys.argv[2]).read("submission.json") == d
    print("ok", len(d) >> 20, "MB raw")
