"""Re-zip a submission with LZMA and a big dictionary, so chunk texts repeated across queries (the same doc in many
queries' lists) compress to almost nothing. Optionally sorts queries so ones sharing docs sit closer.
    python lzbig.py <in zip> <out zip> [dict MB]"""
import json, lzma, sys, time, zipfile, os
dict_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 256
z = zipfile.ZipFile(sys.argv[1]); name = z.namelist()[0]; raw = z.read(name)
if os.environ.get("ORDER"):  # greedy chain of queries by shared chunk docs, so repeated texts sit close together
    sub = json.loads(raw)
    sets = [set(c["doc_id"] for c in s["relevant_chunks"]) for s in sub]
    left, order = set(range(1, len(sub))), [0]
    while left:
        cur = sets[order[-1]]
        nxt = max(left, key=lambda j: len(cur & sets[j]))
        order.append(nxt); left.remove(nxt)
    raw = json.dumps([sub[i] for i in order], ensure_ascii=False).encode("utf-8")


class Big(zipfile.LZMACompressor):
    def _init(self):
        props = lzma._encode_filter_properties({"id": lzma.FILTER_LZMA1, "dict_size": dict_mb << 20, "preset": 9})
        self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[
            lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)])
        return __import__("struct").pack("<BBH", 9, 4, len(props)) + props


zipfile.LZMACompressor = Big
t = time.time()
with zipfile.ZipFile(sys.argv[2], "w", zipfile.ZIP_LZMA) as out:
    out.writestr(name, raw)
print(f"raw {len(raw) / 1e6:.0f}MB -> {os.path.getsize(sys.argv[2]) / 1e6:.1f}MB in {time.time() - t:.0f}s")
back = zipfile.ZipFile(sys.argv[2]).read(name)
assert back == raw
print("round trip ok")
