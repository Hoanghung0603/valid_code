"""One probe = relevant_docs of a doc variant + relevant_chunks of a chunk variant (the two levels are scored apart).
    python merge_combo.py <docs zip> <chunks zip> <out zip>"""
import json, sys, zipfile
def load(p):
    z = zipfile.ZipFile(p); return {s["id"]: s for s in json.loads(z.read(z.namelist()[0]))}
d, c = load(sys.argv[1]), load(sys.argv[2])
assert d.keys() == c.keys() and len(d) == 1200
sub = [{"id": q, "relevant_docs": d[q]["relevant_docs"], "relevant_chunks": c[q]["relevant_chunks"]} for q in sorted(d)]
name = sys.argv[3].replace("\\", "/").rsplit("/", 1)[1][:-4]
with zipfile.ZipFile(sys.argv[3], "w", zipfile.ZIP_BZIP2, compresslevel=9) as z:
    z.writestr(name + ".json", json.dumps(sub, ensure_ascii=False))
import os; print(name, round(os.path.getsize(sys.argv[3]) / 1e6, 1), "MB")
