"""Check that two submission zips hold the same answers (docs per query, and chunk texts), whatever the compression or
query order.   python same_submission.py <a.zip> <b.zip>"""
import json
import sys
import zipfile


def load(p):
    z = zipfile.ZipFile(p)
    return {s["id"]: s for s in json.loads(z.read(z.namelist()[0]))}


a, b = load(sys.argv[1]), load(sys.argv[2])
assert a.keys() == b.keys(), "different query ids"
docs = sum(a[q]["relevant_docs"] == b[q]["relevant_docs"] for q in a)
chunks = sum(a[q]["relevant_chunks"] == b[q]["relevant_chunks"] for q in a)
print(f"{len(a)} queries: same docs {docs}, same chunks {chunks}")
sys.exit(0 if docs == chunks == len(a) else 1)
