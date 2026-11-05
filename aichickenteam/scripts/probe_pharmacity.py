"""Bài THU HOẠCH NHÃN cho doc Pharmacity (cào lại 20/10): mỗi câu PUBLIC nộp --n doc Pharmacity CHƯA có nhãn có điểm GN doc-level (d1_new) cao nhất,
chunk = cửa sổ d3_new (nếu có) nới 8000 ký tự, ngược lại text[:8000]. Câu private: rỗng. Mục đích: đo mật độ gold của Pharmacity theo hạng
và cho LTR (dom_rate, ce/gn) nhãn của miền này.
    python scripts/probe_pharmacity.py --n 40 --out data/cache/tmp/v66ph.zip
"""
import argparse, json, sys, zipfile
import ijson
import pandas as pd
sys.path.insert(0, "scripts")
from fuse_dense import expand  # noqa: E402
from ltr import labels  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", required=True)
ap.add_argument("--kout", default="data/cache/newdocs_out_ph"); a = ap.parse_args()
_, seen = labels()
T = pd.read_parquet("data/docs/corpus-96000.parquet"); T["doc_id"] = T.doc_id.astype(str); TX = dict(zip(T.doc_id, T.text.fillna("")))
d1 = json.load(open(f"{a.kout}/d1_new.json"))
W = {}
with open(f"{a.kout}/d3_new.json", "rb") as f:
    for q, lst in ijson.kvitems(f, "", use_float=True):
        if int(q) in seen:
            W.update({(int(q), str(x[1])): x[2] for x in lst if len(x) > 2})
res, n = [], 0
for q in sorted(int(x) for x in pd.read_parquet("data/ViBioMIR/query.parquet").id):
    if q not in seen:
        res.append({"id": q, "relevant_docs": [], "relevant_chunks": []}); continue
    docs = [str(x[1]) for x in sorted(d1.get(str(q), []), key=lambda x: -x[0]) if str(x[1]) in TX and str(x[1]) not in seen[q]][:a.n]
    ch = []
    for d in docs:
        c = W.get((q, d)) or TX[d][:8000]
        if len(c) < 8000: c = expand(TX[d], c, 8000)
        ch.append({"doc_id": int(d), "chunk_text": c})
    n += len(ch); res.append({"id": q, "relevant_docs": [int(d) for d in docs], "relevant_chunks": ch})
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.writestr("submission.json", json.dumps(res, ensure_ascii=False))
print("wrote", a.out, "chunks", n, "câu có doc", sum(1 for r in res if r["relevant_docs"]))
