"""Whole-doc BM25 (title + text, lowercase syllables, no chunking), to test whether the organizers' baseline is doc-level.

    python bm25doc.py <docs dirs, comma-separated> <out dir> <n_docs ...>
Pass 1 hashes every doc file into a term-count matrix (saved per file) and collects df and doc lengths; pass 2 scores the
1,200 queries against each file with BM25 (k1 1.5, b 0.75) and keeps the top 200 docs per query. Submissions use the
doc text (first 6,000 chars) as the chunk.
"""
import json
import pickle
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

srcs, out = [Path(x) for x in sys.argv[1].split(",")], Path(sys.argv[2])
n_list = [int(x) for x in sys.argv[3:]]
(out / "tf").mkdir(parents=True, exist_ok=True)
K1, B, TOP, NF = 1.5, 0.75, 200, 1 << 23


def prep(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


hv = HashingVectorizer(n_features=NF, token_pattern=r"(?u)\b\w+\b", lowercase=False, preprocessor=prep,
                       alternate_sign=False, norm=None, dtype=np.float32)
files = [f for src in srcs for f in sorted(src.glob("*.parquet"))]
df = np.zeros(NF, np.int64)
ids, dl = [], []
for f in files:
    dst = out / "tf" / (f.parent.name + "_" + f.stem + ".npz")
    pf = pq.ParquetFile(f)
    parts, fid = [], []
    for b in pf.iter_batches(batch_size=20000, columns=["id", "title", "text"]):
        d = b.to_pydict()
        parts.append(hv.transform([f"{t or ''}\n{x or ''}" for t, x in zip(d["title"], d["text"])]).tocsr())
        fid += d["id"]
    m = sp.vstack(parts).tocsr()
    sp.save_npz(dst, m, compressed=False)
    df += np.bincount(m.indices, minlength=NF)
    ids.append(np.array(fid, np.int64))
    dl.append(np.asarray(m.sum(1)).ravel())
    print(f.name, m.shape[0], flush=True)
N = sum(len(x) for x in ids)
avgdl = float(np.concatenate(dl).mean())
idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)

qt = pq.read_table("D:/r2ai_data/raw/query.parquet").to_pydict()
qids = qt["id"]
Q = hv.transform(qt["query"]).tocsr()
Q.data[:] = 1  # each distinct query term counts once
Q = sp.csr_matrix(Q.multiply(idf.reshape(1, -1))).T.tocsr()  # terms x queries, idf-weighted

best_s = np.zeros((len(qids), 0), np.float32)
best_i = np.zeros((len(qids), 0), np.int64)
for f, fid, fdl in zip(files, ids, dl):
    m = sp.load_npz(out / "tf" / (f.parent.name + "_" + f.stem + ".npz")).tocsr()
    norm = (K1 * (1 - B + B * fdl / avgdl)).astype(np.float32)
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    m.data = (m.data * (K1 + 1) / (m.data + norm[rows])).astype(np.float32)
    del rows
    for a in range(0, m.shape[0], 40000):  # 40K docs x 1,200 queries at a time keeps the dense block near 200MB
        s = (m[a:a + 40000] @ Q).toarray().T  # queries x docs
        k = min(TOP, s.shape[1])
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        best_s = np.concatenate([best_s, np.take_along_axis(s, top, 1)], 1)
        best_i = np.concatenate([best_i, fid[a:a + 40000][top]], 1)
        keep = np.argsort(-best_s, 1)[:, :TOP]
        best_s, best_i = np.take_along_axis(best_s, keep, 1), np.take_along_axis(best_i, keep, 1)
    print("scored", f.name, flush=True)
pickle.dump({"qids": qids, "doc": best_i, "score": best_s}, open(out / "bm25doc.pkl", "wb"))

need = set(best_i[:, :max(n_list)].ravel().tolist())
text = {}
for f in files:
    t = pq.read_table(f, columns=["id", "title", "text"]).to_pydict()
    for i, ti, x in zip(t["id"], t["title"], t["text"]):
        if i in need:
            text[i] = x or ti or ""
for n in n_list:
    sub = [{"id": int(q), "relevant_docs": [int(d) for d in best_i[j, :n]],
            "relevant_chunks": [{"doc_id": int(d), "chunk_text": text[int(d)][:6000]} for d in best_i[j, :n]]}
           for j, q in enumerate(qids)]
    js = out / f"bm25doc_d{n}.json"
    js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(js.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(js, arcname=js.name)
    print(js.with_suffix(".zip"), f"{js.with_suffix('.zip').stat().st_size / 1e6:.1f}MB", flush=True)
