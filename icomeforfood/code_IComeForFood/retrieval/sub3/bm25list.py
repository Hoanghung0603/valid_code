"""Turn the whole-doc BM25 ranking into a cands.pkl list: for each query, its top docs as their best s400 chunk.

    python bm25list.py <bm25doc dir> <out pkl> <chunk dirs in run order...>
The best chunk of a doc is the one with the highest BM25-style score (idf from the doc-level run, saturated tf) for the
query. Chunk indices count across the chunk files in the order the run reads them (dirs as given, files sorted).
"""
import pickle
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

bdir, outp = Path(sys.argv[1]), Path(sys.argv[2])
files = [f for d in sys.argv[3:] for f in sorted(Path(d).glob("*.parquet"))]
NF = 1 << 23


def prep(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").lower()


hv = HashingVectorizer(n_features=NF, token_pattern=r"(?u)\b\w+\b", lowercase=False, preprocessor=prep,
                       alternate_sign=False, norm=None, dtype=np.float32)
df = np.zeros(NF, np.int64)
N = 0
for f in sorted((bdir / "tf").glob("*.npz")):
    m = sp.load_npz(f)
    df += np.bincount(m.indices, minlength=NF)
    N += m.shape[0]
idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)

r = pickle.load(open(bdir / "bm25doc.pkl", "rb"))
qids, rank = r["qids"], r["doc"]
qt = pq.read_table("D:/r2ai_data/raw/query.parquet").to_pydict()
assert qt["id"] == list(qids)
Q = hv.transform(qt["query"]).tocsr()
Q.data[:] = 1
Q = sp.csr_matrix(Q.multiply(idf.reshape(1, -1))).T.tocsr()  # terms x queries

wants: dict[int, list[int]] = {}  # doc -> query rows that rank it
for j in range(len(qids)):
    for d in rank[j].tolist():
        wants.setdefault(int(d), []).append(j)
need = pa.array(sorted(wants), pa.int64())
best: dict[tuple[int, int], tuple[float, int]] = {}  # (query row, doc) -> (score, chunk index)
base = 0
for f in files:
    t = pq.read_table(f, columns=["doc_id", "text"])
    n = t.num_rows
    rows = np.flatnonzero(pc.is_in(t.column("doc_id"), value_set=need).to_numpy(zero_copy_only=False))
    if len(rows):
        sub = t.take(pa.array(rows, pa.int64()))
        docs = sub.column("doc_id").to_numpy()
        texts = sub.column("text").to_pylist()
        for a in range(0, len(rows), 20000):
            m = hv.transform(texts[a:a + 20000]).tocsr()
            m.data = m.data / (m.data + 1.2)
            s = (m @ Q).toarray()
            for k in range(s.shape[0]):
                d = int(docs[a + k])
                ci = base + int(rows[a + k])
                for j in wants[d]:
                    v = float(s[k, j])
                    if (j, d) not in best or v > best[(j, d)][0]:
                        best[(j, d)] = (v, ci)
    base += n
    print(f.name, len(rows), flush=True)
out = {}
miss = 0
for j, q in enumerate(qids):
    L = []
    for d in rank[j].tolist():
        b = best.get((j, int(d)))
        if b is None:
            miss += 1
        else:
            L.append(b[1])
    out[int(q)] = {"vi_bm25doc": np.array(L, np.int32)}
pickle.dump(out, open(outp, "wb"))
print("docs without chunks:", miss, "->", outp)
