"""Topic feedback: each doc's vector is the mean of its bge-m3 chunk vectors; for each query the centroid of its top K
pooled docs (pool order = current ranking) scores every pooled doc by cosine. CPU only.

    python prf.py <pool.pkl> <out.pkl> <K list, comma-separated> <chunk dirs> -- <index dirs>
out.pkl: {"K": {qid: {doc: cosine}}}
"""
import pickle, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
a = sys.argv.index("--")
pool = pickle.loads(Path(sys.argv[1]).read_bytes())
outp, Ks = Path(sys.argv[2]), [int(k) for k in sys.argv[3].split(",")]
cdirs, idirs = sys.argv[4:a], sys.argv[a + 1:]
files = [f for d in cdirs for f in sorted(Path(d).glob("*.parquet"))]
doc_of = pq.read_table(files, columns=["doc_id"]).column("doc_id").to_numpy()
mats = [np.load(Path(d) / "dense.npy", mmap_mode="r") for d in idirs]
offs = np.cumsum([0] + [m.shape[0] for m in mats])
assert offs[-1] == len(doc_of), (offs[-1], len(doc_of))
need = np.array(sorted({d for v in pool.values() for d in v}), np.int64)
rows = np.flatnonzero(np.isin(doc_of, need))
print(f"{len(need)} docs, {len(rows)} chunks", flush=True)
t0 = time.time()
dv = {}
for mi, m in enumerate(mats):
    r = rows[(rows >= offs[mi]) & (rows < offs[mi + 1])]
    for s in range(0, len(r), 200000):
        rr = r[s:s + 200000]
        v = np.asarray(m[rr - offs[mi]], dtype=np.float32)
        for d, x in zip(doc_of[rr], v):
            acc = dv.get(int(d))
            dv[int(d)] = x.copy() if acc is None else acc + x
print(f"doc vectors in {time.time() - t0:.0f}s", flush=True)
for d in dv:
    dv[d] /= np.linalg.norm(dv[d]) + 1e-9
out = {k: {} for k in Ks}
for q, v in pool.items():
    docs = [d for d in v if d in dv]
    M = np.stack([dv[d] for d in docs])
    for k in Ks:
        c = M[:k].mean(0); c /= np.linalg.norm(c) + 1e-9
        out[k][q] = dict(zip(docs, (M @ c).tolist()))
pickle.dump(out, open(outp, "wb"))
print("PRF_DONE", flush=True)
