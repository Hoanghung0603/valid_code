"""On the lab: top n docs per query of the current bge ranking (rerank + extra score pkls, zh/en minus pen), each with
its best chunk.   python pool_best_lab.py <n> <pen> <out.pkl> <chunk dirs> -- <rerank pkls...>"""
import pickle, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
a = sys.argv.index("--")
n, pen, outp, cdirs, pk = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4:a], sys.argv[a + 1:]
files = [f for d in cdirs for f in sorted(Path(d).glob("*.parquet"))]
m = pq.read_table(files, columns=["doc_id", "lang"])
doc = m.column("doc_id").to_numpy(); zh = np.array([x != "vi" for x in m.column("lang").to_pylist()])
rr = {}
for p in pk:
    r = pickle.load(open(p, "rb")); r = r.get("scores", r)
    for q, v in r.items():
        for ql, sc in v.items():
            rr.setdefault(q, {}).setdefault(ql, {}).update(sc)
pool = {}
for q, r in rr.items():
    s = dict(r.get("vi", {}))
    for i, z in r.get("t", {}).items():
        s[i] = max(s.get(i, z), z)
    best = {}
    for i, z in s.items():
        d = int(doc[i]); z -= pen if zh[i] else 0
        if d not in best or z > best[d][0]:
            best[d] = (z, i)
    pool[int(q)] = {d: [best[d][1]] for d in sorted(best, key=lambda d: -best[d][0])[:n]}
pickle.dump(pool, open(outp, "wb"))
print("pool", len(pool), np.mean([len(v) for v in pool.values()]), flush=True)
