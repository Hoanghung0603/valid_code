"""Pool for rerank_more.py: the top n docs per query of the sub3 ranking (rerank max(vi, t), zh/en minus pen), with the
chunks already scored for each doc.  python make_pool.py <n> <pen> <out.pkl>"""
import pickle, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
n, pen, outp = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
files = [f for d in ["D:/r2ai_data/chunks_sub2/s400", "D:/r2ai_data/chunks_sub3add/s400"] for f in sorted(Path(d).glob("*.parquet"))]
m = pq.read_table(files, columns=["doc_id", "lang"])
doc = m.column("doc_id").to_numpy(); zh = np.array([x != "vi" for x in m.column("lang").to_pylist()])
rr = pickle.load(open("D:/r2ai_data/runs/sub3/rerank.pkl", "rb"))["scores"]
pool = {}
for q, r in rr.items():
    s = dict(r["vi"])
    for i, z in r.get("t", {}).items():
        s[i] = max(s[i], z)
    best, done = {}, {}
    for i, z in s.items():
        d = int(doc[i]); done.setdefault(d, []).append(i)
        z -= pen if zh[i] else 0
        best[d] = max(best.get(d, -1e9), z)
    pool[int(q)] = {d: done[d] for d in sorted(best, key=best.get, reverse=True)[:n]}
pickle.dump(pool, open(outp, "wb"))
print(len(pool), sum(len(v) for v in pool.values()) / len(pool))
