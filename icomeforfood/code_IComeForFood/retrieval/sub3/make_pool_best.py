"""Pool for vnrr.py: the top n docs per query of the current ranking (rerank + extra scores, zh/en minus pen), each
with its best chunk.  python make_pool_best.py <n> <pen> <out.pkl> <extra pkls...>"""
import pickle, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
n, pen, outp = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
files = [f for d in ["D:/r2ai_data/chunks_sub2/s400", "D:/r2ai_data/chunks_sub3add/s400", "D:/r2ai_data/chunks_sub4add/s400"]
         for f in sorted(Path(d).glob("*.parquet"))]
m = pq.read_table(files, columns=["doc_id", "lang"])
doc = m.column("doc_id").to_numpy(); zh = np.array([x != "vi" for x in m.column("lang").to_pylist()])
rr = pickle.load(open("D:/r2ai_data/runs/sub3/rerank.pkl", "rb"))["scores"]
for p in sys.argv[4:]:
    for q, r in pickle.load(open(p, "rb")).items():
        for ql, sc in r.items():
            rr.setdefault(q, {}).setdefault(ql, {}).update(sc)
pool = {}
for q, r in rr.items():
    s = dict(r["vi"])
    for i, z in r.get("t", {}).items():
        s[i] = max(s.get(i, z), z)
    best = {}
    for i, z in s.items():
        d = int(doc[i]); z -= pen if zh[i] else 0
        if d not in best or z > best[d][0]:
            best[d] = (z, i)
    pool[int(q)] = {d: [best[d][1]] for d in sorted(best, key=lambda d: -best[d][0])[:n]}
pickle.dump(pool, open(outp, "wb"))
print(len(pool), np.mean([len(v) for v in pool.values()]))
