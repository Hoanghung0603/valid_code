"""Pool for rerank_more.py from bm25w.pkl: docs in the top `n` of any bm25w list that have no rerank score yet
(not in rerank.pkl or rerank_more.pkl).  python pool_bm25w.py <n> <out.pkl> <chunk dirs in run order...>"""
import pickle, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
n, outp = int(sys.argv[1]), sys.argv[2]
files = [f for d in sys.argv[3:] for f in sorted(Path(d).glob("*.parquet"))]
doc = pq.read_table(files, columns=["doc_id"]).column("doc_id").to_numpy()
scored = {}
for p in ["runs/sub3/rerank.pkl", "runs/sub3/rerank_more.pkl"]:
    r = pickle.load(open(p, "rb")); r = r.get("scores", r)
    for q, v in r.items():
        s = scored.setdefault(int(q), set())
        for sc in v.values():
            s.update(int(doc[i]) for i in sc)
bw = pickle.load(open("runs/sub3/bm25w.pkl", "rb"))
pool = {}
for j, q in enumerate(bw["qids"]):
    new = []
    for name in ("vi_w", "vi_bg", "en", "zh"):
        if name in bw:
            new += [int(d) for d in bw[name][0][j, :n] if int(d) not in scored.get(int(q), ())]
    pool[int(q)] = {d: [] for d in dict.fromkeys(new)}
pickle.dump(pool, open(outp, "wb"))
print("pool docs/query", sum(len(v) for v in pool.values()) / len(pool), flush=True)
