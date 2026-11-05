"""Candidates among newly added docs only: bge-m3 dense similarity of the Vietnamese query to their chunks, top n docs
per query (by best chunk), written as a rerank_more.py pool.
    python newcands.py <n> <out pool.pkl> <chunk dirs> -- <index dirs>"""
import pickle, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "code/retrieval"))
from pipeline import DenseEncoder  # noqa: E402
a = sys.argv.index("--")
n, outp, cdirs, idirs = int(sys.argv[1]), sys.argv[2], sys.argv[3:a], sys.argv[a + 1:]
doc = pq.read_table([f for d in cdirs for f in sorted(Path(d).glob("*.parquet"))], columns=["doc_id"]).column("doc_id").to_numpy()
M = np.concatenate([np.load(Path(d) / "dense.npy") for d in idirs]).astype(np.float32)
assert len(M) == len(doc)
q = pq.read_table(W / "data/raw/query.parquet", columns=["id", "query"]).to_pydict()
Q = DenseEncoder().encode(q["query"]).astype(np.float32)
S = Q @ M.T
pool = {}
for j, qid in enumerate(q["id"]):
    best = {}
    for i in np.argsort(-S[j])[:n * 6]:
        best.setdefault(int(doc[i]), float(S[j, i]))
    pool[int(qid)] = {d: [] for d in list(best)[:n]}
pickle.dump(pool, open(outp, "wb"))
print("pool docs/query", np.mean([len(v) for v in pool.values()]), flush=True)
