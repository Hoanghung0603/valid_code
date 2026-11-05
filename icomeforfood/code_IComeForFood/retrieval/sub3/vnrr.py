"""Score each pooled (query, best chunk) pair with another cross-encoder (default AITeamVN/Vietnamese_Reranker), with
the Vietnamese query.   python vnrr.py <pool.pkl> <out.pkl> <repo> <chunk dirs in run order...>
pool.pkl: {qid: {doc: [chunk index]}}. out.pkl: {qid: {"vi": {chunk: score}}}."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "code/retrieval"))
from pipeline import CrossEncoder, chunk_text  # noqa: E402
pool, outp, repo = pickle.loads(Path(sys.argv[1]).read_bytes()), Path(sys.argv[2]), sys.argv[3]
files = [f for d in sys.argv[4:] for f in sorted(Path(d).glob("*.parquet"))]
need = {i for v in pool.values() for c in v.values() for i in c}
c = {"text": {}, "heading_path": {}}
base = 0
for f in files:
    for b in pq.ParquetFile(f).iter_batches(batch_size=50000, columns=["text", "heading_path"]):
        tx, hd = b.column("text").to_pylist(), b.column("heading_path").to_pylist()
        for j in range(len(tx)):
            if base + j in need:
                c["text"][base + j], c["heading_path"][base + j] = tx[j], hd[j]
        base += len(tx)
qt = pq.read_table(W / "data/raw/query.parquet", columns=["id", "query"]).to_pydict()
qtext = dict(zip(qt["id"], qt["query"]))
jobs = [(q, i) for q, v in pool.items() for cs in v.values() for i in cs if i in c["text"]]
print(f"{len(jobs)} pairs", flush=True)
model = CrossEncoder(512, repo=repo)
t0 = time.time()
sc = model.score([(qtext[q], chunk_text(c, i)) for q, i in jobs], batch_size=128)
print(f"scored in {time.time() - t0:.0f}s", flush=True)
out = {}
for (q, i), s in zip(jobs, sc):
    out.setdefault(q, {}).setdefault("vi", {})[i] = float(s)
pickle.dump(out, open(outp, "wb"))
print("VNRR_DONE", flush=True)
