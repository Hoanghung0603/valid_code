"""Score each pooled (query, best chunk) pair with Qwen3-Reranker; zh/en chunks are also scored with the translated
query and keep the max.   python q8rr.py <pool.pkl> <out.pkl> <repo> <chunk dirs in run order...>
out.pkl: {qid: {"vi": {chunk: score}}} (vnrr.py layout)."""
import pickle, sys, time
from pathlib import Path
import pyarrow.parquet as pq
W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "code/retrieval"))
from pipeline import Qwen3Reranker, chunk_text  # noqa: E402
from translate_queries import key_of, load as load_translations  # noqa: E402
pool, outp, repo = pickle.loads(Path(sys.argv[1]).read_bytes()), Path(sys.argv[2]), sys.argv[3]
files = [f for d in sys.argv[4:] for f in sorted(Path(d).glob("*.parquet"))]
need = {i for v in pool.values() for c in v.values() for i in c}
c = {"text": {}, "heading_path": {}, "lang": {}}
base = 0
for f in files:
    for b in pq.ParquetFile(f).iter_batches(batch_size=50000, columns=["text", "heading_path", "lang"]):
        tx, hd, lg = b.column("text").to_pylist(), b.column("heading_path").to_pylist(), b.column("lang").to_pylist()
        for j in range(len(tx)):
            if base + j in need:
                c["text"][base + j], c["heading_path"][base + j], c["lang"][base + j] = tx[j], hd[j], lg[j]
        base += len(tx)
qt = pq.read_table(W / "data/raw/query.parquet", columns=["id", "query"]).to_pydict()
qtext = dict(zip(qt["id"], qt["query"]))
trans = load_translations(W / "data/qtrans/v1/translations.jsonl")
jobs, pairs = [], []
for q, v in pool.items():
    tr = trans[key_of(qtext[q])]
    for cs in v.values():
        for i in cs:
            if i not in c["text"]:
                continue
            jobs.append((q, i)); pairs.append((qtext[q], chunk_text(c, i)))
            if c["lang"][i] in ("en", "zh"):
                jobs.append((q, i)); pairs.append((tr[c["lang"][i]], chunk_text(c, i)))
print(f"{len(pairs)} pairs", flush=True)
model = Qwen3Reranker(512, repo=repo)
t0 = time.time()
sc = model.score(pairs, batch_size=32)
print(f"scored in {time.time() - t0:.0f}s", flush=True)
out = {}
for (q, i), s in zip(jobs, sc):
    d = out.setdefault(q, {}).setdefault("vi", {})
    d[i] = max(d.get(i, float(s)), float(s))
pickle.dump(out, open(outp, "wb"))
print("Q8RR_DONE", flush=True)
