"""Score pooled (query, doc) pairs with a Qwen3-Embedding model: cosine of the instructed query and the doc (title +
start of text, 256 tokens), only for the docs in each query's pool.
    python qemb_pool.py <model> <pool.pkl> <top n per query> <out.pkl> <docs dirs...>
out.pkl: {0: {qid: {doc: cosine}}} (prf.py layout, for make_sub3 --prf ... --prf-k 0)."""
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
W = Path(__file__).resolve().parent
model_name, pool_p, n, outp, srcs = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], [Path(x) for x in sys.argv[5:]]
MAXLEN, BATCH = 256, int(os.environ.get("BATCH", 32))
TASK = "Given a Vietnamese medical question, retrieve documents (Vietnamese, English or Chinese) that answer it"
pool = {q: list(v)[:n] for q, v in pickle.load(open(pool_p, "rb")).items()}
need = pa.array(sorted({d for v in pool.values() for d in v}), pa.int64())
ids, texts = [], []
for f in [f for d in srcs for f in sorted(d.glob("*.parquet"))]:
    t = pq.read_table(f, columns=["id", "title", "text"])
    t = t.filter(pc.is_in(t.column("id").cast(pa.int64()), value_set=need)).to_pydict()
    ids += [int(i) for i in t["id"]]
    texts += [f"{ti or ''}\n{(x or '')[:1200]}" for ti, x in zip(t["title"], t["text"])]
print(f"{len(ids)} docs", flush=True)
tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if os.environ.get("QUANT") == "4":
    from transformers import BitsAndBytesConfig
    model = AutoModel.from_pretrained(model_name, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
else:
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def embed(xs):
    b = tok(xs, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
    return F.normalize(model(**b).last_hidden_state[:, -1].float(), dim=-1)


qt = pq.read_table(W / "data/raw/query.parquet", columns=["id", "query"]).to_pydict()
qv = torch.cat([embed([f"Instruct: {TASK}\nQuery:{x}" for x in qt["query"][a:a + BATCH]]) for a in range(0, len(qt["id"]), BATCH)])
qrow = {q: j for j, q in enumerate(qt["id"])}
order = np.argsort([len(x) for x in texts])
dv = torch.zeros((len(ids), qv.shape[1]), dtype=torch.float16)
t0 = time.time()
for a in range(0, len(order), BATCH):
    sel = order[a:a + BATCH]
    dv[torch.as_tensor(sel)] = embed([texts[i] for i in sel]).half().cpu()
    if (a // BATCH) % 500 == 0:
        print(f"{a + len(sel)}/{len(order)} docs, {(a + len(sel)) / (time.time() - t0):.0f} docs/s", flush=True)
row = {d: i for i, d in enumerate(ids)}
qc = qv.cpu()
out = {0: {}}
for q, docs in pool.items():
    ds = [d for d in docs if d in row]
    s = (dv[[row[d] for d in ds]].float() @ qc[qrow[q]]).tolist()
    out[0][int(q)] = dict(zip(ds, s))
pickle.dump(out, open(outp, "wb"))
print("QPOOL_DONE", flush=True)
