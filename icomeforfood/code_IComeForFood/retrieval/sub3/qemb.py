"""Doc-level dense retrieval with Qwen3-Embedding: each doc is title + the start of its text (512 tokens), the 1,200
queries carry the retrieval instruction; keeps the top TOP docs per query (no doc vectors are stored).

    python qemb.py <model> <out.pkl> <docs dirs...>
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

W = Path(__file__).resolve().parent
model_name, outp, srcs = sys.argv[1], Path(sys.argv[2]), [Path(x) for x in sys.argv[3:]]
TOP, MAXLEN, BATCH = 500, 256, int(os.environ.get("BATCH", 64))
TASK = "Given a Vietnamese medical question, retrieve documents (Vietnamese, English or Chinese) that answer it"

tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if os.environ.get("QUANT") == "4":  # 4-bit NF4 when the shared GPU has no room for fp16 weights
    from transformers import BitsAndBytesConfig
    model = AutoModel.from_pretrained(model_name, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
else:
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def embed(texts: list[str]) -> torch.Tensor:
    b = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to("cuda")
    h = model(**b).last_hidden_state[:, -1]  # left padding: the last position is each text's last token
    return F.normalize(h.float(), dim=-1)


q = pq.read_table(W / "data/raw/query.parquet", columns=["id", "query"]).to_pydict()
qv = torch.cat([embed([f"Instruct: {TASK}\nQuery:{x}" for x in q["query"][a:a + BATCH]])
                for a in range(0, len(q["id"]), BATCH)])
part = outp.parent / (outp.stem + "_parts")  # top TOP per query for each docs file, so a restart resumes
part.mkdir(exist_ok=True)
t0, n_done = time.time(), 0
for f in [f for d in srcs for f in sorted(d.glob("*.parquet"))]:
    dst = part / f"{f.parent.name}_{f.stem}.pt"
    if dst.exists():
        continue
    best_s = torch.full((len(q["id"]), 0), -1.0, device="cuda")
    best_i = torch.zeros((len(q["id"]), 0), dtype=torch.long, device="cuda")
    for b in pq.ParquetFile(f).iter_batches(batch_size=8192, columns=["id", "title", "text"]):
        d = b.to_pydict()
        texts = [f"{ti or ''}\n{(x or '')[:1200]}" for ti, x in zip(d["title"], d["text"])]
        order = np.argsort([len(x) for x in texts])
        dv = torch.cat([embed([texts[i] for i in order[a:a + BATCH]]) for a in range(0, len(order), BATCH)])
        s = qv @ dv.T
        idx = torch.as_tensor(np.array(d["id"], np.int64)[order], device="cuda").expand(len(q["id"]), -1)
        best_s, k = torch.cat([best_s, s], 1).topk(min(TOP, best_s.shape[1] + s.shape[1]), dim=1)
        best_i = torch.cat([best_i, idx], 1).gather(1, k)
        n_done += len(texts)
    torch.save({"s": best_s.cpu(), "i": best_i.cpu()}, dst)
    print(f"{f.name}: {n_done} docs this run, {n_done / (time.time() - t0):.0f} docs/s", flush=True)
S = torch.cat([torch.load(p)["s"] for p in sorted(part.glob("*.pt"))], 1)
I = torch.cat([torch.load(p)["i"] for p in sorted(part.glob("*.pt"))], 1)
S, k = S.topk(TOP, dim=1)
pickle.dump({"qids": q["id"], "doc": I.gather(1, k).numpy(), "score": S.numpy()}, open(outp, "wb"))
print("QEMB_DONE", flush=True)
