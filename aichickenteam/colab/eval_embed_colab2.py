"""Colab T4 (8B không vừa 16GB fp16 nên bỏ): so các model embedding mở trên 12k cặp có nhãn thật (eval_pairs) – AUC / P@10 theo câu."""
import gc
import glob
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("/content/eval_pairs.parquet")
MODELS = [("AITeamVN/Vietnamese_Embedding", {}, {}),
          ("jinaai/jina-embeddings-v3", {"task": "retrieval.query", "prompt_name": "retrieval.query"}, {"trust_remote_code": True})]


def metrics(d):
    aucs, p10 = [], []
    for _, g in d.groupby("qid"):
        pos, neg = g[g.label == 1].s.values, g[g.label == 0].s.values
        if len(pos) and len(neg):
            aucs.append((pos[:, None] > neg[None, :]).mean())
        p10.append(g.sort_values("s", ascending=False).label.values[:10].mean())
    return dict(auc=round(float(np.mean(aucs)), 4), p10=round(float(np.mean(p10)), 4))


res = {}
qs = df["query"].drop_duplicates().tolist()
for name, qkw, mkw in MODELS:
    try:
        m = SentenceTransformer(name, model_kwargs={"torch_dtype": torch.float16}, device="cuda", **mkw)
        m.max_seq_length = 512
        qe = dict(zip(qs, m.encode(qs, normalize_embeddings=True, batch_size=16, **qkw)))
        te = m.encode(df["text"].str[:2000].tolist(), normalize_embeddings=True, batch_size=8, **({"task": "retrieval.passage", "prompt_name": "retrieval.passage"} if "jina" in name else {}))
        s = (te * np.stack([qe[x] for x in df["query"]])).sum(1)
        res[name] = metrics(df.assign(s=s)); np.save(f"/content/s_{name.split('/')[1]}.npy", s)
    except Exception as e:
        res[name] = f"error {e}"[:300]
    print(name, res[name], flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open("/content/eval_embed.json", "w"))
