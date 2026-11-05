"""So embedding trên 12k cặp có nhãn (eval_pairs) – chạy local GPU nhỏ (model ≤ ~600M).
    python scripts/eval_emb_local.py GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1 google/embeddinggemma-300m"""
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import os
df = pd.read_parquet("colab/eval_pairs.parquet" if os.path.exists("colab/eval_pairs.parquet") else "/content/eval_pairs.parquet")
qs = df["query"].drop_duplicates().tolist()


def metrics(s):
    d = df.assign(s=s); a, p = [], []
    for _, g in d.groupby("qid"):
        po, ne = g[g.label == 1].s.values, g[g.label == 0].s.values
        if len(po) and len(ne): a.append((po[:, None] > ne[None, :]).mean())
        p.append(g.sort_values("s", ascending=False).label.values[:10].mean())
    return round(float(np.mean(a)), 4), round(float(np.mean(p)), 4)


S = {}
for name in sys.argv[1:]:
    m = SentenceTransformer(name, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16 if "gemma" in name else torch.float16})
    m.max_seq_length = 512
    texts = df["text"].str[:2000].tolist()
    if "embeddinggemma" in name:   # EmbeddingGemma: prompt SearchQuery / Document; bf16 (fp16 tràn số – theo README)
        qe = m.encode(qs, prompt_name="SearchQuery", normalize_embeddings=True, batch_size=16)
        te = m.encode(texts, prompt_name="Document", normalize_embeddings=True, batch_size=8)
    else:
        qe = m.encode(qs, normalize_embeddings=True, batch_size=16); te = m.encode(texts, normalize_embeddings=True, batch_size=8)
    qd = dict(zip(qs, qe)); s = (te * np.stack([qd[x] for x in df["query"]])).sum(1).astype(np.float32)
    S[name] = s; print("RESULT", name, "AUC/P@10", metrics(s), flush=True)
    np.save(("/content/" if os.path.exists("/content") else "data/cache/") + "s_" + name.split("/")[1] + ".npy", s)
    del m; torch.cuda.empty_cache()
if len(S) == 2:
    z = lambda x: (x - x.mean()) / x.std(); a, b = S.values()
    print("RESULT ensemble", metrics(z(a) + z(b)))
