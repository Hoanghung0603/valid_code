"""HyDE: mã hoá đoạn văn giả (colab/hyde_gen.py) bằng GreenNode, tìm trên index doc-level data/cache/dense_gn (650k doc).
Ra 2 nguồn: hyde (chỉ đoạn giả) và mix (chuẩn hoá(q + h)) – định dạng {qid: [[score, doc_id], ...]} như dense_gn_top300.
    python scripts/hyde_search.py data/cache/hyde.jsonl"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

hy = {int(json.loads(l)["id"]): json.loads(l)["text"] for l in open(sys.argv[1])}
q = pd.read_parquet("data/ViBioMIR/query.parquet")
m = SentenceTransformer("GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1", model_kwargs={"torch_dtype": torch.float16}, device="cuda")
m.max_seq_length = 512
Q = m.encode(q["query"].tolist(), normalize_embeddings=True, batch_size=32)
H = m.encode([hy.get(int(i), t) for i, t in zip(q.id, q["query"])], normalize_embeddings=True, batch_size=16)
M = Q + H; M /= np.linalg.norm(M, axis=1, keepdims=True)
K = 300
res = {"hyde": [None, None], "mix": [None, None]}
V = {"hyde": torch.tensor(H).float(), "mix": torch.tensor(M).float()}
ids_all = []
for f in sorted(glob.glob("data/cache/dense_gn/part*.npy")):
    E = torch.tensor(np.load(f)).float(); base = len(ids_all); ids_all += json.load(open(f.replace(".npy", ".ids.json")))
    for k, v in V.items():
        s, i = (v @ E.T).topk(K, dim=1); i = i + base
        if res[k][0] is not None:
            s = torch.cat([res[k][0], s], 1); i = torch.cat([res[k][1], i], 1); o = s.topk(K, dim=1).indices
            s, i = s.gather(1, o), i.gather(1, o)
        res[k] = [s, i]
for k, (s, i) in res.items():
    out = {str(int(qid)): [[float(s[r, j]), ids_all[int(i[r, j])]] for j in range(K)] for r, qid in enumerate(q.id)}
    json.dump(out, open(f"data/cache/dense_{k}_top300.json", "w")); print("wrote", k)
