"""Colab: jina-reranker-v2-base-multilingual cần transformers cũ (code remote dùng create_position_ids_from_input_ids đã bị xoá ở bản mới).
Chạy: pip install --target /content/oldtf "transformers==4.46.3" "tokenizers<0.21" einops ; PYTHONPATH=/content/oldtf python eval_jina.py"""
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification

print("transformers", transformers.__version__, flush=True)
df = pd.read_parquet("/content/eval_pairs.parquet")
pairs = [[q, t[:1500]] for q, t in zip(df["query"], df["text"])]
m = AutoModelForSequenceClassification.from_pretrained("jinaai/jina-reranker-v2-base-multilingual", torch_dtype=torch.float16, trust_remote_code=True).cuda().eval()
s = np.array(m.compute_score(pairs, max_length=1024, batch_size=32))
np.save("/content/ce_jina.npy", s)
d = df.assign(s=s); a, p = [], []
for _, g in d.groupby("qid"):
    po, ne = g[g.label == 1].s.values, g[g.label == 0].s.values
    if len(po) and len(ne): a.append((po[:, None] > ne[None, :]).mean())
    p.append(g.sort_values("s", ascending=False).label.values[:10].mean())
print("RESULT jina", round(float(np.mean(a)), 4), round(float(np.mean(p)), 4), flush=True)
