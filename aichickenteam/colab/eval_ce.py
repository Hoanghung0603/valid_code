"""Colab T4: đánh giá cross-encoder mở trên 12k cặp có nhãn (eval_pairs) – so với bge-reranker-v2-m3 (.765 AUC / .218 P@10)."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "FlagEmbedding"], check=False)
import numpy as np
import pandas as pd
import torch

df = pd.read_parquet("/content/eval_pairs.parquet")
pairs = [[q, t[:1500]] for q, t in zip(df["query"], df["text"])]


def metrics(s):
    d = df.assign(s=s); a, p = [], []
    for _, g in d.groupby("qid"):
        po, ne = g[g.label == 1].s.values, g[g.label == 0].s.values
        if len(po) and len(ne): a.append((po[:, None] > ne[None, :]).mean())
        p.append(g.sort_values("s", ascending=False).label.values[:10].mean())
    return round(float(np.mean(a)), 4), round(float(np.mean(p)), 4)


for name in ["jinaai/jina-reranker-v2-base-multilingual", "BAAI/bge-reranker-v2-gemma"]:
    m = None
    try:
        if "gemma" in name:
            from FlagEmbedding import FlagLLMReranker
            m = FlagLLMReranker(name, use_fp16=True)
            s = np.array(m.compute_score(pairs, batch_size=16, max_length=512))
        else:
            from sentence_transformers import CrossEncoder
            m = CrossEncoder(name, trust_remote_code=True, max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
            s = m.predict(pairs, batch_size=32)
        np.save(f"/content/ce_{name.split('/')[1]}.npy", s)
        print("RESULT", name, metrics(s), flush=True)
    except Exception as e:
        print("RESULT", name, "error", str(e)[:200], flush=True)
    del m; torch.cuda.empty_cache()
