"""bge-m3 (GPU local) cho 240k doc PubMed (data/docs_en/pubmed.parquet) → top-K PubMed mỗi câu hỏi.

Ghi data/cache/pubmed_bgem3/{docs.npy, ids.json, q.npy} và data/cache/dense_pubmed_bgem3.json ({qid: [[s, doc_id], ...]}).
    python scripts/pubmed_embed.py [--k 300] [--bs 64]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

O = "data/cache/pubmed_bgem3"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=300); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args(); os.makedirs(O, exist_ok=True)
    d = pd.read_parquet("data/docs_en/pubmed.parquet"); d = d[d.text.str.len() > 0]
    q = pd.read_parquet("data/ViBioMIR/query.parquet")
    m = SentenceTransformer(os.environ.get("R2AI_BGEM3", "BAAI/bge-m3"), model_kwargs={"torch_dtype": torch.float16}, device="cuda"); m.max_seq_length = 512
    if not os.path.exists(f"{O}/docs.npy"):
        order = d.text.str.len().argsort().values   # gom độ dài gần nhau → ít padding
        e = m.encode(d.text.values[order].tolist(), batch_size=a.bs, normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True).astype(np.float16)
        E = np.empty_like(e); E[order] = e
        np.save(f"{O}/docs.npy", E); json.dump(d.doc_id.astype(int).tolist(), open(f"{O}/ids.json", "w"))
    E = torch.tensor(np.load(f"{O}/docs.npy"), device="cuda"); ids = np.array(json.load(open(f"{O}/ids.json")))
    m.max_seq_length = 256
    Q = m.encode(q["query"].tolist(), batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    np.save(f"{O}/q.npy", Q)
    S = torch.tensor(Q, device="cuda") @ E.T
    v, i = S.float().topk(a.k, dim=1); v, i = v.cpu().numpy(), i.cpu().numpy()
    out = {str(qid): [[round(float(s), 4), int(ids[j])] for s, j in zip(v[r], i[r])] for r, qid in enumerate(q.id)}
    json.dump(out, open("data/cache/dense_pubmed_bgem3.json", "w"))
    print("wrote dense_pubmed_bgem3.json", len(out), "docs", len(ids), "top1 mean", round(float(v[:, 0].mean()), 3))


if __name__ == "__main__":
    main()
