"""Colab T4: mã hoá ứng viên (cand_docs.parquet) + 1200 câu hỏi bằng model chọn qua argv: python encode_model.py <hf_model> <tag> [max_len]
Ra /content/<tag>_docs.npy (fp16), <tag>_ids.json, <tag>_queries.npy, <tag>_qids.json (cùng định dạng data/cache/e5 cho ltr.py)."""
import json
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

name, tag = sys.argv[1], sys.argv[2]
ml = int(sys.argv[3]) if len(sys.argv) > 3 else 256
kw = {"trust_remote_code": True} if "jina" in name else {}
m = SentenceTransformer(name, model_kwargs={"torch_dtype": torch.float16}, device="cuda", **kw)
m.max_seq_length = ml
qk = {"task": "retrieval.query", "prompt_name": "retrieval.query"} if "jina" in name else {}
dk = {"task": "retrieval.passage", "prompt_name": "retrieval.passage"} if "jina" in name else {}
q = pd.read_parquet("/content/query.parquet")
np.save(f"/content/{tag}_queries.npy", m.encode(q["query"].tolist(), normalize_embeddings=True, batch_size=32, **qk).astype(np.float16))
json.dump([int(x) for x in q.id], open(f"/content/{tag}_qids.json", "w"))
d = pd.read_parquet("/content/cand_docs.parquet")
E = m.encode(d.text.tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=True, **dk).astype(np.float16)
np.save(f"/content/{tag}_docs.npy", E); json.dump(d.doc_id.tolist(), open(f"/content/{tag}_ids.json", "w"))
print("DONE", E.shape)
