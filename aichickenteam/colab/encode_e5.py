"""Colab T4: mã hoá ứng viên (cand_docs.parquet: title + 1500 ký tự) + 1200 câu hỏi bằng multilingual-e5-large-instruct.
Ra /content/e5_docs.npy (fp16, chuẩn hoá), e5_ids.json, e5_queries.npy. (offline: GN .786 / e5 .780 / GN+e5 .799 AUC)"""
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

m = SentenceTransformer("intfloat/multilingual-e5-large-instruct", model_kwargs={"torch_dtype": torch.float16}, device="cuda")
m.max_seq_length = 512
q = pd.read_parquet("/content/query.parquet")
P = "Instruct: Given a Vietnamese medical question, retrieve relevant passages\nQuery: "
np.save("/content/e5_queries.npy", m.encode(q["query"].tolist(), prompt=P, normalize_embeddings=True, batch_size=32).astype(np.float16))
json.dump([int(x) for x in q.id], open("/content/e5_qids.json", "w"))
d = pd.read_parquet("/content/cand_docs.parquet")
E = m.encode(d.text.tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype(np.float16)
np.save("/content/e5_docs.npy", E); json.dump(d.doc_id.tolist(), open("/content/e5_ids.json", "w"))
print("DONE", E.shape)
