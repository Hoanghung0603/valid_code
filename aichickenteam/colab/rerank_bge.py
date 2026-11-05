"""Colab T4: bge-reranker-v2-m3 (cross-encoder) chấm top-120 ứng viên/câu theo LTR9 → /content/rr_bge.parquet (qid, doc_id, s)."""
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

p = pd.read_parquet("/content/rr_pairs.parquet")
d = pd.read_parquet("/content/cand_docs.parquet").set_index("doc_id").text
q = pd.read_parquet("/content/query.parquet").set_index("id")["query"]
m = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
p["s"] = m.predict(list(zip(q.loc[p.qid].tolist(), d.loc[p.doc_id].str[:1500].tolist())), batch_size=64, show_progress_bar=True)
p.to_parquet("/content/rr_bge.parquet", index=False)
print("DONE", len(p))
