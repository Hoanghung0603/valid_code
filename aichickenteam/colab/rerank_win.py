"""Colab T4: bge-reranker-v2-m3 chấm (câu hỏi, cửa sổ dense tốt nhất của doc) cho top-120/câu → /content/rr_win.parquet (qid, doc_id, s)."""
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

p = pd.read_parquet("/content/rr_win_pairs.parquet")
q = pd.read_parquet("/content/query.parquet").set_index("id")["query"]
m = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
p["s"] = m.predict(list(zip(q.loc[p.qid].tolist(), p.text.str[:1600].tolist())), batch_size=64, show_progress_bar=True)
p[["qid", "doc_id", "s"]].to_parquet("/content/rr_win.parquet", index=False)
print("DONE", len(p))
