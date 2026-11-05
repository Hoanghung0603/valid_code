"""Kaggle GPU: fine-tune GreenNode (bi-encoder) theo fold câu hỏi trên nhãn thật, rồi chấm cos(query, doc) cho top-300 ứng viên/câu
(câu public trong fold f ← model không thấy fold f; câu private (-1) ← model train trên tất cả). Out: /kaggle/working/gnft_<folds>.parquet"""
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

FOLDS = [int(x) for x in os.environ.get("FOLDS", "0 1 2").split()]
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr = pd.read_parquet(G("ce_train.parquet")); sc = pd.read_parquet(G("gn_score_pairs.parquet"))
txt = pd.read_parquet(G("cand_docs.parquet")).set_index("doc_id").text
Q = pd.read_parquet(G("query.parquet")).set_index("id")["query"]
_c = [p for p in glob.glob("/kaggle/input/**/modules.json", recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), "model.safetensors"))]   # GreenNode bị gated trên HF -> bản copy riêng tư
BASE = os.path.dirname(_c[0]) if _c else "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
OUT = "/kaggle/working/gnft_" + "_".join(map(str, FOLDS)) + ".parquet"
res = []
for f in FOLDS:
    trf = tr[tr.fold != f] if f >= 0 else tr
    rng = random.Random(f + 10); A, P, N = [], [], []
    for qid, g in trf.groupby("qid"):
        pos, neg = g[g.label == 1].text.tolist(), g[g.label == 0].text.tolist()
        if not pos or not neg:
            continue
        for p in pos:
            for n in rng.sample(neg, min(2, len(neg))):
                A.append(Q[qid]); P.append(p); N.append(n)
    ds = Dataset.from_dict({"anchor": A, "positive": P, "negative": N}).shuffle(seed=f + 1)
    m = SentenceTransformer(BASE, model_kwargs={"torch_dtype": torch.float32}); m.max_seq_length = 256
    args = SentenceTransformerTrainingArguments(output_dir=f"/kaggle/working/ck{f}", num_train_epochs=1, per_device_train_batch_size=64,
                                                learning_rate=1e-5, warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
                                                logging_steps=50, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=m, args=args, train_dataset=ds, loss=CachedMultipleNegativesRankingLoss(m, mini_batch_size=16)).train()
    m.half(); m.max_seq_length = 512
    part = sc[sc.fold == f]; docs = part.doc_id.unique().tolist(); qs = part.qid.unique().tolist()
    De = dict(zip(docs, m.encode(txt.loc[docs].str[:1700].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)))
    Qe = dict(zip(qs, m.encode(Q.loc[qs].tolist(), normalize_embeddings=True, batch_size=64)))
    s = [float(De[d] @ Qe[q]) for q, d in zip(part.qid, part.doc_id)]
    res.append(part[["qid", "doc_id"]].assign(s=s, fold=f)); pd.concat(res).to_parquet(OUT, index=False)
    print(f"FOLD {f} done: triplets {len(ds)} scored {len(part)}", flush=True)
    del m; torch.cuda.empty_cache()
print("ALLDONE", flush=True)
