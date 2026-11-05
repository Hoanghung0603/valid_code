"""Kaggle GPU: fine-tune GreenNode-Embedding trên nhãn thật (chunk trúng/trượt từ các lần chấm), đánh giá trên 100 câu giữ riêng.
Output: /kaggle/working/gn-ft (model) + ft_eval.json (AUC/P@10 base vs ft trên tập eval)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 1 GPU: tránh lỗi NCCL của DataParallel (lần 1 chết ở bước 146/147)
import glob
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

BASE = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
df = pd.read_parquet(glob.glob("/kaggle/input/**/ft_pairs.parquet", recursive=True)[0])
tr, ev = df[df.split == "train"], df[df.split == "eval"]
FT_ALL = True   # lần 1 (500 train/100 eval): AUC .693->.731, P@10 .203->.221 -> giờ train trên tất cả câu có nhãn
if FT_ALL:
    tr = df


def metrics(model, d):
    qs = d["query"].drop_duplicates().tolist()
    qe = dict(zip(qs, model.encode(qs, normalize_embeddings=True, batch_size=32)))
    te = model.encode(d["text"].tolist(), normalize_embeddings=True, batch_size=32)
    s = (te * np.stack([qe[x] for x in d["query"]])).sum(1)
    d = d.assign(s=s); aucs, p10 = [], []
    for _, g in d.groupby("qid"):
        pos, neg = g[g.label == 1].s.values, g[g.label == 0].s.values
        if len(pos) and len(neg):
            aucs.append((pos[:, None] > neg[None, :]).mean())
        p10.append(g.sort_values("s", ascending=False).label.values[:10].mean())
    return dict(auc=float(np.mean(aucs)), p10=float(np.mean(p10)))


model = SentenceTransformer(BASE, model_kwargs={"torch_dtype": torch.float32})
model.max_seq_length = 256
res = {"base": metrics(model, ev)}
print("base", res["base"], flush=True)

# triplets (query, positive, hard negative cùng câu)
rng = random.Random(0); A, P, N = [], [], []
for qid, g in tr.groupby("qid"):
    pos, neg = g[g.label == 1].text.tolist(), g[g.label == 0].text.tolist()
    if not pos or not neg:
        continue
    qtext = g["query"].iloc[0]
    for p in pos:
        for n in rng.sample(neg, min(2, len(neg))):
            A.append(qtext); P.append(p); N.append(n)
train = Dataset.from_dict({"anchor": A, "positive": P, "negative": N}).shuffle(seed=0)
print("triplets", len(train), flush=True)
args = SentenceTransformerTrainingArguments(
    output_dir="/kaggle/working/ckpt", num_train_epochs=1, per_device_train_batch_size=64, learning_rate=1e-5,
    warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=20, save_strategy="steps", save_steps=50, save_total_limit=1, report_to=[])
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss).train()
model.save("/kaggle/working/gn-ft")   # lưu ngay sau train, trước khi đánh giá
res["ft"] = metrics(model, ev)
print("ft", res["ft"], flush=True)
json.dump(res, open("/kaggle/working/ft_eval.json", "w"))
model.save("/kaggle/working/gn-ft")
