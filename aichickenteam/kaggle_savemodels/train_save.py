"""Kaggle GPU: train lại 3 model fine-tune trên TẤT CẢ nhãn (fold -1) và LƯU lại (fp16) để chấm cặp (câu, doc MỚI crawl) sau này
mà không phải train lại: CE đầu bài (như colab/ft_ce.py), CE cửa sổ dense (colab/ft_ce_win.py), GreenNode bi-encoder (kaggle_gnft_a).
Out: /kaggle/working/models/{ce_doc,ce_win,gn}."""
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
import time

import numpy as np
import pandas as pd
import torch

G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
Q = pd.read_parquet(G("query.parquet")).set_index("id")["query"]
OUTD = "/kaggle/working/models"


def train_ce(train_file, name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    NAME, ML, BS, LR = "BAAI/bge-reranker-v2-m3", 384, 8, 1e-5
    tr = pd.read_parquet(G(train_file)); tok = AutoTokenizer.from_pretrained(NAME)
    t0 = time.time(); np.random.seed(0); torch.manual_seed(0)
    m = AutoModelForSequenceClassification.from_pretrained(NAME).cuda(); m.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01); scaler = torch.cuda.amp.GradScaler()
    steps = (len(tr) + BS - 1) // BS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    m.train(); idx = np.random.permutation(len(tr))
    for i in range(0, len(tr), BS):
        b = tr.iloc[idx[i:i + BS]]
        enc = {k: v.cuda() for k, v in tok(Q.loc[b.qid].tolist(), b.text.tolist(), truncation="only_second", max_length=ML, padding=True,
                                           return_tensors="pt").items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logit = m(**enc).logits.view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(b.label.values, device="cuda", dtype=torch.float))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    m.half().save_pretrained(f"{OUTD}/{name}"); tok.save_pretrained(f"{OUTD}/{name}")
    print(f"{name} saved: train {len(tr)} in {time.time() - t0:.0f}s", flush=True)
    del m, opt; torch.cuda.empty_cache()


def train_gn():
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    tr = pd.read_parquet(G("ce_train.parquet")); f = -1
    _c = [p for p in glob.glob("/kaggle/input/**/modules.json", recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), "model.safetensors"))]
    rng = random.Random(f + 10); A, P, N = [], [], []
    for qid, g in tr.groupby("qid"):
        pos, neg = g[g.label == 1].text.tolist(), g[g.label == 0].text.tolist()
        if not pos or not neg:
            continue
        for p in pos:
            for n in rng.sample(neg, min(2, len(neg))):
                A.append(Q[qid]); P.append(p); N.append(n)
    ds = Dataset.from_dict({"anchor": A, "positive": P, "negative": N}).shuffle(seed=f + 1)
    m = SentenceTransformer(os.path.dirname(_c[0]), model_kwargs={"torch_dtype": torch.float32}); m.max_seq_length = 256
    args = SentenceTransformerTrainingArguments(output_dir="/kaggle/working/ck", num_train_epochs=1, per_device_train_batch_size=64,
                                                learning_rate=1e-5, warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
                                                logging_steps=50, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=m, args=args, train_dataset=ds, loss=CachedMultipleNegativesRankingLoss(m, mini_batch_size=16)).train()
    m.half(); m.max_seq_length = 512; m.save(f"{OUTD}/gn")
    os.system("rm -rf /kaggle/working/ck")
    print(f"gn saved: triplets {len(ds)}", flush=True)
    del m; torch.cuda.empty_cache()


train_gn()
train_ce("ce_train.parquet", "ce_doc")
train_ce("ce_train_win.parquet", "ce_win")
print("ALLDONE", flush=True)
