"""Kaggle T4×2: fine-tune lại GreenNode bi-encoder trên TOÀN BỘ nhãn (dữ liệu của kaggle_ce2: scripts/ce2_prep.py), fold như LTR.
Như kaggle_gnft_a nhưng nhãn mới (~5×). Mỗi GPU một tiến trình con (CUDA_VISIBLE_DEVICES): GPU0 = fold 0,1,2; GPU1 = fold 3,4,-1.
Out: /kaggle/working/gnft2_g{gpu}.parquet (qid, doc_id, s, fold) – ghi sau mỗi fold; chạy lại sẽ bỏ qua fold đã có."""
import os
import subprocess
import sys

WORKER = r'''
import glob, os, random, sys, time
import numpy as np, pandas as pd, torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
g = int(sys.argv[1]); FOLDS = [int(x) for x in sys.argv[2].split(",")]
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
D = pd.read_parquet(G("docs.parquet")); TX = dict(zip(D.doc_id.astype(str), D.text)); del D
tr = pd.read_parquet(G("train.parquet")); sc = pd.read_parquet(G("score.parquet"))
tr["doc_id"] = tr.doc_id.astype(str); sc["doc_id"] = sc.doc_id.astype(str)
Q = pd.read_parquet(G("query.parquet")).set_index("id")["query"]
_c = [p for p in glob.glob("/kaggle/input/**/modules.json", recursive=True) if os.path.exists(os.path.join(os.path.dirname(p), "model.safetensors"))]
BASE = os.path.dirname(_c[0])
OUT = f"/kaggle/working/gnft2_g{g}.parquet"
done = set(pd.read_parquet(OUT).fold.unique()) if os.path.exists(OUT) else set()
t0 = time.time()
for f in FOLDS:
    if f in done:
        continue
    trf = tr[tr.fold != f] if f >= 0 else tr
    rng = random.Random(f + 10); A, P, N = [], [], []
    for qid, gq in trf.groupby("qid"):
        pos, neg = gq[gq.label == 1].doc_id.tolist(), gq[gq.label == 0].doc_id.tolist()
        if not pos or not neg:
            continue
        for p in pos:
            for n in rng.sample(neg, min(2, len(neg))):
                A.append(Q[qid]); P.append(TX[p][:1700]); N.append(TX[n][:1700])
    ds = Dataset.from_dict({"anchor": A, "positive": P, "negative": N}).shuffle(seed=f + 1)
    m = SentenceTransformer(BASE, model_kwargs={"torch_dtype": torch.float32}); m.max_seq_length = 256
    args = SentenceTransformerTrainingArguments(output_dir=f"/tmp/ck{f}", num_train_epochs=1, per_device_train_batch_size=64,
                                                learning_rate=1e-5, warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
                                                logging_steps=100, save_strategy="no", report_to=[])
    SentenceTransformerTrainer(model=m, args=args, train_dataset=ds, loss=CachedMultipleNegativesRankingLoss(m, mini_batch_size=16)).train()
    m.half(); m.max_seq_length = 512
    part = sc[sc.fold == f]; docs = part.doc_id.unique().tolist(); qs = part.qid.unique().tolist()
    docs.sort(key=lambda d: len(TX[d]))
    De = dict(zip(docs, m.encode([TX[d][:1700] for d in docs], normalize_embeddings=True, batch_size=64, show_progress_bar=False)))
    Qe = dict(zip(qs, m.encode(Q.loc[qs].tolist(), normalize_embeddings=True, batch_size=64)))
    s = [float(De[d] @ Qe[q]) for q, d in zip(part.qid, part.doc_id)]
    prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
    pd.concat([prev, part[["qid", "doc_id"]].assign(s=s, fold=f)]).to_parquet(OUT, index=False)
    if f == -1:
        m.save(f"/kaggle/working/gnft2_all")
    print(f"[{time.time() - t0:.0f}s] FOLD {f} done (gpu{g}): triplets {len(ds)} scored {len(part)}", flush=True)
    del m; torch.cuda.empty_cache()
'''
open("/kaggle/working/worker.py", "w").write(WORKER)
NG = len(os.popen("nvidia-smi -L").read().strip().splitlines())
plan = ["0,1,2", "3,4,-1"] if NG > 1 else ["0,1,2,3,4,-1"]
ps = [subprocess.Popen([sys.executable, "/kaggle/working/worker.py", str(g), fs], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
      for g, fs in enumerate(plan)]
print("ALLDONE", [p.wait() for p in ps], flush=True)
os.remove("/kaggle/working/worker.py")
