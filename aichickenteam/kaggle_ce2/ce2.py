"""Kaggle T4×2: fine-tune lại bge-reranker-v2-m3 trên TOÀN BỘ nhãn (scripts/ce2_prep.py), 5 fold câu (OOF public) + 1 model tất cả (private).
Mỗi GPU một tiến trình (fork): GPU0 = fold 0,1,2; GPU1 = fold 3,4,-1. Mỗi fold xong ghi /kaggle/working/ce2_g{gpu}.parquet (qid, doc_id, s, fold).
Model fold -1 (fp16) lưu /kaggle/working/ce2_all để chấm doc mới về sau."""
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME, ML, BS, LR = "BAAI/bge-reranker-v2-m3", 384, 16, 1e-5
W_ = "/kaggle/working"
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
D = pd.read_parquet(G("docs.parquet")); TX = dict(zip(D.doc_id.astype(str), D.text)); del D
tr = pd.read_parquet(G("train.parquet")); sc = pd.read_parquet(G("score.parquet"))
tr["doc_id"] = tr.doc_id.astype(str); sc["doc_id"] = sc.doc_id.astype(str)
Q = pd.read_parquet(G("query.parquet")).set_index("id")["query"]
tok = AutoTokenizer.from_pretrained(NAME)
AutoModelForSequenceClassification.from_pretrained(NAME)   # tải về cache trước khi fork
t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:.0f}s]", *x, flush=True)


def batches(df, bs, dev, shuffle):
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), bs):
        b = df.iloc[idx[i:i + bs]]
        enc = tok(Q.loc[b.qid].tolist(), [TX[d] for d in b.doc_id], truncation="only_second", max_length=ML, padding=True, return_tensors="pt")
        yield b, {k: v.to(dev) for k, v in enc.items()}


def worker(g, folds):
    dev = torch.device(f"cuda:{g}"); torch.cuda.set_device(dev); np.random.seed(g)
    OUT = f"{W_}/ce2_g{g}.parquet"
    for f in folds:
        t1 = time.time()
        m = AutoModelForSequenceClassification.from_pretrained(NAME).to(dev); m.gradient_checkpointing_enable()
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01); scaler = torch.cuda.amp.GradScaler()
        trf = tr[tr.fold != f] if f >= 0 else tr
        steps = (len(trf) + BS - 1) // BS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
        m.train()
        for i, (b, enc) in enumerate(batches(trf, BS, dev, True)):
            with torch.autocast("cuda", dtype=torch.float16):
                logit = m(**enc).logits.view(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(b.label.values, device=dev, dtype=torch.float))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if i % 500 == 0:
                log(f"gpu{g} fold {f} step {i}/{steps} loss {loss.item():.3f}")
        m.eval(); m.half()
        part = sc[sc.fold == f]
        part = part.iloc[np.argsort([len(TX[d]) for d in part.doc_id])]   # gom độ dài → ít padding
        s = []
        with torch.no_grad():
            for b, enc in batches(part, 128, dev, False):
                s += m(**enc).logits.view(-1).float().cpu().tolist()
        res = part[["qid", "doc_id"]].assign(s=s, fold=f)
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        pd.concat([prev, res]).to_parquet(OUT, index=False)
        if f == -1:
            m.save_pretrained(f"{W_}/ce2_all"); tok.save_pretrained(f"{W_}/ce2_all")
        log(f"FOLD {f} done (gpu{g}): train {len(trf)} score {len(part)} in {time.time() - t1:.0f}s")
        del m, opt; torch.cuda.empty_cache()


NG = len(os.popen("nvidia-smi -L").read().strip().splitlines())   # không gọi torch.cuda trước khi fork

if __name__ == "__main__":
    log("train", len(tr), "pos", int(tr.label.sum()), "score", len(sc), "gpus", NG)
    plan = [[0, 1, 2], [3, 4, -1]] if NG > 1 else [[0, 1, 2, 3, 4, -1]]
    ctx = mp.get_context("fork"); ps = [ctx.Process(target=worker, args=(g, fs)) for g, fs in enumerate(plan)]
    [p.start() for p in ps]; [p.join() for p in ps]
    log("ALLDONE", [p.exitcode for p in ps])
