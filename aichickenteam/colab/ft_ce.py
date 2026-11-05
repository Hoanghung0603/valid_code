"""Colab T4: fine-tune bge-reranker-v2-m3 trên nhãn thật theo 5 fold câu hỏi (OOF cho câu public) + 1 model trên tất cả (cho câu private).
In: /content/ce_train.parquet (qid, fold, doc_id, label, text), ce_score.parquet (qid, fold[-1=private], doc_id, text), query.parquet
Out: /content/ce_ft.parquet (qid, doc_id, s) – mỗi fold xong ghi nối tiếp (an toàn nếu phiên bị ngắt)."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME, ML, BS, LR = "BAAI/bge-reranker-v2-m3", 384, 8, 1e-5
tr = pd.read_parquet("/content/ce_train.parquet"); sc = pd.read_parquet("/content/ce_score.parquet")
Q = pd.read_parquet("/content/query.parquet").set_index("id")["query"]
tok = AutoTokenizer.from_pretrained(NAME)
folds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2, 3, 4, -1]
OUT = "/content/ce_ft_" + "_".join(map(str, folds)) + ".parquet"
done = set(pd.read_parquet(OUT).fold.unique()) if os.path.exists(OUT) else set()
folds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2, 3, 4, -1]


def batches(df, bs, shuffle):
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), bs):
        b = df.iloc[idx[i:i + bs]]
        enc = tok(Q.loc[b.qid].tolist(), b.text.tolist(), truncation="only_second", max_length=ML, padding=True, return_tensors="pt")
        yield b, {k: v.cuda() for k, v in enc.items()}


for f in folds:
    if f in done:
        continue
    t0 = time.time()
    m = AutoModelForSequenceClassification.from_pretrained(NAME).cuda(); m.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    trf = tr[tr.fold != f] if f >= 0 else tr
    steps = (len(trf) + BS - 1) // BS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    m.train()
    for b, enc in batches(trf, BS, True):
        with torch.autocast("cuda", dtype=torch.float16):
            logit = m(**enc).logits.view(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.tensor(b.label.values, device="cuda", dtype=torch.float))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    m.eval(); s = []
    part = sc[sc.fold == f]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b, enc in batches(part, 64, False):
            s += m(**enc).logits.view(-1).float().cpu().tolist()
    res = part[["qid", "doc_id"]].assign(s=s, fold=f)
    prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
    pd.concat([prev, res]).to_parquet(OUT, index=False)
    if os.environ.get("SAVE_MODEL"):   # lưu model (fp16) để chấm lại ứng viên mới mà không phải train lại
        m.half().save_pretrained(f"{os.environ['SAVE_MODEL']}/ce_fold{f}"); tok.save_pretrained(f"{os.environ['SAVE_MODEL']}/ce_fold{f}")
    print(f"FOLD {f} done: train {len(trf)} score {len(part)} in {time.time() - t0:.0f}s", flush=True)
    del m, opt; torch.cuda.empty_cache()
print("ALLDONE", flush=True)
