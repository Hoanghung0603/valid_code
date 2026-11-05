"""Kaggle GPU: đánh giá Qwen3-Reranker (4B) trên 12k cặp có nhãn thật (từ details v4) – so với GreenNode AUC .787 / P@10 .232."""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RR_MODEL", "Qwen/Qwen3-Reranker-4B")
df = pd.read_parquet(glob.glob("/kaggle/input/**/eval_pairs.parquet", recursive=True)[0])

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
          "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
inst = "Given a Vietnamese medical question, judge whether the Vietnamese document passage contains information relevant to answering it"


def fmt(q, d):
    return f"{prefix}<Instruct>: {inst}\n<Query>: {q}\n<Document>: {d[:2500]}{suffix}"


scores = []
B = 8
with torch.no_grad():
    for i in range(0, len(df), B):
        batch = [fmt(q, d) for q, d in zip(df["query"].iloc[i:i + B], df["text"].iloc[i:i + B])]
        enc = tok(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(model.device)
        logits = model(**enc).logits[:, -1, :]
        s = torch.log_softmax(torch.stack([logits[:, no], logits[:, yes]], 1).float(), 1)[:, 1].exp()
        scores += s.cpu().tolist()
        if i % 800 == 0:
            print(i, len(df), flush=True)
df["s"] = scores


def metrics(df):
    aucs, p10, p30 = [], [], []
    for _, g in df.groupby("qid"):
        pos, neg = g[g.label == 1].s.values, g[g.label == 0].s.values
        if len(pos) and len(neg):
            aucs.append((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        o = g.sort_values("s", ascending=False).label.values
        p10.append(o[:10].mean()); p30.append(o[:30].mean())
    return dict(auc=float(np.mean(aucs)), p10=float(np.mean(p10)), p30=float(np.mean(p30)))


res = {"model": MODEL, **metrics(df)}
print(json.dumps(res))
json.dump(res, open("/kaggle/working/eval_result.json", "w"))
df[["qid", "rank", "label", "s"]].to_parquet("/kaggle/working/eval_scores.parquet")
