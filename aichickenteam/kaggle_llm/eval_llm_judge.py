"""Kaggle GPU: LLM mở (Qwen3-8B, tắt thinking) làm giám khảo pointwise "đoạn có liên quan câu hỏi không" → P(yes).
Đánh giá trên cặp có nhãn thật (eval_pairs: 12k cặp) – so với GreenNode AUC .787 / P@10 .232, Qwen3-Reranker-4B .752/.210."""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("LLM", "Qwen/Qwen3-8B")
NQ = int(os.environ.get("NQ", "200"))   # số câu đánh giá (giữ trọn cặp của câu)
df = pd.read_parquet(glob.glob("/kaggle/input/**/eval_pairs.parquet", recursive=True)[0])
qs = sorted(df.qid.unique())[:NQ]; df = df[df.qid.isin(qs)].reset_index(drop=True)
print("pairs", len(df), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()
yes, no = tok.encode("Có", add_special_tokens=False)[0], tok.encode("Không", add_special_tokens=False)[0]
SYS = ("Bạn là chuyên gia y khoa đánh giá kết quả tìm kiếm. Cho một câu hỏi y tế của người dùng và một đoạn trích từ bài báo/trang sức khoẻ "
       "tiếng Việt. Đoạn trích được coi là LIÊN QUAN nếu nó chứa thông tin giúp trả lời câu hỏi hoặc bàn về đúng bệnh/triệu chứng/thuốc/"
       "vấn đề mà câu hỏi đề cập. Chỉ trả lời một từ: Có hoặc Không.")


def fmt(q, d):
    m = [{"role": "system", "content": SYS}, {"role": "user", "content": f"Câu hỏi: {q}\n\nĐoạn trích: {d[:1800]}\n\nĐoạn trích có liên quan không?"}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)


scores = []; B = 8
with torch.no_grad():
    for i in range(0, len(df), B):
        batch = [fmt(q, d) for q, d in zip(df["query"].iloc[i:i + B], df["text"].iloc[i:i + B])]
        enc = tok(batch, padding=True, truncation=True, max_length=1280, return_tensors="pt").to(model.device)
        lg = model(**enc).logits[:, -1, :].float()
        scores += torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1), 1)[:, 1].exp().cpu().tolist()
        if i % 400 == 0:
            print(i, len(df), flush=True)
df["s"] = scores


def metrics(df, col):
    aucs, p10, p30 = [], [], []
    for _, g in df.groupby("qid"):
        pos, neg = g[g.label == 1][col].values, g[g.label == 0][col].values
        if len(pos) and len(neg):
            aucs.append((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        o = g.sort_values(col, ascending=False).label.values
        p10.append(o[:10].mean()); p30.append(o[:30].mean())
    return dict(auc=float(np.mean(aucs)), p10=float(np.mean(p10)), p30=float(np.mean(p30)))


res = {"model": MODEL, "nq": NQ, "llm": metrics(df, "s")}
if "gn" in df.columns:
    res["gn_same_subset"] = metrics(df, "gn")
print(json.dumps(res))
json.dump(res, open("/kaggle/working/eval_result.json", "w"))
df.drop(columns=["text", "query"]).to_parquet("/kaggle/working/eval_scores.parquet")
