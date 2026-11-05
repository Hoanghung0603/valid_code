"""Kaggle GPU (T4×2): dịch 1200 câu hỏi VI → ZH giản thể bằng Qwen2.5-7B-Instruct (mở, 7B, 09/2024) – gợi ý Query Translation (Mentor Day 3).
Giữ nguyên thuật ngữ y khoa; chỉ trả bản dịch. Out: /kaggle/working/q_zh.json {id: text}."""
import glob, json, time
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
Q = pd.read_parquet(glob.glob("/kaggle/input/**/query.parquet", recursive=True)[0])
NAME = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(NAME); tok.padding_side = "left"
m = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16, device_map="auto").eval(); log("loaded")
SYS = "你是专业的医学翻译。把用户给出的越南语医学问题准确翻译成简体中文，保留所有症状、疾病、药物、检查名称等医学术语的准确含义，不要回答问题，不要添加解释，只输出译文。"
def prompt(q):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
out = {}; qs = list(zip(Q.id.astype(int), Q["query"])); qs.sort(key=lambda x: len(x[1]))
B = 16
for i in range(0, len(qs), B):
    b = qs[i:i + B]; enc = tok([prompt(q) for _, q in b], return_tensors="pt", padding=True).to(m.device)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=min(512, 64 + 2 * max(len(q) for _, q in b) // 3), do_sample=False, repetition_penalty=1.05)
    for (qid, _), seq in zip(b, g[:, enc["input_ids"].shape[1]:]):
        out[qid] = tok.decode(seq, skip_special_tokens=True).strip()
    if i % 160 == 0: log(i, out[b[0][0]][:80])
json.dump(out, open("/kaggle/working/q_zh.json", "w"), ensure_ascii=False); log("ALLDONE", len(out))
