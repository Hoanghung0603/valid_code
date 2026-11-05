"""Colab T4: HyDE – LLM mở (Qwen3-4B-Instruct-2507, ≤15B, phát hành 07/2025) viết đoạn văn kiểu bài báo sức khoẻ trả lời từng câu hỏi
(dùng thuật ngữ y khoa) → /content/hyde.jsonl {id, text}. Đoạn này được mã hoá bằng GreenNode để truy hồi / làm đặc trưng LTR."""
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = os.environ.get("LLM", "Qwen/Qwen3-4B-Instruct-2507")
q = pd.read_parquet("/content/query.parquet")
tok = AutoTokenizer.from_pretrained(NAME, padding_side="left")
m = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float16, device_map="cuda").eval()
SYS = ("Bạn là bác sĩ viết bài cho chuyên mục sức khoẻ của một báo điện tử Việt Nam. Với câu hỏi của bạn đọc, hãy viết MỘT đoạn văn "
       "khoảng 120-160 từ như trích từ bài báo trả lời câu hỏi đó: nêu tên bệnh/tình trạng có khả năng, nguyên nhân, triệu chứng, "
       "cách chẩn đoán và điều trị, dùng thuật ngữ y khoa tiếng Việt chuẩn (kèm tên thuốc/xét nghiệm nếu phù hợp). Không chào hỏi, không ghi chú.")
OUT = "/content/hyde.jsonl"
done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
todo = [(int(i), t) for i, t in zip(q.id, q["query"]) if int(i) not in done]
B = 24
with open(OUT, "a") as f, torch.no_grad():
    for s in range(0, len(todo), B):
        part = todo[s:s + B]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": t}], tokenize=False,
                                           add_generation_prompt=True) for _, t in part]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
        out = m.generate(**enc, max_new_tokens=280, do_sample=False, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
        for (i, _), o in zip(part, out):
            f.write(json.dumps({"id": i, "text": tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()}, ensure_ascii=False) + "\n")
        f.flush()
        print(f"{s + len(part)}/{len(todo)}", flush=True)
print("HYDE DONE", flush=True)
