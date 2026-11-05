"""Benchmark bge-m3 (dense, CLS, fp16) tren GPU: do thong luong embed doc.

Dung: python embed_bench.py --docs bench_docs.jsonl --max-len 512 --batch 16 --limit 1000
Chi dat tai GPU; CPU chi de tokenize. Ghi so do ra stdout (chay qua .bat co log).
"""
import argparse
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    texts, ids = [], []
    with open(a.docs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"])
            texts.append(r["text"])
            if len(texts) >= a.limit:
                break
    print(f"docs={len(texts)}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    print(f"load_model={time.time() - t0:.1f}s vram={torch.cuda.memory_allocated() / 2**20:.0f}MB", flush=True)

    t0 = time.time()
    enc = tok(texts, truncation=True, max_length=a.max_len)["input_ids"]
    lens = [len(x) for x in enc]
    total_tokens = sum(lens)
    print(f"tokenize={time.time() - t0:.1f}s tokens={total_tokens} mean_len={total_tokens / len(lens):.0f}", flush=True)

    order = sorted(range(len(enc)), key=lambda i: lens[i])  # giam padding
    pad = tok.pad_token_id
    out = [None] * len(enc)

    def run(batch_idx):
        m = max(lens[i] for i in batch_idx)
        ids_t = torch.full((len(batch_idx), m), pad, dtype=torch.long)
        att = torch.zeros((len(batch_idx), m), dtype=torch.long)
        for r, i in enumerate(batch_idx):
            ids_t[r, : lens[i]] = torch.tensor(enc[i])
            att[r, : lens[i]] = 1
        with torch.inference_mode():
            h = model(input_ids=ids_t.cuda(), attention_mask=att.cuda()).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
        return h.cpu()

    run(order[: a.batch])  # khoi dong
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for s in range(0, len(order), a.batch):
        b = order[s : s + a.batch]
        e = run(b)
        for r, i in enumerate(b):
            out[i] = e[r]
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"embed={dt:.1f}s docs/s={len(order) / dt:.1f} tokens/s={total_tokens / dt:.0f} "
          f"peak_vram={torch.cuda.max_memory_allocated() / 2**20:.0f}MB batch={a.batch} max_len={a.max_len}", flush=True)
    print(f"ETA 112269 doc @512: {112269 / (len(order) / dt) / 60:.0f} phut (mean_len bench={total_tokens / len(lens):.0f})", flush=True)

    if a.out:
        torch.save({"ids": ids, "emb": torch.stack(out).half()}, a.out)
        print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
