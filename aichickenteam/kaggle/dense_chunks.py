"""Kaggle (GPU T4x2): dense retrieval CẤP ĐOẠN bằng GreenNode trên toàn bộ doc VI.

Mỗi doc → tối đa MAX_W cửa sổ ~1600 ký tự (gộp đoạn '\\n\\n'). Encode tất cả cửa sổ, tìm top-K cửa sổ/câu,
gộp lên doc bằng max, giữ cửa sổ tốt nhất làm chunk_text (nguyên văn).
Input : /kaggle/input/<dataset>/corpus-*.parquet, query.parquet
Output: /kaggle/working/dense_chunks_top.json  {qid: [[score, doc_id, chunk_text], ...]} (top N doc)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

DATA = os.environ.get("R2AI_DATA", "/kaggle/input")
OUT = os.environ.get("R2AI_OUT", "/kaggle/working")
_ft = glob.glob("/kaggle/input/**/gn-ft/config.json", recursive=True)   # model fine-tune (kernel_sources r2ai-finetune-gn)
MODEL = os.path.dirname(_ft[0]) if _ft else "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
MAX_W, TARGET, TOPK_W, TOPN_DOC = 6, 1600, 1500, 300


def windows(text):
    paras, pos = [], 0
    for part in text.split("\n\n"):
        s = text.find(part, pos); e = s + len(part); pos = e
        if part.strip():
            paras.append((s, e))
    out, i = [], 0
    while i < len(paras) and len(out) < MAX_W:
        s, e = paras[i]; j = i
        while e - s < TARGET and j + 1 < len(paras):
            j += 1; e = paras[j][1]
        out.append(text[s:e]); i = j + 1
    return out


def main():
    from sentence_transformers import SentenceTransformer
    q = pd.read_parquet(glob.glob(f"{DATA}/**/query.parquet", recursive=True)[0])
    m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 512
    Q = torch.tensor(m.encode(q["query"].tolist(), normalize_embeddings=True, batch_size=64)).float().cuda()
    best_s = torch.full((len(q), 0), -1.0, device="cuda"); best_i = torch.zeros((len(q), 0), dtype=torch.long, device="cuda")
    meta = []                      # global window index -> (doc_id, file_idx, start_char) ; text giữ ở ổ đĩa để tiết kiệm RAM
    texts_all = []
    seen = set()
    files = sorted(glob.glob(f"{DATA}/**/corpus-*.parquet", recursive=True))
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        t = t[t.lang == "vi"]
        W, D = [], []
        for did, ti, tx in zip(t.doc_id.astype(str), t.title.fillna(""), t.text.fillna("")):
            if did in seen or not tx:
                continue
            seen.add(did)
            for w in windows(tx):
                W.append(w); D.append(did)
        if not W:
            continue
        inp = [f"{w}" for w in W]
        E = m.encode(inp, batch_size=64, normalize_embeddings=True)
        E = torch.tensor(E).float().cuda()
        base = len(meta)
        meta += D; texts_all += W
        s = Q @ E.T
        k = min(TOPK_W, s.shape[1])
        ts, ti = s.topk(k, dim=1)
        best_s = torch.cat([best_s, ts], 1); best_i = torch.cat([best_i, ti + base], 1)
        o = best_s.topk(min(TOPK_W, best_s.shape[1]), dim=1).indices
        best_s = best_s.gather(1, o); best_i = best_i.gather(1, o)
        print(f"file {fi + 1}/{len(files)} windows {len(meta)}", flush=True)
    res = {}
    bs, bi = best_s.cpu().numpy(), best_i.cpu().numpy()
    for r, qid in enumerate(q.id):
        docs = {}
        for s, i in zip(bs[r], bi[r]):
            d = meta[i]
            if d not in docs:
                docs[d] = (float(s), texts_all[i])
            if len(docs) >= TOPN_DOC:
                break
        res[int(qid)] = [[s, d, w] for d, (s, w) in docs.items()]
    json.dump(res, open(f"{OUT}/dense_chunks_top.json", "w"), ensure_ascii=False)
    print("done", len(res))


if __name__ == "__main__":
    main()
