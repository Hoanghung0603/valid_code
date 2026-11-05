"""Dense retrieval tiếng Việt bằng GreenNode-Embedding trên TOÀN BỘ doc VI (bổ sung ứng viên BM25 bỏ sót).

    python scripts/dense_vi.py encode --out data/cache/dense_gn --max-len 256     # GPU, checkpoint từng phần
    python scripts/dense_vi.py search --index data/cache/dense_gn --k 300 --out data/cache/dense_gn_top300.json
Văn bản mã hoá cho mỗi doc = title + 1500 ký tự đầu (73% chunk gold nằm trong 1500 ký tự đầu bài).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODEL = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"
PART = 25000


def doc_iter():
    seen = set()
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        t = t[t.lang == "vi"]
        for did, ti, tx in zip(t.doc_id.astype(str), t.title.fillna(""), t.text.fillna("")):
            if did not in seen:
                seen.add(did)
                yield did, f"{ti}\n{tx[:1500]}"


def encode(a):
    import torch
    from sentence_transformers import SentenceTransformer
    os.makedirs(a.out, exist_ok=True)
    m = SentenceTransformer(a.model, model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = a.max_len
    buf_ids, buf_tx, part = [], [], 0

    def flush():
        nonlocal buf_ids, buf_tx, part
        f = os.path.join(a.out, f"part{part:03d}.npy")
        if not os.path.exists(f):
            e = m.encode(buf_tx, normalize_embeddings=True, batch_size=a.batch).astype(np.float16)
            np.save(f, e)
            json.dump(buf_ids, open(f.replace(".npy", ".ids.json"), "w"))
            print(f"part {part} saved ({(part + 1) * PART})", flush=True)
        part += 1
        buf_ids, buf_tx = [], []

    for did, tx in doc_iter():
        buf_ids.append(did); buf_tx.append(tx)
        if len(buf_ids) == PART:
            flush()
    if buf_ids:
        flush()


def search(a):
    import torch
    from sentence_transformers import SentenceTransformer
    q = pd.read_parquet("data/ViBioMIR/query.parquet")
    m = SentenceTransformer(a.model, model_kwargs={"torch_dtype": torch.float16})
    if a.plans:   # đa truy vấn: câu gốc + tối đa 3 biến thể/ý con từ query plan; điểm doc = max qua các biến thể
        plans = {int(json.loads(l)["id"]): json.loads(l) for l in open(a.plans)}
        var = [[t] + [v for v in plans.get(int(i), {}).get("vi", [])[:3] if v.strip()] for i, t in zip(q.id, q["query"])]
        flat = [v for vs in var for v in vs]; owner = [r for r, vs in enumerate(var) for _ in vs]
        QV = torch.tensor(m.encode(flat, normalize_embeddings=True, batch_size=32)).float()
        owner = torch.tensor(owner)
    Q = torch.tensor(m.encode(q["query"].tolist(), normalize_embeddings=True, batch_size=32)).float()
    best_s = torch.full((len(q), 0), -1.0); best_i = []
    ids_all = []
    for f in sorted(glob.glob(os.path.join(a.index, "part*.npy"))):
        E = torch.tensor(np.load(f)).float()
        ids = json.load(open(f.replace(".npy", ".ids.json")))
        base = len(ids_all); ids_all += ids
        if a.plans:
            sv = QV @ E.T
            s = torch.full((len(q), E.shape[0]), -1.0).scatter_reduce(0, owner[:, None].expand(-1, E.shape[0]), sv, reduce="amax")
        else:
            s = Q @ E.T
        ts, ti = s.topk(min(a.k, s.shape[1]), dim=1)
        best_s = torch.cat([best_s, ts], 1)
        best_i.append(ti + base)
        bi = torch.cat(best_i, 1)
        o = best_s.topk(min(a.k, best_s.shape[1]), dim=1).indices
        best_s = best_s.gather(1, o); best_i = [bi.gather(1, o)]
    bi = best_i[0]
    out = {int(qid): [[float(best_s[r, j]), ids_all[int(bi[r, j])]] for j in range(bi.shape[1])] for r, qid in enumerate(q.id)}
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--out", required=True)
    e.add_argument("--max-len", type=int, default=256); e.add_argument("--batch", type=int, default=64)
    e.add_argument("--model", default=MODEL)
    s = sub.add_parser("search"); s.add_argument("--index", required=True); s.add_argument("--k", type=int, default=300)
    s.add_argument("--out", required=True); s.add_argument("--model", default=MODEL)
    s.add_argument("--plans", default="", help="query_plans.jsonl -> đa truy vấn (max)")
    a = ap.parse_args()
    encode(a) if a.cmd == "encode" else search(a)
