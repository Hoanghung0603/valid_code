"""Rerank ứng viên VI (từ pipeline_v2 --dump-preds) bằng cross-encoder hoặc embedding, rồi ghi zip nộp.

    python scripts/rerank_vi.py score --preds data/cache/vi_preds_300.jsonl --kind reranker \
        --model BAAI/bge-reranker-v2-m3 --out data/cache/rr_bge.npy          # GPU, có thể chạy trên Kaggle/Colab
    python scripts/rerank_vi.py write --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_bge.npy \
        --alpha 0.5 --n-docs 120 --n-chunks 60 --out submissions/v5_rr.zip
Điểm cuối = alpha * rank_norm(model) + (1-alpha) * rank_norm(BM25)  (trộn theo thứ hạng, bền với thang điểm khác nhau).
"""
import argparse
import json
import zipfile

import numpy as np
import pandas as pd


def load_preds(path):
    with open(path) as f:
        txt = f.read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def pairs(preds, top=0):
    q = dict(pd.read_parquet("data/ViBioMIR/query.parquet").values)
    out = []
    for e in preds:
        ch = {str(c["doc_id"]): c["chunk_text"] for c in e["relevant_chunks"]}
        docs = e["relevant_docs"][:top] if top else e["relevant_docs"]
        for r, d in enumerate(docs):
            out.append((int(e["id"]), q[int(e["id"])], r, str(d), ch.get(str(d), "")))
    return out


def score(a):
    P = pairs(load_preds(a.preds), a.top)
    qs, ts = [p[1] for p in P], [p[4] for p in P]
    import torch
    if a.kind == "embed":
        # fp16 + lưu từng phần (checkpoint) để chạy lại không mất công
        from sentence_transformers import SentenceTransformer
        import os
        m = SentenceTransformer(a.model, model_kwargs={"torch_dtype": torch.float16})
        m.max_seq_length = a.max_len
        uq = list(dict.fromkeys(qs))
        qe = dict(zip(uq, m.encode(uq, normalize_embeddings=True, batch_size=a.batch)))
        parts, step = [], 20000
        for i in range(0, len(ts), step):
            f = f"{a.out}.part{i // step:03d}.npy"
            if not os.path.exists(f):
                de = m.encode(ts[i:i + step], normalize_embeddings=True, batch_size=a.batch)
                np.save(f, (de * np.stack([qe[x] for x in qs[i:i + step]])).sum(1).astype(np.float32))
                print(f"part {i // step} done ({i + step}/{len(ts)})", flush=True)
            parts.append(np.load(f))
        np.save(a.out, np.concatenate(parts))
        print("saved", a.out, len(ts))
        return
    if a.kind == "reranker":
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(a.model, max_length=a.max_len, model_kwargs={"torch_dtype": "auto"})
        s = m.predict(list(zip(qs, ts)), batch_size=a.batch, show_progress_bar=True)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(a.model, model_kwargs={"torch_dtype": "auto"})
        m.max_seq_length = a.max_len
        uq = list(dict.fromkeys(qs))
        qe = dict(zip(uq, m.encode(uq, normalize_embeddings=True, batch_size=a.batch)))
        de = m.encode(ts, normalize_embeddings=True, batch_size=a.batch, show_progress_bar=True)
        s = (de * np.stack([qe[x] for x in qs])).sum(1)
    np.save(a.out, np.asarray(s, np.float32))
    print("saved", a.out, len(s))


def write(a):
    preds = load_preds(a.preds)
    P = pairs(preds, a.top)
    S = np.load(a.scores)
    assert len(S) == len(P), (len(S), len(P))
    df = pd.DataFrame(P, columns=["qid", "q", "bm25_rank", "doc", "text"]).assign(s=S)
    out = []
    for qid, g in df.groupby("qid", sort=True):
        n = len(g)
        rm = g.s.rank(ascending=False).values / n          # 0..1, nhỏ = tốt
        rb = (g.bm25_rank.values + 1) / n
        g = g.assign(f=a.alpha * rm + (1 - a.alpha) * rb).sort_values("f")
        docs = g.doc.tolist()[:a.n_docs]
        chunks = [{"doc_id": int(d), "chunk_text": t} for d, t in zip(g.doc[:a.n_chunks], g.text[:a.n_chunks]) if t]
        out.append({"id": int(qid), "relevant_docs": [int(d) for d in docs], "relevant_chunks": chunks})
    ids = set(pd.read_parquet("data/ViBioMIR/query.parquet").id.astype(int))
    have = {e["id"] for e in out}
    out += [{"id": i, "relevant_docs": [], "relevant_chunks": []} for i in sorted(ids - have)]
    out.sort(key=lambda e: e["id"])
    assert len(out) == 1200
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("submission.json", json.dumps(out, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--preds", required=True); s.add_argument("--kind", choices=["reranker", "embed"], required=True)
    s.add_argument("--model", required=True); s.add_argument("--out", required=True)
    s.add_argument("--max-len", type=int, default=512); s.add_argument("--batch", type=int, default=16)
    s.add_argument("--top", type=int, default=0, help="chỉ chấm top-N ứng viên BM25 mỗi câu")
    w = sub.add_parser("write")
    w.add_argument("--preds", required=True); w.add_argument("--scores", required=True); w.add_argument("--out", required=True)
    w.add_argument("--alpha", type=float, default=0.5); w.add_argument("--n-docs", type=int, default=120)
    w.add_argument("--n-chunks", type=int, default=60)
    w.add_argument("--top", type=int, default=0, help="phải giống --top lúc score")
    a = ap.parse_args()
    score(a) if a.cmd == "score" else write(a)
