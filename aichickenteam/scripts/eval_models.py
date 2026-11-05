"""So sánh embedding / reranker bằng NHÃN THẬT từ kết quả chấm của BTC (không tốn lượt nộp).

Nhãn: mỗi chunk đã nộp có cờ `relevant` (overlap>=0.4 với gold) trong details.jsonl.
    python scripts/eval_models.py --sub submissions/v4_vi100_c60.zip --details data/eval/details_4286_v4.jsonl \
        --model BAAI/bge-reranker-v2-m3 --kind reranker
    python scripts/eval_models.py ... --model GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1 --kind embed
    python scripts/eval_models.py ... --kind bm25        # baseline = thứ tự hiện tại
Chỉ số: AUC (chunk trúng vs trượt, trong từng câu), P@10 và P@30 sau khi xếp lại. Model nào cao hơn bm25 → đáng nộp thử.
Chạy trên GPU (Kaggle/Colab). --max-q để thử nhanh.
"""
import argparse
import json
import zipfile

import numpy as np
import pandas as pd


def load(sub, details, max_q):
    p = json.loads(zipfile.ZipFile(sub).read(zipfile.ZipFile(sub).namelist()[0]))
    P = {e["id"]: e for e in p}
    q = dict(pd.read_parquet("data/ViBioMIR/query.parquet").values)
    rows = []
    for line in open(details):
        d = json.loads(line)
        qid = int(d["id"])
        chunks = P[qid]["relevant_chunks"]
        for pr in d["chunk"]["predictions"]:
            i = pr["chunk_order"]
            if i < len(chunks):
                rows.append((qid, q[qid], i, chunks[i]["chunk_text"], int(pr["relevant"])))
    df = pd.DataFrame(rows, columns=["qid", "query", "rank", "text", "label"])
    keep = df.groupby("qid").label.sum()
    keep = keep[keep > 0].index[:max_q] if max_q else keep[keep > 0].index   # chỉ câu có ít nhất 1 chunk trúng
    return df[df.qid.isin(keep)].reset_index(drop=True)


def metrics(df, score):
    df = df.assign(s=score)
    aucs, p10, p30 = [], [], []
    for _, g in df.groupby("qid"):
        pos, neg = g[g.label == 1].s.values, g[g.label == 0].s.values
        if len(pos) and len(neg):
            aucs.append((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        o = g.sort_values("s", ascending=False).label.values
        p10.append(o[:10].mean()); p30.append(o[:30].mean())
    return dict(auc=np.mean(aucs), p10=np.mean(p10), p30=np.mean(p30), n_q=df.qid.nunique(), n=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--details", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--kind", choices=["bm25", "reranker", "embed"], default="bm25")
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    df = load(a.sub, a.details, a.max_q)
    if a.kind == "bm25":
        score = -df["rank"].values.astype(float)
    elif a.kind == "reranker":
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(a.model, max_length=a.max_len, model_kwargs={"torch_dtype": "auto"})
        score = m.predict(list(zip(df["query"], df["text"])), batch_size=a.batch, show_progress_bar=True)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(a.model, model_kwargs={"torch_dtype": "auto"})
        m.max_seq_length = a.max_len
        qe = m.encode(df["query"].drop_duplicates().tolist(), normalize_embeddings=True, batch_size=a.batch)
        qmap = dict(zip(df["query"].drop_duplicates(), qe))
        de = m.encode(df["text"].tolist(), normalize_embeddings=True, batch_size=a.batch, show_progress_bar=True)
        score = (de * np.stack([qmap[x] for x in df["query"]])).sum(1)
    print(json.dumps({"model": a.model or "bm25-order", "kind": a.kind, **metrics(df, np.asarray(score))}))


if __name__ == "__main__":
    main()
