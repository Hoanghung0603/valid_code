"""Đặc trưng từ câu hỏi DỊCH sang ZH (kaggle_qtrans, Qwen2.5-7B): (1) BM25 jieba trên doc ZH (title + text[:3000]), (2) bge-m3 dense ZH-query↔ZH-doc
(dùng zh_emb.npy sẵn có). Out: parquet (qid, doc_id, bm25, bm25_rank, dzq, dzq_rank) cho hợp top-300 bm25 ∪ top-300 dzq mỗi câu.
    python scripts/zh_qfeat.py --qzh data/cache/qtrans_out/q_zh.json --feat data/cache/zhfeat3_out --out data/cache/zh_qfeat.parquet
"""
import argparse, json, re, sys
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--qzh", required=True); ap.add_argument("--feat", required=True); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=300)
a = ap.parse_args()
import jieba, bm25s
jieba.setLogLevel(60)
QZ = {int(k): v for k, v in json.load(open(a.qzh)).items()}
ids = json.load(open(f"{a.feat}/zh_ids.json"))
D = pd.read_parquet("data/cache/kg_zhdocs/zh_docs.parquet", columns=["doc_id", "title", "text"]); D["doc_id"] = D.doc_id.astype(str)
TX = dict(zip(D.doc_id, (D.title.fillna("") + "\n" + D.text.fillna("").str[:3000])))
STOP = set("的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 吗 呢 吧 啊 么 什么 怎么 如何 请问 可以 能 哪些 为什么".split())
tok = lambda s: [w for w in jieba.lcut_for_search(re.sub(r"\s+", " ", s.lower())) if w.strip() and w not in STOP and not re.fullmatch(r"\W+", w)]
docs = [tok(TX.get(d, "")) for d in ids]; print("tokenized docs", len(docs), flush=True)
ret = bm25s.BM25(k1=1.2, b=0.75); ret.index(docs, show_progress=False)
qids = sorted(QZ); qt = [tok(QZ[q]) for q in qids]
res, sc = ret.retrieve(qt, k=a.top, show_progress=False, n_threads=2)
rows = {}
for j, q in enumerate(qids):
    for r, (i, s) in enumerate(zip(res[j], sc[j])):
        rows[(q, ids[i])] = [float(s), r + 1, np.nan, np.nan]
print("bm25 done", flush=True)
from sentence_transformers import SentenceTransformer
import torch
m = SentenceTransformer("data/models/bge-m3-st", device="cuda", model_kwargs={"dtype": torch.float16}); m.max_seq_length = 256
QE = m.encode([QZ[q] for q in qids], batch_size=32, normalize_embeddings=True).astype(np.float16); del m; torch.cuda.empty_cache()
E = np.load(f"{a.feat}/zh_emb.npy"); S = (QE @ E.T).astype(np.float32)
for j, q in enumerate(qids):
    top = np.argpartition(-S[j], a.top)[:a.top]; order = top[np.argsort(-S[j][top])]
    for r, i in enumerate(order):
        rows.setdefault((q, ids[i]), [0.0, 999, np.nan, np.nan])[2:] = [float(S[j, i]), r + 1]
P = pd.DataFrame([(q, d, *v) for (q, d), v in rows.items()], columns=["qid", "doc_id", "bm25", "bm25_rank", "dzq", "dzq_rank"])
# điểm dzq cho cặp chỉ có trong bm25
qi = {q: j for j, q in enumerate(qids)}; di = {d: i for i, d in enumerate(ids)}
miss = P.dzq.isna(); P.loc[miss, "dzq"] = [float(S[qi[q], di[d]]) for q, d in zip(P.qid[miss], P.doc_id[miss])]; P["dzq_rank"] = P.dzq_rank.fillna(999)
P.to_parquet(a.out, index=False); print("wrote", a.out, len(P))
