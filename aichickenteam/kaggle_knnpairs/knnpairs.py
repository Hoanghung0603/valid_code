"""Kaggle GPU: chấm mọi đặc trưng LTR cho cặp (câu, doc) CHO TRƯỚC – doc mở rộng theo láng giềng (scripts/knn_expand.py --kaggle).
Văn bản như kaggle_newdocs: doc = title\\n text[:1500], cửa sổ = ≤6 × ~1600 ký tự; cửa sổ cho CE = cửa sổ GN tốt nhất của cặp.
Out (/kaggle/working/pairs.parquet): qid, doc_id, d1 (GN doc), d3 (GN cửa sổ max), d2 (bge-m3 cửa sổ max), rr, rr2, rr3, rr4, gnft."""
import glob
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

W_ = "/kaggle/working"
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
docs = pd.read_parquet(G("exp_docs.parquet")); P = pd.read_parquet(G("exp_pairs.parquet")); q = pd.read_parquet(G("query.parquet"))
_c = [p for p in glob.glob("/kaggle/input/**/modules.json", recursive=True)
      if os.path.exists(os.path.join(os.path.dirname(p), "model.safetensors")) and "/models/" not in p]
GN = os.path.dirname(_c[0])
MD = os.path.dirname(G("ce_doc/config.json"))[:-len("/ce_doc")]
docs["doc_id"] = docs.doc_id.astype(str); P["doc_id"] = P.doc_id.astype(str)
P = P[P.doc_id.isin(set(docs.doc_id))].reset_index(drop=True)
QS = dict(zip(q.id.astype(int), q["query"])); qq = [QS[int(x)] for x in P.qid]
pos = {d: i for i, d in enumerate(docs.doc_id)}
dtext = [f"{ti}\n{tx[:1500]}" for ti, tx in zip(docs.title.fillna(""), docs.text.fillna(""))]
MAX_W, TARGET = 6, 1600
t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:.0f}s]", *x, flush=True)


def windows(text):
    paras, p0 = [], 0
    for part in text.split("\n\n"):
        s = text.find(part, p0); e = s + len(part); p0 = e
        if part.strip():
            paras.append((s, e))
    out, i = [], 0
    while i < len(paras) and len(out) < MAX_W:
        s, e = paras[i]; j = i
        while e - s < TARGET and j + 1 < len(paras):
            j += 1; e = paras[j][1]
        out.append(text[s:e]); i = j + 1
    return out or [text[:TARGET]]


WT, WD = [], []
for di, tx in enumerate(docs.text.fillna("")):
    for w in windows(tx):
        WT.append(w); WD.append(di)
WD = np.array(WD); wins = pd.Series(range(len(WD))).groupby(WD).apply(list).to_dict()
log("docs", len(docs), "windows", len(WT), "pairs", len(P))
di = P.doc_id.map(pos).values


def enc(m, texts, **kw):
    return torch.tensor(m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False, **kw)).half().cuda()


def win_max(model):
    """(điểm cửa sổ tốt nhất, chỉ số cửa sổ) cho từng cặp."""
    m = SentenceTransformer(model, model_kwargs={"torch_dtype": torch.float16}, device="cuda"); m.max_seq_length = 512
    Q = enc(m, list(QS.values())); qi = {k: i for i, k in enumerate(QS)}; E = enc(m, WT); del m; torch.cuda.empty_cache()
    best, arg = np.empty(len(P), np.float32), np.empty(len(P), np.int64)
    for r, (qid, d) in enumerate(zip(P.qid, di)):
        w = wins[d]; s = (E[w] @ Q[qi[int(qid)]]).float().cpu().numpy(); k = int(s.argmax()); best[r], arg[r] = s[k], w[k]
    return best, arg


# 1) GN doc-level (max_len 256, như dense_vi.py)
m = SentenceTransformer(GN, model_kwargs={"torch_dtype": torch.float16}, device="cuda"); m.max_seq_length = 256
E1 = enc(m, dtext); Q1 = enc(m, list(QS.values())); qi = {k: i for i, k in enumerate(QS)}; del m; torch.cuda.empty_cache()
P["d1"] = [float(E1[d] @ Q1[qi[int(x)]]) for x, d in zip(P.qid, di)]; log("d1 done")
P["d3"], a3 = win_max(GN); log("d3 done")
P["d2"], _ = win_max("BAAI/bge-m3"); log("d2 done")
win = [WT[i] for i in a3]; doc = [dtext[d] for d in di]
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
P["rr"] = ce.predict(list(zip(qq, [x[:1500] for x in doc])), batch_size=64, show_progress_bar=False)
P["rr2"] = ce.predict(list(zip(qq, [x[:1600] for x in win])), batch_size=64, show_progress_bar=False); del ce; torch.cuda.empty_cache(); log("rr done")
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def ft_score(path, texts):
    tok = AutoTokenizer.from_pretrained(path); mm = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            e = tok(qq[i:i + 64], texts[i:i + 64], truncation="only_second", max_length=384, padding=True, return_tensors="pt")
            out += mm(**{k: v.cuda() for k, v in e.items()}).logits.view(-1).float().cpu().tolist()
    del mm; torch.cuda.empty_cache(); return out


P["rr3"] = ft_score(f"{MD}/ce_doc", doc); P["rr4"] = ft_score(f"{MD}/ce_win", win); log("ft ce done")
m = SentenceTransformer(f"{MD}/gn", device="cuda"); m.max_seq_length = 512
De = m.encode([x[:1700] for x in doc], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
Qe = dict(zip(QS, m.encode(list(QS.values()), normalize_embeddings=True, batch_size=64)))
P["gnft"] = [float(De[i] @ Qe[int(x)]) for i, x in enumerate(P.qid)]
P.to_parquet(f"{W_}/pairs.parquet", index=False); log("ALLDONE", len(P))
