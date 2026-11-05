"""Kaggle GPU: tính mọi đặc trưng LTR cho doc MỚI crawl (dataset r2ai-newdocs từ scripts/newdocs_prep.py) để gộp vào pipeline cũ.
Văn bản giống hệt pipeline cũ: doc = title\\n text[:1500] (dense_vi / cand_docs), cửa sổ = kaggle/dense_chunks.windows (≤6 × ~1600 ký tự).
Model fine-tune lấy từ output kernel r2ai-savemodels (train trên tất cả nhãn).
Out (/kaggle/working): gn_docs.npy + gn_ids.json (GN doc-level, cho PRF), d1_new.json / d2_new.json / d3_new.json
({qid: [[s, doc, (cửa sổ)], ...]} top-300 trong doc mới), e5_docs.npy + e5_ids.json, pairs.parquet (qid, doc_id, rr, rr2, rr3, rr4, gnft)."""
import glob
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

W_ = "/kaggle/working"
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
docs = pd.read_parquet(G("new_docs.parquet")); q = pd.read_parquet(G("query.parquet")); thr = json.load(open(G("thr.json")))
QIDS = [str(int(x)) for x in q.id]; QT = q["query"].tolist()
_c = [p for p in glob.glob("/kaggle/input/**/modules.json", recursive=True)
      if os.path.exists(os.path.join(os.path.dirname(p), "model.safetensors")) and "/models/" not in p]
GN = os.path.dirname(_c[0])
MD = os.path.dirname(G("ce_doc/config.json"))[:-len("/ce_doc")]
print("GN", GN, "MD", MD, flush=True)
ids = docs.doc_id.astype(str).tolist()
dtext = [f"{ti}\n{tx[:1500]}" for ti, tx in zip(docs.title.fillna(""), docs.text.fillna(""))]
MAX_W, TARGET, TOPN, NPAIR = 6, 1600, 300, 100
t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:.0f}s]", *x, flush=True)


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


WT, WD = [], []
for di, tx in enumerate(docs.text.fillna("")):
    for w in windows(tx):
        WT.append(w); WD.append(di)
WD = np.array(WD)
log("docs", len(ids), "windows", len(WT))


def top_lists(S, key):
    """S: (nq, ndoc) tensor -> {qid: [[s, doc]]} top-300 doc mới, và doc vượt ngưỡng list cũ (thr[key])."""
    k = min(TOPN, S.shape[1]); ts, ti = S.topk(k, dim=1); ts, ti = ts.cpu().numpy(), ti.cpu().numpy()
    out, ok = {}, {}
    for r, qid in enumerate(QIDS):
        out[qid] = [[float(s), ids[i]] for s, i in zip(ts[r], ti[r])]
        ok[qid] = [ids[i] for s, i in zip(ts[r], ti[r]) if s > thr.get(qid, [-1, -1, -1])[key]]
    return out, ok


def doc_max(model, ml, qprompt=None):
    """Điểm cửa sổ tốt nhất mỗi (câu, doc) + chỉ số cửa sổ đó."""
    m = SentenceTransformer(model, model_kwargs={"torch_dtype": torch.float16}, device="cuda"); m.max_seq_length = ml
    Q = torch.tensor(m.encode(QT, normalize_embeddings=True, batch_size=64, prompt=qprompt)).half().cuda()
    E = torch.tensor(m.encode(WT, normalize_embeddings=True, batch_size=64, show_progress_bar=False)).half().cuda()
    del m; torch.cuda.empty_cache()
    best = torch.full((len(QIDS), len(ids)), -1.0, device="cuda"); arg = torch.zeros((len(QIDS), len(ids)), dtype=torch.long, device="cuda")
    wd = torch.tensor(WD, device="cuda")
    for i in range(0, len(WT), 20000):
        s = (Q @ E[i:i + 20000].T).float(); d = wd[i:i + 20000]
        # mỗi doc ≤6 cửa sổ: cập nhật theo "thứ tự cửa sổ trong doc" ⇒ trong 1 lượt không có doc trùng (gán chỉ số an toàn)
        rank_in_doc = torch.tensor(pd.Series(WD[i:i + 20000]).groupby(WD[i:i + 20000]).cumcount().values, device="cuda")
        for r in range(int(rank_in_doc.max()) + 1):
            sel = (rank_in_doc == r).nonzero().view(-1); dd = d[sel]; ss = s[:, sel]
            cur = best[:, dd]; u = ss > cur
            best[:, dd] = torch.where(u, ss, cur); arg[:, dd] = torch.where(u, sel + i, arg[:, dd])
    return best, arg


res = {}
# 1) GreenNode doc-level (như dense_vi.py: doc max_len 256, câu hỏi không prompt)
m = SentenceTransformer(GN, model_kwargs={"torch_dtype": torch.float16}, device="cuda"); m.max_seq_length = 256
E1 = m.encode(dtext, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
Q1 = m.encode(QT, normalize_embeddings=True, batch_size=64)
np.save(f"{W_}/gn_docs.npy", E1.astype(np.float16)); json.dump(ids, open(f"{W_}/gn_ids.json", "w"))
d1, ok1 = top_lists(torch.tensor(Q1).float().cuda() @ torch.tensor(E1).float().cuda().T, 0)
json.dump(d1, open(f"{W_}/d1_new.json", "w")); del m; torch.cuda.empty_cache(); log("d1 done")
# 2) GreenNode cấp đoạn (d3) – cửa sổ tốt nhất cũng là văn bản cho CE cửa sổ
b3, a3 = doc_max(GN, 512)
a3n = a3.cpu().numpy()
k = min(TOPN, len(ids)); ts, ti = b3.topk(k, dim=1); ts, ti = ts.cpu().numpy(), ti.cpu().numpy()
d3 = {qid: [[float(s), ids[i], WT[a3n[r, i]]] for s, i in zip(ts[r], ti[r])] for r, qid in enumerate(QIDS)}
json.dump(d3, open(f"{W_}/d3_new.json", "w"), ensure_ascii=False); log("d3 done")
# 3) bge-m3 cấp đoạn (d2)
b2, a2 = doc_max("BAAI/bge-m3", 512); a2n = a2.cpu().numpy()
ts2, ti2 = b2.topk(k, dim=1); ts2, ti2 = ts2.cpu().numpy(), ti2.cpu().numpy()
d2 = {qid: [[float(s), ids[i], WT[a2n[r, i]]] for s, i in zip(ts2[r], ti2[r])] for r, qid in enumerate(QIDS)}
ok2 = {qid: [ids[i] for s, i in zip(ts2[r], ti2[r]) if s > thr.get(qid, [-1, -1, -1])[1]] for r, qid in enumerate(QIDS)}
json.dump(d2, open(f"{W_}/d2_new.json", "w"), ensure_ascii=False); del b2, a2; torch.cuda.empty_cache(); log("d2 done")
# 4) e5 (như colab/encode_e5.py)
m = SentenceTransformer("intfloat/multilingual-e5-large-instruct", model_kwargs={"torch_dtype": torch.float16}, device="cuda"); m.max_seq_length = 512
np.save(f"{W_}/e5_docs.npy", m.encode(dtext, normalize_embeddings=True, batch_size=64, show_progress_bar=False).astype(np.float16))
json.dump(ids, open(f"{W_}/e5_ids.json", "w")); del m; torch.cuda.empty_cache(); log("e5 done")
# 5) cặp cần chấm CE: doc mới vượt ngưỡng của ít nhất 1 list cũ (d1 / d2 / d3), tối đa NPAIR/câu theo điểm GN đoạn
ok3 = {qid: [ids[i] for s, i in zip(ts[r], ti[r]) if s > thr.get(qid, [-1, -1, -1])[2]] for r, qid in enumerate(QIDS)}
pos = {d: i for i, d in enumerate(ids)}; qi = {qid: r for r, qid in enumerate(QIDS)}
rows = []
for qid in QIDS:
    cand = set(ok1[qid]) | set(ok2[qid]) | set(ok3[qid])
    cand = sorted(cand, key=lambda d: -float(b3[qi[qid], pos[d]]))[:NPAIR]
    rows += [(int(qid), d, dtext[pos[d]][:1500], WT[a3n[qi[qid], pos[d]]]) for d in cand]
P = pd.DataFrame(rows, columns=["qid", "doc_id", "doc", "win"]); log("pairs", len(P))
QS = dict(zip(QIDS, QT)); qq = [QS[str(x)] for x in P.qid]
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
P["rr"] = ce.predict(list(zip(qq, P.doc.str[:1500])), batch_size=64, show_progress_bar=False)
P["rr2"] = ce.predict(list(zip(qq, P.win.str[:1600])), batch_size=64, show_progress_bar=False); del ce; torch.cuda.empty_cache(); log("rr done")
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def ft_score(path, texts):
    tok = AutoTokenizer.from_pretrained(path); mm = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(qq[i:i + 64], texts[i:i + 64], truncation="only_second", max_length=384, padding=True, return_tensors="pt")
            out += mm(**{k: v.cuda() for k, v in enc.items()}).logits.view(-1).float().cpu().tolist()
    del mm; torch.cuda.empty_cache(); return out


P["rr3"] = ft_score(f"{MD}/ce_doc", P.doc.tolist()); P["rr4"] = ft_score(f"{MD}/ce_win", P.win.tolist()); log("ft ce done")
_ce2 = glob.glob("/kaggle/input/**/ce2_all/config.json", recursive=True)   # ce2 train trên tất cả nhãn (kernel r2ai-ce2) → rr5
if _ce2:
    P["rr5"] = ft_score(os.path.dirname(_ce2[0]), P.doc.tolist()); log("ce2 done")
m = SentenceTransformer(f"{MD}/gn", device="cuda"); m.max_seq_length = 512
De = m.encode(P.doc.str[:1700].tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
Qe = dict(zip(QIDS, m.encode(QT, normalize_embeddings=True, batch_size=64)))
P["gnft"] = [float(De[i] @ Qe[str(x)]) for i, x in enumerate(P.qid)]
P.drop(columns=["doc", "win"]).to_parquet(f"{W_}/pairs.parquet", index=False); log("ALLDONE", len(P))
