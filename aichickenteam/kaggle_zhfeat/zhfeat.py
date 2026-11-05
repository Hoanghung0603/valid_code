"""Kaggle GPU (T4): đặc trưng cho doc ZH (bộ 235k của corpus mới, cào lại 20/10) với câu hỏi VI (cross-lingual).
1) bge-m3 dense: doc = title\\n text[:1500] (max_len 512), top-300/câu → zh_dense.json {qid: [[s, doc_id], ...]}
2) top-NCE/câu: bge-reranker-v2-m3 zero-shot (ce0) + ce2_all (fine-tune VI, kernel r2ai-ce2) (ce2) → zh_pairs.parquet (qid, doc_id, dense, rank, ce0, ce2)
Input dataset r2ai-zhdocs: zh_docs.parquet (doc_id, title, text), query.parquet."""
import glob, json, os, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
W = "/kaggle/working"; G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
D = pd.read_parquet(G("zh_docs.parquet")[0]); D["doc_id"] = D.doc_id.astype(str)
Q = pd.read_parquet(G("query.parquet")[0]); QIDS = [int(x) for x in Q.id]; QT = Q["query"].tolist()
dt = (D.title.fillna("") + "\n" + D.text.fillna("").str[:1500]).tolist(); ids = D.doc_id.tolist(); log("docs", len(ids))
NCE = 100
m = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16}); m.max_seq_length = 512
order = np.argsort([len(x) for x in dt])
E = np.zeros((len(dt), 1024), np.float16)
E[order] = m.encode([dt[i] for i in order], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
QE = m.encode(QT, batch_size=64, normalize_embeddings=True).astype(np.float16); log("encoded"); del m; torch.cuda.empty_cache()
Et = torch.tensor(E, device="cuda"); S = torch.tensor(QE, device="cuda") @ Et.T
ts, ti = S.topk(300, dim=1); ts, ti = ts.float().cpu().numpy(), ti.cpu().numpy(); del Et, S; torch.cuda.empty_cache()
json.dump({str(q): [[round(float(s), 5), ids[i]] for s, i in zip(ts[r], ti[r])] for r, q in enumerate(QIDS)}, open(f"{W}/zh_dense.json", "w"))
np.save(f"{W}/zh_emb.npy", E); json.dump(ids, open(f"{W}/zh_ids.json", "w")); log("dense done")
rows = [(q, ids[i], float(s), k) for r, q in enumerate(QIDS) for k, (s, i) in enumerate(zip(ts[r][:NCE], ti[r][:NCE]))]
P = pd.DataFrame(rows, columns=["qid", "doc_id", "dense", "rank"]); pos = {d: i for i, d in enumerate(ids)}
qs = dict(zip(QIDS, QT)); pairs = [(qs[q], dt[pos[d]]) for q, d in zip(P.qid, P.doc_id)]
ce = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda", model_kwargs={"torch_dtype": torch.float16})
P["ce0"] = ce.predict(pairs, batch_size=64, show_progress_bar=False); del ce; torch.cuda.empty_cache(); log("ce0 done")
c2 = G("ce2_all/config.json")
if c2:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    p = os.path.dirname(c2[0]); tok = AutoTokenizer.from_pretrained(p); mm = AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), 64):
            b = pairs[i:i + 64]; enc = tok([x for x, _ in b], [y for _, y in b], truncation="only_second", max_length=384, padding=True, return_tensors="pt")
            out += mm(**{k: v.cuda() for k, v in enc.items()}).logits.view(-1).float().cpu().tolist()
    P["ce2"] = out; log("ce2 done")
P.to_parquet(f"{W}/zh_pairs.parquet", index=False); log("ALLDONE", len(P))
