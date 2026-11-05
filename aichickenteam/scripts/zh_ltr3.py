"""zh_ltr v3: thêm đặc trưng NGƯỢC (câu q đứng hạng mấy trong 1200 câu với doc d; khoảng cách tới câu khớp nhất) + mở rộng ứng viên lên top-300 dense.
Dương có q hạng 1 với doc 65% vs âm 33% (21/10). Cặp ngoài top-100 không có ce0/ce2 (NaN).
    python scripts/zh_ltr3.py --feat data/cache/zhfeat2_out --out data/cache/zh_probs3.parquet
"""
import argparse, json, sys
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--feat", required=True); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=300)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--qfeat", default="", help="parquet từ scripts/zh_qfeat.py (câu hỏi dịch ZH: bm25, dzq) – mở rộng ứng viên + đặc trưng")
ap.add_argument("--extra", default="", help="parquet (qid, doc_id, s) – vd CE ZH fine-tune (kaggle_cezh) → cột cezh")
a = ap.parse_args()
Q = pd.read_parquet("data/ViBioMIR/query.parquet"); QIDS = [int(x) for x in Q.id]; qi = {q: i for i, q in enumerate(QIDS)}
QE = np.load("data/cache/zh_qemb_bgem3.npy").astype(np.float16); E = np.load(f"{a.feat}/zh_emb.npy"); ids = json.load(open(f"{a.feat}/zh_ids.json")); di = {d: j for j, d in enumerate(ids)}
S = (QE @ E.T).astype(np.float32); colmax = S.max(0); col2 = np.partition(S, -2, axis=0)[-2]
import os
QRF = f"{a.feat}/zh_qrank.npy"   # cache theo bộ đặc trưng (kích thước khác nhau giữa các lần cào)
qr = np.load(QRF) if os.path.exists(QRF) and np.load(QRF, mmap_mode="r").shape == S.shape else None
if qr is None:
    qr = np.empty(S.shape, np.int16)
    for j in range(0, S.shape[1], 20000): qr[:, j:j + 20000] = np.argsort(np.argsort(-S[:, j:j + 20000], 0), 0) + 1
    np.save(QRF, qr)
rows = []
for q in QIDS:
    r = qi[q]; top = np.argpartition(-S[r], a.top)[:a.top]
    for j in top: rows.append((q, ids[j], float(S[r, j]), int(qr[r, j]), float(colmax[j] - S[r, j]), float(colmax[j]), float(col2[j])))
P = pd.DataFrame(rows, columns=["qid", "doc_id", "dense", "qrank", "qgap", "dmax", "d2nd"])
if a.qfeat:   # ứng viên thêm từ câu hỏi dịch ZH (BM25 / dense ZH-ZH) + đặc trưng
    QF = pd.read_parquet(a.qfeat); QF["doc_id"] = QF.doc_id.astype(str)
    have = set(zip(P.qid, P.doc_id)); new_ = QF[[(q, d) not in have for q, d in zip(QF.qid, QF.doc_id)]]
    add = [(q, d, float(S[qi[q], di[d]]), int(qr[qi[q], di[d]]), float(colmax[di[d]] - S[qi[q], di[d]]), float(colmax[di[d]]), float(col2[di[d]])) for q, d in zip(new_.qid, new_.doc_id) if d in di]
    P = pd.concat([P, pd.DataFrame(add, columns=P.columns)], ignore_index=True)
    P = P.merge(QF, on=["qid", "doc_id"], how="left"); P["bm25"] = P.bm25.fillna(0); P["bm25_rank"] = P.bm25_rank.fillna(999); P["dzq_rank"] = P.dzq_rank.fillna(999)
    print("ứng viên thêm từ câu hỏi dịch:", len(add), flush=True)
C = pd.read_parquet(f"{a.feat}/zh_pairs.parquet")[["qid", "doc_id", "ce0", "ce2"]]; C["doc_id"] = C.doc_id.astype(str)
P = P.merge(C, on=["qid", "doc_id"], how="left")
if a.extra:
    X_ = pd.read_parquet(a.extra)[["qid", "doc_id", "s"]].rename(columns={"s": "cezh"}); X_["doc_id"] = X_.doc_id.astype(str); P = P.merge(X_, on=["qid", "doc_id"], how="left")
D = pd.read_parquet("data/cache/kg_zhdocs/zh_docs.parquet", columns=["doc_id", "url", "title", "text"]); D["doc_id"] = D.doc_id.astype(str)
D["host"] = D.url.str.extract(r"//([^/]+)/")[0]; D["dlen"] = D.text.fillna("").str.len(); D["tlen"] = D.title.fillna("").str.len()
P = P.merge(D[["doc_id", "host", "dlen", "tlen"]], on="doc_id", how="left"); P["qlen"] = P.qid.map(dict(zip(QIDS, Q["query"].str.len())))
P["nown"] = P.groupby("qid").qrank.transform(lambda s: (s == 1).sum())   # số doc mà q là câu khớp nhất
for c in ("dense", "ce0", "ce2") + (("cezh",) if a.extra else ()) + (("bm25", "dzq") if a.qfeat else ()):
    g = P.groupby("qid")[c]; P[f"{c}_rank"] = g.rank(ascending=False); P[f"{c}_gap"] = g.transform("max") - P[c]
pos, seen = labels()
P["lab"] = [d in seen.get(q, ()) for q, d in zip(P.qid, P.doc_id)]; P["y"] = [int(d in pos.get(q, ())) for q, d in zip(P.qid, P.doc_id)]
print("cặp", len(P), "có nhãn", int(P.lab.sum()), "dương", int(P.y[P.lab].sum()), "| dương ZH đã biết nằm trong ứng viên:", int(P.y.sum()), "/", sum(1 for q in pos for d in pos[q] if d in di))
feats = [c for c in P.columns if c not in ("qid", "doc_id", "host", "lab", "y")] + ["host_rate"]
qs = np.array(sorted(set(seen))); rng = np.random.default_rng(0); rng.shuffle(qs); folds = np.array_split(qs, 5)
prm = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2)
def host_rate(tr):
    t = tr.groupby("host").y.agg(["sum", "count"]); return ((t["sum"] + 0.09 * 30) / (t["count"] + 30)).to_dict()
def fit(tr):
    hr = host_rate(tr); X = tr.assign(host_rate=tr.host.map(hr).fillna(0.09))
    return hr, [lgb.train({**prm, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}, lgb.Dataset(X[feats], X.y), 500) for s in range(a.seeds)]
def pred(hr, ms, X):
    X = X.assign(host_rate=X.host.map(hr).fillna(0.09)); return np.mean([m.predict(X[feats]) for m in ms], 0)
P["p"] = np.nan
for k in range(5):
    te = P.qid.isin(folds[k]); hr, ms = fit(P[~te & P.lab]); P.loc[te, "p"] = pred(hr, ms, P[te])
hr, ms = fit(P[P.lab]); pr = P.p.isna(); P.loc[pr, "p"] = pred(hr, ms, P[pr])
L = P[P.lab]
for c in ("dense", "qrank") + (("cezh",) if a.extra else ()) + (("bm25", "dzq") if a.qfeat else ()) + ("p",):
    sgn = -1 if c == "qrank" else 1
    LL = L[L[c].notna()]; au = [roc_auc_score(g.y, sgn * g[c]) for _, g in LL.groupby("qid") if 0 < g.y.sum() < len(g)]; print(f"{c:6s} AUC/câu {np.mean(au):.3f} (cặp có giá trị {len(LL)})")
for k in (5, 10, 20):
    top = L.sort_values("p", ascending=False).groupby("qid").head(k); print(f"OOF precision@{k} (cặp có nhãn): {top.y.mean():.3f} | mean p {top.p.mean():.3f}")
imp = sorted(zip(ms[0].feature_importance("gain"), feats), reverse=True)[:8]; print("importance", [(f, int(v)) for v, f in imp])
P[["qid", "doc_id", "p", "dense", "ce0", "ce2", "qrank"]].to_parquet(a.out, index=False); print("wrote", a.out)
