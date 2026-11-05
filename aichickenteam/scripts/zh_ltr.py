"""LTR riêng cho doc ZH (bộ 235k): cặp top-100/câu theo bge-m3 dense (kernel kaggle_zhfeat) + nhãn (thăm dò v69 + v3a).
Đặc trưng: dense, ce0 (bge-reranker-v2-m3), ce2 (ce2_all), hạng/khoảng cách trong câu, tiên nghiệm host (OOF), độ dài doc/tiêu đề, độ dài câu hỏi.
OOF 5 fold theo câu (giống ltr.py) ⇒ xác suất trung thực cho câu public; câu private dùng model toàn bộ.
    python scripts/zh_ltr.py --out data/cache/zh_probs.parquet
"""
import argparse, json, sys
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
P = pd.read_parquet("data/cache/zhfeat_out/zh_pairs.parquet"); P["doc_id"] = P.doc_id.astype(str)
D = pd.read_parquet("data/cache/kg_zhdocs/zh_docs.parquet", columns=["doc_id", "url", "title", "text"]); D["doc_id"] = D.doc_id.astype(str)
D["host"] = D.url.str.extract(r"//([^/]+)/")[0]; D["dlen"] = D.text.fillna("").str.len(); D["tlen"] = D.title.fillna("").str.len()
P = P.merge(D[["doc_id", "host", "dlen", "tlen"]], on="doc_id", how="left")
Q = pd.read_parquet("data/ViBioMIR/query.parquet"); P["qlen"] = P.qid.map(dict(zip(Q.id.astype(int), Q["query"].str.len())))
for c in ("dense", "ce0", "ce2"):
    g = P.groupby("qid")[c]; P[f"{c}_rank"] = g.rank(ascending=False); P[f"{c}_gap"] = g.transform("max") - P[c]; P[f"{c}_z"] = (P[c] - g.transform("mean")) / (g.transform("std") + 1e-6)
pos, seen = labels()
P["lab"] = [d in seen.get(q, ()) for q, d in zip(P.qid, P.doc_id)]; P["y"] = [int(d in pos.get(q, ())) for q, d in zip(P.qid, P.doc_id)]
print("cặp", len(P), "có nhãn", int(P.lab.sum()), "dương", int(P.y[P.lab].sum()))
feats = [c for c in P.columns if c not in ("qid", "doc_id", "host", "lab", "y")] + ["host_rate"]
qs = np.array(sorted(set(seen))); rng = np.random.default_rng(0); rng.shuffle(qs); folds = np.array_split(qs, 5)
prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2)


def host_rate(tr):
    t = tr.groupby("host").y.agg(["sum", "count"]); return ((t["sum"] + 0.09 * 30) / (t["count"] + 30)).to_dict()


def fit(tr):
    hr = host_rate(tr); X = tr.assign(host_rate=tr.host.map(hr).fillna(0.09))
    ms = [lgb.train({**prm, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}, lgb.Dataset(X[feats], X.y), 400) for s in range(a.seeds)]
    return hr, ms


def pred(hr, ms, X):
    X = X.assign(host_rate=X.host.map(hr).fillna(0.09)); return np.mean([m.predict(X[feats]) for m in ms], 0)


P["p"] = np.nan
for k in range(5):
    te = P.qid.isin(folds[k]); hr, ms = fit(P[~te & P.lab]); P.loc[te, "p"] = pred(hr, ms, P[te])
hr, ms = fit(P[P.lab]); pr = P.p.isna(); P.loc[pr, "p"] = pred(hr, ms, P[pr])
L = P[P.lab]
for c in ("dense", "ce0", "ce2", "p"):
    au = [roc_auc_score(g.y, g[c]) for _, g in L.groupby("qid") if 0 < g.y.sum() < len(g)]; print(f"{c:6s} AUC/câu {np.mean(au):.3f}")
for k in (5, 10, 20):
    top = L.sort_values("p", ascending=False).groupby("qid").head(k); print(f"OOF precision@{k} (cặp có nhãn): {top.y.mean():.3f}  | mean p {top.p.mean():.3f}")
imp = sorted(zip(ms[0].feature_importance("gain"), feats), reverse=True)[:8]; print("importance", [(f, int(v)) for v, f in imp])
P[["qid", "doc_id", "p", "dense", "ce0", "ce2"]].to_parquet(a.out, index=False); print("wrote", a.out)
