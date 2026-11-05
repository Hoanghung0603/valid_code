"""Dataset Kaggle cho kaggle_cezh/: CE (câu VI ↔ doc ZH). Train = mọi cặp ZH có nhãn (dương + tối đa --neg×dương âm/câu), fold = fold câu của ltr.py;
Score = top --score doc ZH/câu theo zh_probs3 (cả 1200 câu; private fold -1). Doc text = title\\n text[:1500].
    python scripts/cezh_prep.py --probs data/cache/zh_probs3.parquet --out data/cache/kg_cezh
"""
import argparse, os, shutil, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--probs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--neg", type=float, default=4.0); ap.add_argument("--score", type=int, default=150); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pos, seen = labels()
D = pd.read_parquet("data/cache/kg_zhdocs/zh_docs.parquet", columns=["doc_id", "title", "text"]); D["doc_id"] = D.doc_id.astype(str); zh = set(D.doc_id)
qs = np.array(sorted(set(seen))); rng = np.random.default_rng(0); rng.shuffle(qs); fold = {int(q): k for k, f in enumerate(np.array_split(qs, 5)) for q in f}
tr = []
for q in seen:
    P_ = [d for d in seen[q] if d in zh and d in pos[q]]; N_ = [d for d in seen[q] if d in zh and d not in pos[q]]
    if not P_ and not N_: continue
    rng.shuffle(N_); N_ = N_[:max(int(a.neg * len(P_)), 5)]
    tr += [(int(q), fold[int(q)], d, 1) for d in P_] + [(int(q), fold[int(q)], d, 0) for d in N_]
tr = pd.DataFrame(tr, columns=["qid", "fold", "doc_id", "label"])
P = pd.read_parquet(a.probs); P["doc_id"] = P.doc_id.astype(str)
sc = P.sort_values("p", ascending=False).groupby("qid").head(a.score)[["qid", "doc_id"]].copy(); sc["fold"] = sc.qid.map(lambda q: fold.get(int(q), -1))
need = set(tr.doc_id) | set(sc.doc_id); D = D[D.doc_id.isin(need)].copy(); D["text"] = D.title.fillna("") + "\n" + D.text.fillna("").str[:1500]
D[["doc_id", "text"]].to_parquet(f"{a.out}/docs.parquet", index=False); tr.to_parquet(f"{a.out}/train.parquet", index=False); sc.to_parquet(f"{a.out}/score.parquet", index=False)
shutil.copy("data/ViBioMIR/query.parquet", f"{a.out}/query.parquet")
print("train", len(tr), "pos", int(tr.label.sum()), "score", len(sc), "private", int((sc.fold == -1).sum()), "docs", len(D))
