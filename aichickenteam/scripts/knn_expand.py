"""Mở rộng ứng viên theo láng giềng: doc gần (GN doc–doc, data/cache/gn_knn.npz) của top-S doc LTR (thứ tự OOF ⇒ không rò nhãn).

Láng giềng của doc gold là gold ~36–42% (tỷ lệ chung của doc đã nộp ~10%), nhiều láng giềng nằm ngoài pool hiện tại.
Điểm của doc B = Σ_{A ∈ seeds, sim(A,B) ≥ t} sim(A,B) / S  (vừa độ gần vừa số hạt giống trỏ tới).
Ghi --out {qid: [[s, doc], ...]} top-K (cả doc trong lẫn ngoài pool) → ltr.py --dense4; --kaggle D: doc ngoài pool cần chấm CE.
    python scripts/knn_expand.py --order data/cache/ltr_order_oof23.json --out data/cache/knn_exp.json [--kaggle data/cache/kg_knn]
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--t", type=float, default=0.85)
    ap.add_argument("--k", type=int, default=400); ap.add_argument("--kaggle", default="")
    ap.add_argument("--npair", type=int, default=80, help="số doc ngoài pool/câu đưa đi chấm CE")
    a = ap.parse_args()
    Z = np.load("data/cache/gn_knn.npz"); ids = Z["ids"]; nb = Z["nb"]; sim = Z["sim"].astype(np.float32)
    ix = {d: i for i, d in enumerate(ids)}
    order = json.load(open(a.order)); out, new = {}, {}
    for q, docs in order.items():
        sc = collections.defaultdict(float)
        for A in docs[:a.seeds]:
            i = ix.get(A)
            if i is None:
                continue
            for j, s in zip(nb[i], sim[i]):
                if s >= a.t:
                    sc[ids[j]] += float(s) / a.seeds
        top = sorted(sc.items(), key=lambda x: -x[1])[:a.k]
        out[q] = [[round(s, 5), d] for d, s in top]
        pool = set(docs); new[q] = [d for d, _ in top if d not in pool][:a.npair]
    json.dump(out, open(a.out, "w"))
    n = [len(v) for v in new.values()]
    print("wrote", a.out, len(out), "list/q", np.mean([len(v) for v in out.values()]).round(1), "outside-pool/q", np.mean(n).round(1))
    if a.kaggle:
        os.makedirs(a.kaggle, exist_ok=True)
        need = {d for v in new.values() for d in v}; parts = []
        for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
            t = pd.read_parquet(f, columns=["doc_id", "title", "text"]); t["doc_id"] = t.doc_id.astype(str)
            parts.append(t[t.doc_id.isin(need)])
        D = pd.concat(parts).drop_duplicates("doc_id"); D.to_parquet(f"{a.kaggle}/exp_docs.parquet", index=False)
        pd.DataFrame([(int(q), d) for q, v in new.items() for d in v], columns=["qid", "doc_id"]).to_parquet(f"{a.kaggle}/exp_pairs.parquet", index=False)
        pd.read_parquet("data/ViBioMIR/query.parquet").to_parquet(f"{a.kaggle}/query.parquet", index=False)
        print("kaggle dataset", a.kaggle, "docs", len(D), "pairs", sum(n))


if __name__ == "__main__":
    main()
