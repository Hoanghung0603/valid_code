"""Dataset Kaggle cho kaggle_ce2win/: như ce2_prep nhưng văn bản = CỬA SỔ dense tốt nhất của cặp (bản toàn-nhãn của ce_ftwin).

Cặp (train ∪ score của data/cache/kg_ce2) có cửa sổ trong dense_chunks_top_v{ver} (d3: [s, doc, chunk_text]) → lấy luôn;
còn lại ghi full text doc vào fulldocs.parquet để kernel chọn cửa sổ GN tốt nhất (như kaggle_knnpairs).
Out D/: pairs.parquet (qid, fold, doc_id, label[-1 = chỉ chấm], train[bool], score[bool], wtext|None), fulldocs.parquet, query.parquet
    python scripts/ce2win_prep.py --ver 8 --out data/cache/kg_ce2win
"""
import argparse
import glob
import json
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ver", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="data/cache/kg_ce2"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = pd.read_parquet(f"{a.src}/train.parquet"); sc = pd.read_parquet(f"{a.src}/score.parquet")
    P = sc.assign(score=True).merge(tr.assign(train=True), on=["qid", "fold", "doc_id"], how="outer")
    P["score"] = P.score.fillna(False).astype(bool); P["train"] = P.train.fillna(False).astype(bool); P["label"] = P.label.fillna(-1).astype(int)
    D3 = json.load(open(f"data/cache/dense_chunks_top_v{a.ver}.json"))
    W = {(int(q), str(x[1])): x[2] for q, v in D3.items() for x in v if len(x) > 2}
    P["wtext"] = [W.get((int(q), d)) for q, d in zip(P.qid, P.doc_id)]
    need = set(P.doc_id[P.wtext.isna()]); parts = []
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pd.read_parquet(f, columns=["doc_id", "text"]); t["doc_id"] = t.doc_id.astype(str)
        parts.append(t[t.doc_id.isin(need)])
    F = pd.concat(parts).drop_duplicates("doc_id"); F["text"] = F.text.fillna("").str[:12000]   # ≤ 6 cửa sổ × ~1600 ký tự
    P = P[P.wtext.notna() | P.doc_id.isin(set(F.doc_id))]
    P.to_parquet(f"{a.out}/pairs.parquet", index=False); F.to_parquet(f"{a.out}/fulldocs.parquet", index=False)
    shutil.copy("data/ViBioMIR/query.parquet", f"{a.out}/query.parquet")
    print("pairs", len(P), "train", int(P.train.sum()), "score", int(P.score.sum()), "have window", int(P.wtext.notna().sum()), "fulldocs", len(F))


if __name__ == "__main__":
    main()
