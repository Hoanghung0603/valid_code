"""Dataset Kaggle cho kaggle_ce2/ (fine-tune lại cross-encoder trên TOÀN BỘ nhãn hiện có, ~5× lúc train ce_ft).

Fold câu hỏi giống hệt ltr.py (qid có nhãn sắp xếp, rng(0).shuffle, chia 5) ⇒ điểm câu public là OOF đúng nghĩa.
Train: mọi cặp dương + tối đa --neg × số dương cặp âm mỗi câu (một nửa âm khó: hạng cao trong --order, một nửa ngẫu nhiên).
Score: mọi cặp trong pool LTR (rr_bge_all_v{ver}); fold = -1 cho câu private.
Out D/: docs.parquet (doc_id, text = title\\n text[:1500]), train.parquet (qid, fold, doc_id, label), score.parquet (qid, fold, doc_id), query.parquet
    python scripts/ce2_prep.py --ver 8 --order data/cache/ltr_order_oof24.json --out data/cache/kg_ce2
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402
from newdocs_merge import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ver", type=int, required=True); ap.add_argument("--order", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--neg", type=float, default=2.0); a = ap.parse_args()
    import json
    os.makedirs(a.out, exist_ok=True)
    pos, seen = labels(); order = json.load(open(a.order))
    qs = np.array(sorted(seen)); rng = np.random.default_rng(0); rng.shuffle(qs)
    fold = {int(q): k for k, part in enumerate(np.array_split(qs, 5)) for q in part}
    rs = np.random.default_rng(1); tr = []
    for q in sorted(seen):
        P = sorted(pos[q]); N = [d for d in seen[q] if d not in pos[q]]
        rk = {d: i for i, d in enumerate(order.get(str(q), []))}
        N.sort(key=lambda d: rk.get(d, 1e9)); n = min(len(N), int(a.neg * max(len(P), 5)))
        hard = N[:n // 2]; rest = N[n // 2:]
        neg = hard + (list(rs.choice(rest, min(len(rest), n - len(hard)), replace=False)) if rest else [])
        tr += [(q, fold[q], d, 1) for d in P] + [(q, fold[q], d, 0) for d in neg]
    tr = pd.DataFrame(tr, columns=["qid", "fold", "doc_id", "label"])
    sc = pd.read_parquet(paths(a.ver)["rr"]("rr_bge_all"), columns=["qid", "doc_id"]); sc["doc_id"] = sc.doc_id.astype(str)
    sc = pd.concat([sc, tr[["qid", "doc_id"]]]).drop_duplicates()
    sc["fold"] = sc.qid.map(lambda q: fold.get(int(q), -1))
    need = set(sc.doc_id) | set(tr.doc_id); parts = []
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pd.read_parquet(f, columns=["doc_id", "title", "text"]); t["doc_id"] = t.doc_id.astype(str)
        parts.append(t[t.doc_id.isin(need)])
    D = pd.concat(parts).drop_duplicates("doc_id")
    D = pd.DataFrame({"doc_id": D.doc_id, "text": D.title.fillna("") + "\n" + D.text.fillna("").str[:1500]})
    miss = need - set(D.doc_id); tr = tr[~tr.doc_id.isin(miss)]; sc = sc[~sc.doc_id.isin(miss)]
    D.to_parquet(f"{a.out}/docs.parquet", index=False); tr.to_parquet(f"{a.out}/train.parquet", index=False)
    sc.to_parquet(f"{a.out}/score.parquet", index=False); shutil.copy("data/ViBioMIR/query.parquet", f"{a.out}/query.parquet")
    print("docs", len(D), "missing", len(miss), "train", len(tr), "pos", int(tr.label.sum()), "per fold", tr.fold.value_counts().sort_index().tolist(),
          "score", len(sc), "private", int((sc.fold == -1).sum()))


if __name__ == "__main__":
    main()
