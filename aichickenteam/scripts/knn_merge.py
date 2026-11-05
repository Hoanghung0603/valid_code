"""Gộp điểm cặp doc mở rộng (kaggle_knnpairs → data/cache/knn_out/pairs.parquet) vào cache bản --ver (từ bản --ver - 1).

rr/rr2/rr3/rr4/gnft: nối dòng (qid, doc_id, s). d1/d2/d3: doc mở rộng nằm ngoài top-300 cũ ⇒ nối vào CUỐI list (hạng ≥ 300, theo điểm).
e5 và embedding GN doc-level (PRF) dùng lại bản cũ (thư mục e5 được symlink).
    python scripts/knn_merge.py --ver 8
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "scripts")
from newdocs_merge import RR, paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ver", type=int, required=True)
    ap.add_argument("--pairs", default="data/cache/knn_out/pairs.parquet"); a = ap.parse_args()
    OLD, NEW = paths(a.ver - 1), paths(a.ver)
    P = pd.read_parquet(a.pairs); P["doc_id"] = P.doc_id.astype(str); print("pairs", len(P))
    for col, name in RR.items():
        R = pd.read_parquet(OLD["rr"](name)); R["doc_id"] = R.doc_id.astype(str)
        add = P[["qid", "doc_id", col]].rename(columns={col: "s"}).astype({"s": R.s.dtype})
        pd.concat([R, add], ignore_index=True).drop_duplicates(["qid", "doc_id"]).to_parquet(NEW["rr"](name), index=False)
        print(name, "->", NEW["rr"](name), len(R), "+", len(add))
    for k in ("d1", "d2", "d3"):
        D = json.load(open(OLD[k]))
        for q, g in P.groupby("qid"):
            v = D.setdefault(str(q), []); have = {str(x[1]) for x in v}
            v += [[float(s), d] for s, d in sorted(zip(g[k], g.doc_id), reverse=True) if d not in have]
        json.dump(D, open(NEW[k], "w"), ensure_ascii=False); print(k, "->", NEW[k])
    if not os.path.exists(NEW["e5"]):
        os.symlink(os.path.abspath(OLD["e5"]), NEW["e5"])


if __name__ == "__main__":
    main()
