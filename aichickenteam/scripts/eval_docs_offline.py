"""Ước lượng DOCS F2 offline trên câu public: doc đúng = doc từng có chunk trúng (mọi details_*.jsonl), số gold/câu lấy từ details.
Doc chưa từng được chấm coi là sai ⇒ cận dưới. So các thứ tự doc (RRF dump / LTR) với cùng số doc/câu.
    python scripts/eval_docs_offline.py --orders a.json b.json --ndocs data/cache/ndocs_v31.json"""
import argparse
import collections
import glob
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", nargs="+", required=True); ap.add_argument("--ndocs", required=True)
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0])
    a = ap.parse_args()
    pos, gold = collections.defaultdict(set), {}
    for f in glob.glob("data/eval/details_*.jsonl"):
        for l in open(f):
            j = json.loads(l); q = str(int(j["id"])); gold[q] = j["document"]["gold"]
            for p in j["chunk"]["predictions"]:
                if p["relevant"]:
                    pos[q].add(p["doc_id"].split(":")[1])
    nd = json.load(open(a.ndocs))
    for o in a.orders:
        od = json.load(open(o))
        for sc in a.scale:
            f2s = []
            for q, g in gold.items():
                if g == 0 or q not in od:
                    continue
                n = max(1, int(round(nd[q] * sc)))
                hit = len(set(map(str, od[q][:n])) & pos[q]); P, R = hit / n, hit / g
                f2s.append(5 * P * R / (4 * P + R) if hit else 0)
            print(f"{o} scale={sc}: docsF2~{np.mean(f2s):.4f} (n={len(f2s)})")


if __name__ == "__main__":
    main()
