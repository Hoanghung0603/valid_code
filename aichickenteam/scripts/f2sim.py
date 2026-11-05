"""F2 docs mô phỏng trên câu public (nhãn hiện có; doc chưa thấy = sai) cho thứ tự LTR + cắt ngưỡng ef2 như fuse_dense.
    python scripts/f2sim.py data/cache/ltrx3_out/ltr_order_oof30 [khác...]   → in "<base> <F2>"
"""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402


def f2sim(base, pos, gold, c=1.9, lo=30, hi=280):
    od = json.load(open(base + ".json")); pr = json.load(open(base + "_probs.json")); fs = []
    for q, g in gold.items():
        if g == 0:
            fs.append(0.0); continue
        p = np.array(pr[str(q)]); n = np.arange(1, len(p) + 1)
        nd = int(np.clip(np.argmax(5 * np.cumsum(p) / (4 * c * p.sum() + n)) + 1, lo, hi))
        rel = sum(d in pos[q] for d in od[str(q)][:nd]); P, R = rel / nd, rel / g
        fs.append(5 * P * R / (4 * P + R) if rel else 0.0)
    return float(np.mean(fs))


if __name__ == "__main__":
    pos, _ = labels()
    gold = {int(json.loads(l)["id"]): json.loads(l)["document"]["gold"] for l in open("data/eval/details_5601_v57_ce2.jsonl")}
    for b in sys.argv[1:]:
        print(b, round(f2sim(b, pos, gold), 5))
