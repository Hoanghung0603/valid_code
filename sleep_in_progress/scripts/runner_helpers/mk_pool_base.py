"""Tao pool base tu pool p1 (rr_merge): giu doc thuoc pool_for_mode('base') voi logit bge da co.

Chay tren b660: python mk_pool_base.py <dir_p1> <dir_base>
Doc base phai la tap con cua pool p1 (kiem bang assert), thu tu theo logit giam dan."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\r2ai\runner\scripts")
import runner_core as rc  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
bm300 = {q: [[h[0], h[1]] for h in v[:300]] for q, v in rc.load_json(src / "cache_k300.json").items()}
dn = rc.load_json(src / "dense_top200.json")
hy = rc.load_json(src / "hyde_top200.json")
p1 = rc.load_json(src / "pool.json")
out, miss = {}, 0
for q in sorted(p1, key=lambda x: int(x)):
    base = rc.pool_for_mode("base", bm300.get(q, []), dn.get(q, []), hy.get(q, []))
    have = {str(h[0]): h for h in p1[q]}
    rows = []
    for h in base:
        r = have.get(str(h[0]))
        if r is None:
            miss += 1
            continue
        rows.append(r)
    out[q] = rc.sort_by_score(rows)
assert miss == 0, f"doc base thieu logit trong p1: {miss}"
rc.write_json(dst / "pool.json", out)
n = max(len(out), 1)
print("pool base:", len(out), "cau; TB", round(sum(map(len, out.values())) / n, 2), "doc/cau", flush=True)
