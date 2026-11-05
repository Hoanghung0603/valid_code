"""Tao idx map vi36 (doc id -> hang trong ids.npy) va doi chieu voi emb da co.

Chay tren b660: python mk_idx_vi36.py  (ghi idx_vi36.json)
Chay tren z87 : python mk_idx_vi36.py check  (kiem tong, trung, thieu voi blk_*.npz)
"""
import json
import sys
from pathlib import Path

import numpy as np

IDS = Path(r"D:\r2ai\data\index\vi36\ids.npy")
OUT = Path(r"D:\r2ai\work\runner\full1200\idx_vi36.json")
EMB_DIRS = [Path(r"D:\r2ai\work\runner\rsmoke20\emb_clean"), Path(r"D:\r2ai\work\vi36\emb")]


def build_map():
    ids = np.load(IDS)
    if len(np.unique(ids)) != len(ids):
        raise SystemExit("ids.npy co id trung")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m = {str(int(d)): i for i, d in enumerate(ids)}
    OUT.write_text(json.dumps(m), encoding="utf-8")
    print("idx map", len(m), flush=True)


def check():
    m = {int(k): v for k, v in json.loads(OUT.read_text(encoding="utf-8")).items()}
    seen = []
    for d in EMB_DIRS:
        for f in sorted(d.glob("blk_*.npz")):
            seen.append(np.load(f)["ids"])
    allids = np.concatenate(seen).astype(np.int64)
    uniq = np.unique(allids)
    print("emb vectors", len(allids), "unique", len(uniq), "trung", len(allids) - len(uniq), flush=True)
    print("map", len(m), "emb-khong-co-map", int(sum(1 for d in uniq if int(d) not in m)), flush=True)
    print("map-khong-co-emb", len(m) - int(len(set(m) & set(int(x) for x in uniq))), flush=True)
    bad_order = sum(1 for d in uniq[:1000] if m.get(int(d)) is None)
    print("thieu map trong 1000 dau", bad_order, flush=True)


if __name__ == "__main__":
    check() if len(sys.argv) > 1 and sys.argv[1] == "check" else build_map()
