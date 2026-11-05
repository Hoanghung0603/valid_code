"""Láng giềng gần doc–doc trên toàn kho bằng embedding GreenNode doc-level (data/cache/dense_gn/part*.npy), GPU.

Ghi data/cache/gn_knn.npz: ids (doc_id str), nb (N×K chỉ số vào ids), sim (N×K float16), bỏ chính nó.
Dùng cho: gom bài trùng (sim rất cao) và mở rộng ứng viên theo láng giềng của doc LTR tin nhất.
    python scripts/doc_knn.py [--k 20] [--block 1024]
"""
import argparse
import glob
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=20); ap.add_argument("--block", type=int, default=1024)
    a = ap.parse_args()
    ids, E = [], []
    for f in sorted(glob.glob("data/cache/dense_gn/part*.npy")):
        ids += json.load(open(f.replace(".npy", ".ids.json"))); E.append(np.load(f).astype(np.float16))
    E = np.concatenate(E); seen = {}
    keep = [seen.setdefault(d, i) == i for i, d in enumerate(ids)]   # id lặp giữa các part: giữ bản đầu
    ids = [d for d, k in zip(ids, keep) if k]; E = E[np.array(keep)]
    T = torch.tensor(E, device="cuda"); T = T / T.float().norm(dim=1, keepdim=True).half()
    N = len(ids); nb = np.empty((N, a.k), np.int32); sim = np.empty((N, a.k), np.float16)
    for s in range(0, N, a.block):
        S = T[s:s + a.block] @ T.T
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -1   # bỏ chính nó
        v, i = S.topk(a.k, dim=1)
        nb[s:s + len(v)] = i.cpu().numpy(); sim[s:s + len(v)] = v.cpu().numpy()
        if s % (100 * a.block) == 0:
            print(s, N, flush=True)
    np.savez("data/cache/gn_knn.npz", ids=np.array(ids), nb=nb, sim=sim)
    print("wrote gn_knn.npz", N, "sim>=0.98:", int((sim[:, 0] >= 0.98).sum()), "sim>=0.95:", int((sim[:, 0] >= 0.95).sum()))


if __name__ == "__main__":
    main()
