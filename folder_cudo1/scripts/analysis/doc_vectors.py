"""Doc-level dense vectors for every Vietnamese doc: L2-normalised mean of its chunk vectors (BGE-M3, from the memmap).

  python scripts/analysis/doc_vectors.py   # -> data/indexes/dense/<run>/doc_mean_vi.f16.npy + doc_vi_ids.npy
Written block by block (BLOCK docs at a time) straight into the float16 memmap that retrieval_bench.py reads: one
fp32 accumulator for all 735k docs plus its fp16 copy peaked at ~4.5 GB, above the laptop's RAM guard (2026-10-20).
A topic-level representation of the whole document, next to the head (first chunk) and max-chunk signals; no GPU.
Each distinct input vector of a doc is counted once (repeated boilerplate chunks share an input).
"""

from __future__ import annotations

import time

import numpy as np

from medret.config import get_config
from medret.index.meta import DB_FILE, MetaStore

BLOCK = 100_000


def main() -> None:
    cfg = get_config()
    t0 = time.time()
    meta = MetaStore(cfg.paths.index_dir / DB_FILE)
    rows = meta.con.execute("""SELECT DISTINCT c.doc_id, c.input_id FROM chunks c JOIN documents d USING (doc_id)
                               WHERE d.lang = 'vi' AND c.input_id IS NOT NULL ORDER BY c.input_id""").df()
    docs, doc_idx = np.unique(rows["doc_id"].to_numpy(), return_inverse=True)
    dense = cfg.paths.index_dir / "dense" / cfg.index.dense_run
    vec = np.load(dense / "vectors.f16.npy", mmap_mode="r")
    ids = rows["input_id"].to_numpy()
    by_doc = np.argsort(doc_idx, kind="stable")
    bounds = np.searchsorted(doc_idx[by_doc], np.arange(0, len(docs) + BLOCK, BLOCK))
    out = np.lib.format.open_memmap(dense / "doc_mean_vi.f16.npy", mode="w+", dtype=np.float16,
                                    shape=(len(docs), vec.shape[1]))
    for b in range(len(bounds) - 1):
        lo, sel = b * BLOCK, by_doc[bounds[b]: bounds[b + 1]]
        if lo >= len(docs):
            break
        sel = sel[np.argsort(ids[sel])]  # sorted reads from the memmap
        acc = np.zeros((min(BLOCK, len(docs) - lo), vec.shape[1]), dtype=np.float32)
        np.add.at(acc, doc_idx[sel] - lo, vec[ids[sel]].astype(np.float32))
        acc /= np.linalg.norm(acc, axis=1, keepdims=True).clip(min=1e-6)
        out[lo: lo + len(acc)] = acc
        print(f"{lo + len(acc):,}/{len(docs):,} docs, {time.time() - t0:.0f}s", flush=True)
    out.flush()
    np.save(dense / "doc_vi_ids.npy", docs)
    print(f"{len(docs):,} doc vectors -> {dense / 'doc_mean_vi.f16.npy'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
