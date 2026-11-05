import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.index import tokenize

IDX = ROOT / "data/cache/bm25_zh"
BITS = 23
K1, B = 1.2, 0.75
MAX_CHARS = 6000

def thash(toks):
    import zlib
    m = (1 << BITS) - 1
    return np.fromiter((zlib.crc32(t.encode()) & m for t in toks), dtype=np.int64, count=len(toks))

def zh_files():
    files = sorted(glob.glob(str(ROOT / "data/docs_zh/corpus-*.parquet")))
    
    # Check Google Drive common paths if the user downloaded it there
    drive_dirs = [
        Path.home() / "Google Drive/MyDrive/r2ai/docs_zh",
        Path.home() / "Drive/MyDrive/r2ai/docs_zh",
        Path.home() / "gdrive/MyDrive/r2ai/docs_zh",
        Path("/content/drive/MyDrive/r2ai/docs_zh"),
        Path.home() / "Downloads/r2ai/docs_zh",
        Path.home() / "Downloads/docs_zh",
    ]
    for d in drive_dirs:
        if d.exists():
            files += sorted(glob.glob(str(d / "corpus-*.parquet")))
            
    return sorted(list(set(files)))

def build(shard_docs=100_000):
    IDX.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seen, buf, shard = set(), [], 0
    df_total = np.zeros(1 << BITS, np.int64)
    meta = []

    def flush():
        nonlocal buf, shard
        if not buf: return
        indptr, ind, dat, ids, dl = [0], [], [], [], []
        for did, h, c, n_tok in buf:
            ind.append(h); dat.append(c)
            indptr.append(indptr[-1] + len(h)); ids.append(did); dl.append(n_tok)
        ind = np.concatenate(ind); dat = np.concatenate(dat).astype(np.float32)
        tf = sp.csr_matrix((dat, ind, np.asarray(indptr, np.int64)), shape=(len(ids), 1 << BITS))
        df_total[:] += np.bincount(ind, minlength=1 << BITS)
        sp.save_npz(IDX / f"tf_{shard}.npz", tf, compressed=False)
        np.save(IDX / f"dl_{shard}.npy", np.asarray(dl, np.float32))
        meta.append({"shard": shard, "doc_ids": ids})
        print(f"shard {shard}: {len(ids)} docs, nnz {tf.nnz}, {time.time() - t0:.0f}s", flush=True)
        shard += 1
        buf = []

    files = zh_files()
    if not files:
        print("No Chinese parquet files found.")
        return

    for f in files:
        t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        t = t[t.lang == "zh"]
        for did, ti, tx in zip(t.doc_id.astype(str), t.title.fillna(""), t.text):
            if did in seen: continue
            seen.add(did)
            toks = tokenize(f"{ti}\n{tx[:MAX_CHARS]}", "zh")
            h, c = np.unique(thash(toks), return_counts=True)
            buf.append((did, h.astype(np.int32), c.astype(np.float16), len(toks)))
            if len(buf) >= shard_docs:
                flush()
    flush()
    if not meta:
        print("No documents were indexed.")
        return
    n = sum(len(m["doc_ids"]) for m in meta)
    np.save(IDX / "df.npy", df_total)
    json.dump({"n_docs": n, "shards": meta, "bits": BITS}, open(IDX / "meta.json", "w"))
    print(f"done: {n} docs in {time.time() - t0:.0f}s")


class Index:
    def __init__(self):
        meta_path = IDX / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Index not found at {meta_path}. Run build first.")
        meta = json.load(open(meta_path))
        n = meta["n_docs"]
        df = np.load(IDX / "df.npy")
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dls = [np.load(IDX / f"dl_{m['shard']}.npy") for m in meta["shards"]]
        avgdl = float(np.concatenate(dls).mean()) if dls else 1.0
        self.shards = []
        for m, dl in zip(meta["shards"], dls):
            tf = sp.load_npz(IDX / f"tf_{m['shard']}.npz").tocsr()
            norm = K1 * (1 - B + B * dl / avgdl)
            rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
            tf.data = (tf.data * (K1 + 1) / (tf.data + norm[rows])) * idf[tf.indices]
            self.shards.append((tf.tocsc(), m["doc_ids"]))
            del rows
        print(f"loaded {n} docs", flush=True)

    def search(self, toks_weighted, k=20):
        if not toks_weighted: return []
        cols = np.fromiter(toks_weighted.keys(), np.int64)
        w = np.fromiter(toks_weighted.values(), np.float32)
        out = []
        for w_csc, ids in self.shards:
            s = w_csc[:, cols] @ w
            kk = min(k, len(s))
            if kk == 0: continue
            top = np.argpartition(-s, kk - 1)[:kk]
            out += [(float(s[i]), ids[i]) for i in top if s[i] > 0]
        return sorted(out, reverse=True)[:k]

def qvec(texts_weights):
    d = {}
    for text, wt in texts_weights:
        toks = tokenize(text, "zh")
        for h in np.unique(thash(toks)):
            d[int(h)] = max(d.get(int(h), 0.0), wt)
    return d

_ix = None

def search_zh(plan, k=20):
    global _ix
    if _ix is None:
        _ix = Index()
    tw = []
    if "zh" in plan:
        for v in plan["zh"]:
            tw.append((v, 1.0))
    if "keywords" in plan and "zh" in plan["keywords"]:
        tw.append((" ".join(plan["keywords"]["zh"]), 1.0))
    return _ix.search(qvec(tw), k=k)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
