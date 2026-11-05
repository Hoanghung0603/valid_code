"""BM25 tiếng Trung nhanh trên Colab: 1 lượt tách từ (jieba) -> CSR băm, nhân ma trận với 1200 truy vấn.

    python zh_search_fast.py --docs /content/docs_zh --plans query_plans.jsonl --out /content/zh_preds.jsonl --k 100
Dùng lại tokenize / select_chunks / get_query_terms của zh_search.py (cùng thư mục).
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
import zlib

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zh_search import get_query_terms, select_chunks, tokenize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
BITS = 22
V = 1 << BITS
K1, B = 1.2, 0.75
MAX_CHARS = 4000


def h(tok):
    return zlib.crc32(tok.encode()) & (V - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--chunk-chars", type=int, default=750)
    ap.add_argument("--max-files", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    plans = [json.loads(l) for l in open(a.plans, encoding="utf-8") if l.strip()]
    qw = []
    for p in plans:
        w = {}
        for t, x in get_query_terms("", p).items():
            j = h(t)
            w[j] = max(w.get(j, 0.0), x)
        qw.append(w)
    rows, cols, vals = [], [], []
    for i, w in enumerate(qw):
        for j, x in w.items():
            rows.append(i); cols.append(j); vals.append(x)
    Q = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(plans), V))
    logging.info("queries %d, nnz %d", len(plans), Q.nnz)

    # Lượt 1 (duy nhất): tách từ, lưu CSR tf theo từng file
    files = sorted(glob.glob(os.path.join(a.docs, "corpus-*.parquet")))
    if a.max_files:
        files = files[:a.max_files]
    seen, mats, ids_all, dls, loc = set(), [], [], [], {}
    df = np.zeros(V, np.int64)
    for fi, f in enumerate(files):
        try:
            t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        except Exception as e:  # shard hỏng (chép dở từ Drive) -> bỏ qua
            logging.warning("skip %s: %s", f, str(e)[:120])
            continue
        indptr, ind, dat, ids = [0], [], [], []
        for did, lang, ti, tx in zip(t.doc_id.astype(str), t.lang, t.title.fillna(""), t.text.fillna("")):
            if lang != "zh" or did in seen or not tx:
                continue
            seen.add(did)
            toks = tokenize(ti) * 2 + tokenize(tx[:MAX_CHARS])   # title boost x2
            if not toks:
                continue
            hh, c = np.unique(np.fromiter((h(x) for x in toks), np.int64, len(toks)), return_counts=True)
            ind.append(hh.astype(np.int32)); dat.append(c.astype(np.float32))
            indptr.append(indptr[-1] + len(hh)); ids.append(did); dls.append(len(toks))
            loc[did] = fi
        if not ids:
            continue
        ind = np.concatenate(ind); dat = np.concatenate(dat)
        df += np.bincount(ind, minlength=V)
        mats.append(sp.csr_matrix((dat, ind, np.asarray(indptr, np.int64)), shape=(len(ids), V)))
        ids_all.append(ids)
        if fi % 10 == 0:
            logging.info("tokenized %d/%d files, %d docs, %.0fs", fi + 1, len(files), len(seen), time.time() - t0)
    n = sum(m.shape[0] for m in mats)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    avgdl = float(np.mean(dls))
    logging.info("docs %d, avgdl %.0f, tokenize done %.0fs", n, avgdl, time.time() - t0)

    # Chấm: W (docs×V, BM25) @ Q^T (V×1200) theo từng file, giữ top-k/truy vấn
    QT = Q.T.tocsr()
    best_s = np.full((len(plans), 0), -1.0, np.float32)
    best_d = np.empty((len(plans), 0), dtype=object)
    off = 0
    for m, ids in zip(mats, ids_all):
        dl = np.asarray(dls[off:off + m.shape[0]], np.float32); off += m.shape[0]
        norm = K1 * (1 - B + B * dl / avgdl)
        r = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
        m.data = m.data * (K1 + 1) / (m.data + norm[r]) * idf[m.indices]
        S = (m @ QT).toarray().T                       # 1200 × docs
        kk = min(a.k, S.shape[1])
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        ts = np.take_along_axis(S, top, 1)
        td = np.array(ids, dtype=object)[top]
        best_s = np.concatenate([best_s, ts], 1); best_d = np.concatenate([best_d, td], 1)
        if best_s.shape[1] > a.k:
            o = np.argsort(-best_s, axis=1)[:, :a.k]
            best_s = np.take_along_axis(best_s, o, 1); best_d = np.take_along_axis(best_d, o, 1)
    del mats
    logging.info("scored %.0fs", time.time() - t0)

    # Lấy text các doc cần để chọn chunk
    need = {d for row in best_d for d in row}
    texts = {}
    for fi in sorted({loc[d] for d in need}):
        t = pq.read_table(files[fi], columns=["doc_id", "text"]).to_pandas()
        t = t[t.doc_id.astype(str).isin(need)]
        texts.update(zip(t.doc_id.astype(str), t.text))
    with open(a.out, "w", encoding="utf-8") as fo:
        for i, p in enumerate(plans):
            o = np.argsort(-best_s[i])
            docs, chunks = [], []
            for j in o:
                s, d = float(best_s[i, j]), best_d[i, j]
                if s <= 0:
                    continue
                docs.append([s, d])
                sel = select_chunks("", p, texts.get(d, ""), target_chars=a.chunk_chars)
                if sel and sel[0] in texts.get(d, ""):
                    chunks.append({"doc_id": d, "chunk_text": sel[0], "score": s})
            fo.write(json.dumps({"id": p["id"], "docs": docs, "chunks": chunks}, ensure_ascii=False) + "\n")
    logging.info("wrote %s in %.0fs", a.out, time.time() - t0)


if __name__ == "__main__":
    main()
