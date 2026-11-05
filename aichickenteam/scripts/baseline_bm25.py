"""Baseline v0: BM25 tiếng Việt cấp doc trên các trang đã crawl, chunk = đoạn tốt nhất theo BM25.

Không dùng LLM/PubMed. Mục đích: xác nhận định dạng nộp + có điểm tham chiếu.
    python scripts/baseline_bm25.py --out submissions/baseline_v0.zip
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import scipy.sparse as sp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunk import chunk_doc  # noqa: E402
from src.index import tokenize  # noqa: E402
from src.submit import complete, validate, write_zip  # noqa: E402

CHUNK_CFG = {"mode": "paragraph", "target": {"vi": 600}, "max": {"vi": 1000}, "min": {"vi": 80}}


class SparseBM25:
    """BM25 (Lucene idf) trên ma trận thưa doc×term; RAM ~ 12 byte / (doc, term) khác nhau."""

    def __init__(self, token_lists, k1=1.2, b=0.75):
        self.vocab = {}
        indptr, indices, data = [0], [], []
        for toks in token_lists:
            ids, cnt = np.unique(np.fromiter((self.vocab.setdefault(t, len(self.vocab)) for t in toks),
                                             dtype=np.int64, count=len(toks)), return_counts=True)
            indices.append(ids.astype(np.int32))
            data.append(cnt.astype(np.float32))
            indptr.append(indptr[-1] + len(ids))
        n, V = len(indptr) - 1, len(self.vocab)
        indices = np.concatenate(indices) if indices else np.zeros(0, np.int32)
        data = np.concatenate(data) if data else np.zeros(0, np.float32)
        tf = sp.csr_matrix((data, indices, np.asarray(indptr, np.int64)), shape=(n, V))
        dl = np.asarray(tf.sum(1)).ravel()
        df = np.bincount(indices, minlength=V)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = (tf.data * (k1 + 1) / (tf.data + norm[rows])) * idf[tf.indices]
        self.w = tf.tocsc()  # cột = term → cộng nhanh theo term của query

    def query(self, toks, k):
        q = sorted({self.vocab[t] for t in toks if t in self.vocab})
        if not q:
            return np.array([], int), np.array([])
        s = np.asarray(self.w[:, q].sum(1)).ravel()
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return top, s[top]


def bm25_build(token_lists):
    return SparseBM25(token_lists), None


def bm25_query(m, _vocab, toks, k):
    return m.query(toks, k)


def cutoff(scores, ratio, max_k):
    if len(scores) == 0 or scores[0] <= 0:
        return 0
    n = 1
    while n < min(max_k, len(scores)) and scores[n] >= ratio * scores[0]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submissions/baseline_v0.zip")
    ap.add_argument("--doc-ratio", type=float, default=0.8)
    ap.add_argument("--doc-max", type=int, default=3)
    ap.add_argument("--chunk-ratio", type=float, default=0.8)
    ap.add_argument("--chunk-max", type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob("data/docs/corpus-*.parquet"))
    docs = pd.concat([pd.read_parquet(f, columns=["doc_id", "lang", "source", "title", "text"]) for f in files])
    docs = docs[docs.lang == "vi"].drop_duplicates("doc_id", keep="last").reset_index(drop=True)
    print(f"{len(docs)} vi docs from {len(files)} shards", flush=True)

    m, vocab = bm25_build(tokenize(f"{ti}\n{tx[:6000]}", "vi") for ti, tx in zip(docs.title.fillna(""), docs.text))
    print(f"indexed in {time.time() - t0:.0f}s, vocab {len(m.vocab)}", flush=True)

    q = pd.read_parquet("data/ViBioMIR/query.parquet")
    pred = []
    for qid, query in zip(q.id, q["query"]):
        qt = tokenize(query, "vi")
        idx, sc = bm25_query(m, vocab, qt, 20)
        n = cutoff(sc, a.doc_ratio, a.doc_max)
        sel = [docs.iloc[i] for i in idx[:n]]
        # chunk: BM25 trên các đoạn của các doc đã chọn
        chunks = [c for d in sel for c in chunk_doc(d.to_dict(), CHUNK_CFG)]
        out_chunks = []
        if chunks:
            cm, cv = bm25_build([tokenize(c["text"], "vi") for c in chunks])
            cidx, csc = bm25_query(cm, cv, qt, len(chunks))
            for i in cidx[:cutoff(csc, a.chunk_ratio, a.chunk_max * max(n, 1))]:
                out_chunks.append({"doc_id": chunks[i]["doc_id"], "chunk_text": chunks[i]["text"]})
        pred.append({"id": int(qid), "relevant_docs": [str(d.doc_id) for d in sel], "relevant_chunks": out_chunks})

    pred = complete(pred, [int(i) for i in q.id])
    errs = validate(pred, [int(i) for i in q.id], doc_lookup=dict(zip(docs.doc_id.astype(str), docs.text)))
    if errs:
        print("VALIDATION ERRORS:", errs[:10])
        sys.exit(1)
    write_zip(pred, a.out)
    nd = np.mean([len(p["relevant_docs"]) for p in pred])
    nc = np.mean([len(p["relevant_chunks"]) for p in pred])
    print(f"wrote {a.out}: {len(pred)} queries, avg {nd:.2f} docs, {nc:.2f} chunks, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
