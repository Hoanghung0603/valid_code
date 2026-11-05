"""v1: BM25 tiếng Việt trên toàn bộ doc VI đã crawl (shard + hashed vocab để vừa RAM).

    python scripts/pipeline_v1.py build                 # build index -> data/cache/bm25_vi/
    python scripts/pipeline_v1.py predict --out submissions/v1a.zip [--doc-max 1 ...]
"""
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
from src.index import tokenize  # noqa: E402
from src.submit import complete, validate, write_zip  # noqa: E402

import os
IDX = ROOT / os.environ.get("R2AI_BM25_VI_DIR", "data/cache/bm25_vi")
VI_TOK = {"vi_bigrams": os.environ.get("R2AI_VI_BIGRAMS", "1") == "1"}
BITS = 23
K1, B = 1.2, 0.75
MAX_CHARS = int(os.environ.get("R2AI_MAX_CHARS", "6000"))


def thash(toks):
    # hash ổn định (không phụ thuộc PYTHONHASHSEED)
    import zlib
    m = (1 << BITS) - 1
    return np.fromiter((zlib.crc32(t.encode()) & m for t in toks), dtype=np.int64, count=len(toks))


def vi_files():
    return sorted(glob.glob(str(ROOT / "data/docs/corpus-*.parquet")))


def build(shard_docs=100_000):
    IDX.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seen, buf, shard = set(), [], 0
    df_total = np.zeros(1 << BITS, np.int64)
    meta = []

    def flush():
        nonlocal buf, shard
        if not buf:
            return
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

    for f in vi_files():
        t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        t = t[t.lang == "vi"]
        for did, ti, tx in zip(t.doc_id.astype(str), t.title.fillna(""), t.text):
            if did in seen:
                continue
            seen.add(did)
            toks = tokenize(f"{ti}\n{tx[:MAX_CHARS]}", "vi", VI_TOK)
            h, c = np.unique(thash(toks), return_counts=True)
            buf.append((did, h.astype(np.int32), c.astype(np.float16), len(toks)))
            if len(buf) >= shard_docs:
                flush()
    flush()
    n = sum(len(m["doc_ids"]) for m in meta)
    np.save(IDX / "df.npy", df_total)
    json.dump({"n_docs": n, "shards": meta, "bits": BITS}, open(IDX / "meta.json", "w"))
    print(f"done: {n} docs in {time.time() - t0:.0f}s")


class Index:
    def __init__(self):
        meta = json.load(open(IDX / "meta.json"))
        n = meta["n_docs"]
        df = np.load(IDX / "df.npy")
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dls = [np.load(IDX / f"dl_{m['shard']}.npy") for m in meta["shards"]]
        avgdl = float(np.concatenate(dls).mean())
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
        """toks_weighted: dict hash->weight."""
        if not toks_weighted:
            return []
        cols = np.fromiter(toks_weighted.keys(), np.int64)
        w = np.fromiter(toks_weighted.values(), np.float32)
        out = []
        for w_csc, ids in self.shards:
            s = w_csc[:, cols] @ w
            kk = min(k, len(s))
            top = np.argpartition(-s, kk - 1)[:kk]
            out += [(float(s[i]), ids[i]) for i in top if s[i] > 0]
        return sorted(out, reverse=True)[:k]


def qvec(texts_weights):
    d = {}
    for text, wt in texts_weights:
        toks = tokenize(text, "vi")
        for h in np.unique(thash(toks)):  # BM25 chuẩn: mỗi term của query tính 1 lần
            d[int(h)] = max(d.get(int(h), 0.0), wt)
    return d


def fetch_docs(doc_ids):
    need, out = set(doc_ids), {}
    for f in vi_files():
        t = pq.read_table(f, columns=["doc_id", "title", "text"]).to_pandas()
        t = t[t.doc_id.astype(str).isin(need)]
        for did, ti, tx in zip(t.doc_id.astype(str), t.title, t.text):
            out[did] = (ti or "", tx)
    return out


def paragraphs(text, min_chars):
    """Đoạn = khối giữa '\\n\\n'; đoạn ngắn (tiêu đề/câu hỏi) gộp với đoạn sau. Trả về (start,end)."""
    spans, pos = [], 0
    for part in text.split("\n\n"):
        s = text.find(part, pos)
        e = s + len(part)
        pos = e
        if part.strip():
            spans.append([s, e])
    merged, cur = [], None
    for s, e in spans:
        if cur is None:
            cur = [s, e]
        else:
            cur[1] = e
        if cur[1] - cur[0] >= min_chars:
            merged.append(tuple(cur)); cur = None
    if cur is not None:
        if merged and cur[1] - cur[0] < min_chars:
            merged[-1] = (merged[-1][0], cur[1])
        else:
            merged.append(tuple(cur))
    return merged


def cutoff(scores, ratio, max_k):
    if not scores or scores[0] <= 0:
        return 0
    n = 1
    while n < min(max_k, len(scores)) and scores[n] >= ratio * scores[0]:
        n += 1
    return n


def predict(a):
    q = pd.read_parquet(ROOT / "data/ViBioMIR/query.parquet")
    plans = {}
    if a.use_plan:
        for line in open(ROOT / "data/cache/query_plans.jsonl"):
            p = json.loads(line); plans[int(p["id"])] = p
    ix = Index()
    t0 = time.time()
    hits = {}
    for qid, query in zip(q.id, q["query"]):
        tw = [(query, 1.0)]
        p = plans.get(int(qid))
        if p:
            tw += [(v, a.plan_w) for v in p.get("vi", [])[:3]]
            tw += [(" ".join(p.get("keywords", {}).get("vi", [])), a.plan_w)]
        hits[int(qid)] = ix.search(qvec(tw), k=a.doc_max)
    print(f"searched {len(hits)} in {time.time() - t0:.0f}s", flush=True)
    del ix

    sel_docs = {qid: [d for _, d in h[:cutoff([s for s, _ in h], a.doc_ratio, a.doc_max)]] for qid, h in hits.items()}
    texts = fetch_docs({d for v in sel_docs.values() for d in v})
    pred = []
    for qid, query in zip(q.id, q["query"]):
        qid = int(qid)
        qt = set(tokenize(query, "vi"))
        cands = []
        for d in sel_docs[qid]:
            if d not in texts:
                continue
            tx = texts[d][1]
            paras = paragraphs(tx, a.min_chunk)
            scored = []
            for s, e in paras:
                ct = set(tokenize(tx[s:e], "vi"))
                scored.append(len(qt & ct) / (max(len(ct), 1) ** a.len_pow))
            if not paras:
                continue
            best = max(range(len(paras)), key=lambda i: scored[i])
            lo = hi = best
            while paras[hi][1] - paras[lo][0] < a.chunk_chars and (lo > 0 or hi < len(paras) - 1):
                left = scored[lo - 1] if lo > 0 else -1
                right = scored[hi + 1] if hi < len(paras) - 1 else -1
                if right >= left:
                    hi += 1
                else:
                    lo -= 1
            cands.append((scored[best], d, tx[paras[lo][0]:paras[hi][1]]))
        cands.sort(key=lambda x: -x[0])
        per_doc, chunks = {}, []
        for sc, d, ctext in cands:
            if per_doc.get(d, 0) >= a.chunk_per_doc:
                continue
            per_doc[d] = per_doc.get(d, 0) + 1
            chunks.append({"doc_id": d, "chunk_text": ctext})
        pred.append({"id": qid, "relevant_docs": sel_docs[qid], "relevant_chunks": chunks})
    ids = [int(i) for i in q.id]
    pred = complete(pred, ids)
    errs = validate(pred, ids, doc_lookup={d: v[1] for d, v in texts.items()})
    if errs:
        print("VALIDATION ERRORS", errs[:5]); sys.exit(1)
    # BTC yêu cầu doc_id là số nguyên (id trong links_corpus); chuỗi bị hiểu thành pubmed:<id>
    for p in pred:
        p["relevant_docs"] = [int(x) for x in p["relevant_docs"]]
        for c in p["relevant_chunks"]:
            c["doc_id"] = int(c["doc_id"])
    write_zip(pred, a.out)
    json.dump({k: v for k, v in vars(a).items() if k != "func"}, open(str(a.out) + ".cfg.json", "w"))
    print(f"wrote {a.out}: docs/q {np.mean([len(p['relevant_docs']) for p in pred]):.2f}, "
          f"chunks/q {np.mean([len(p['relevant_chunks']) for p in pred]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("predict")
    p.add_argument("--out", required=True)
    p.add_argument("--doc-ratio", type=float, default=0.0)
    p.add_argument("--doc-max", type=int, default=40)
    p.add_argument("--chunk-per-doc", type=int, default=1)
    p.add_argument("--min-chunk", type=int, default=120)
    p.add_argument("--len-pow", type=float, default=0.3)
    p.add_argument("--chunk-chars", type=int, default=800)
    p.add_argument("--use-plan", action="store_true")
    p.add_argument("--plan-w", type=float, default=0.5)
    a = ap.parse_args()
    build() if a.cmd == "build" else predict(a)
