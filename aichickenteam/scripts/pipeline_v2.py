import argparse
import os
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
from src.submit import complete, validate, write_zip

BITS = 23
K1, B = 1.2, 0.75
MAX_CHARS = 6000

def thash(toks):
    import zlib
    m = (1 << BITS) - 1
    return np.fromiter((zlib.crc32(t.encode()) & m for t in toks), dtype=np.int64, count=len(toks))

def zh_files():
    return sorted(glob.glob(str(ROOT / "data/docs_zh/corpus-*.parquet")))

def build_zh(shard_docs=100_000):
    IDX = ROOT / "data/cache/bm25_zh"
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

    for f in zh_files():
        t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        t = t[t.lang == "zh"]
        for did, ti, tx in zip(t.doc_id.astype(str), t.title.fillna(""), t.text):
            if did in seen: continue
            seen.add(did)
            toks = tokenize(f"{ti}\n{tx[:MAX_CHARS]}", "zh")
            h, c = np.unique(thash(toks), return_counts=True)
            buf.append((did, h.astype(np.int32), c.astype(np.float16), len(toks)))
            if len(buf) >= shard_docs: flush()
    flush()
    n = sum(len(m["doc_ids"]) for m in meta)
    np.save(IDX / "df.npy", df_total)
    json.dump({"n_docs": n, "shards": meta, "bits": BITS}, open(IDX / "meta.json", "w"))
    print(f"done ZH index: {n} docs in {time.time() - t0:.0f}s")


class Index:
    def __init__(self, path):
        self.path = Path(path)
        meta = json.load(open(self.path / "meta.json"))
        n = meta["n_docs"]
        df = np.load(self.path / "df.npy")
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dls = [np.load(self.path / f"dl_{m['shard']}.npy") for m in meta["shards"]]
        avgdl = float(np.concatenate(dls).mean()) if dls else 1.0
        self.shards = []
        for m, dl in zip(meta["shards"], dls):
            tf = sp.load_npz(self.path / f"tf_{m['shard']}.npz").tocsr()
            norm = K1 * (1 - B + B * dl / avgdl)
            rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
            tf.data = (tf.data * (K1 + 1) / (tf.data + norm[rows])) * idf[tf.indices]
            self.shards.append((tf.tocsc(), m["doc_ids"]))
            del rows
        print(f"loaded {n} docs from {path}", flush=True)

    def search(self, toks_weighted, k=20):
        if not toks_weighted: return []
        cols = np.fromiter(toks_weighted.keys(), np.int64)
        w = np.fromiter(toks_weighted.values(), np.float32)
        out = []
        for w_csc, ids in self.shards:
            s = w_csc[:, cols] @ w
            kk = min(k, len(s))
            top = np.argpartition(-s, kk - 1)[:kk]
            out += [(float(s[i]), ids[i]) for i in top if s[i] > 0]
        return sorted(out, reverse=True)[:k]


def qvec(texts_weights, lang):
    d = {}
    for text, wt in texts_weights:
        toks = tokenize(text, lang, {"vi_bigrams": os.environ.get("R2AI_VI_BIGRAMS", "1") == "1"} if lang == "vi" else None)
        for h in np.unique(thash(toks)):
            d[int(h)] = max(d.get(int(h), 0.0), wt)
    return d

def fetch_docs_both(doc_ids):
    need = set(doc_ids)
    out = {}
    for folder, lang in [("data/docs", "vi"), ("data/docs_zh", "zh")]:
        files = glob.glob(str(ROOT / folder / "corpus-*.parquet"))
        for f in files:
            t = pq.read_table(f, columns=["doc_id", "title", "text"]).to_pandas()
            t = t[t.doc_id.astype(str).isin(need)]
            for did, ti, tx in zip(t.doc_id.astype(str), t.title, t.text):
                out[did] = (ti or "", tx, lang)
    return out

def paragraphs(text, min_chars):
    spans, pos = [], 0
    for part in text.split("\n\n"):
        s = text.find(part, pos)
        e = s + len(part)
        pos = e
        if part.strip(): spans.append([s, e])
    merged, cur = [], None
    for s, e in spans:
        if cur is None: cur = [s, e]
        else: cur[1] = e
        if cur[1] - cur[0] >= min_chars:
            merged.append(tuple(cur)); cur = None
    if cur is not None:
        if merged and cur[1] - cur[0] < min_chars: merged[-1] = (merged[-1][0], cur[1])
        else: merged.append(tuple(cur))
    return merged

def predict(a):
    q = pd.read_parquet(ROOT / "data/ViBioMIR/query.parquet")
    plans = {}
    if a.use_plan and os.environ.get("R2AI_NO_PLAN") != "1":
        for line in open(ROOT / "data/cache/query_plans.jsonl"):
            p = json.loads(line); plans[int(p["id"])] = p
    ix_vi = Index(ROOT / os.environ.get("R2AI_BM25_VI_DIR", "data/cache/bm25_vi"))
    try:
        ix_zh = Index(ROOT / "data/cache/bm25_zh")
    except:
        ix_zh = None

    t0 = time.time()
    hits_vi, hits_zh = {}, {}
    for qid, query in zip(q.id, q["query"]):
        tw_vi = [(query, 1.0)]
        p = plans.get(int(qid))
        if p:
            tw_vi += [(v, a.plan_w) for v in p.get("vi", [])[:3]]
            tw_vi += [(" ".join(p.get("keywords", {}).get("vi", [])), a.plan_w)]
        hits_vi[int(qid)] = ix_vi.search(qvec(tw_vi, "vi"), k=a.doc_max_vi)

        if ix_zh and p:
            tw_zh = [(v, a.plan_w) for v in p.get("zh", [])[:3]]
            tw_zh += [(" ".join(p.get("keywords", {}).get("zh", [])), a.plan_w)]
            hits_zh[int(qid)] = ix_zh.search(qvec(tw_zh, "zh"), k=a.doc_max_zh)
        else:
            hits_zh[int(qid)] = []

    sel_docs = {}
    for qid in q.id.astype(int):
        d_vi = [d for _, d in hits_vi[qid][:a.doc_max_vi]]
        d_zh = [d for _, d in hits_zh[qid][:a.doc_max_zh]]
        sel_docs[qid] = d_vi + d_zh

    texts = fetch_docs_both({d for v in sel_docs.values() for d in v})
    pred = []
    for qid, query in zip(q.id, q["query"]):
        qid = int(qid)
        qt_vi = set(tokenize(query, "vi"))
        
        # For ZH, what is the best query terms to match? 
        # Usually translating the query to zh for chunk matching. 
        # But we can just use the plan["zh"] as the query to match chunk.
        p = plans.get(qid)
        zh_query = " ".join(p.get("zh", [])) + " " + " ".join(p.get("keywords", {}).get("zh", [])) if p else query
        qt_zh = set(tokenize(zh_query, "zh"))

        cands = []
        for d in sel_docs[qid]:
            if d not in texts: continue
            _, tx, lang = texts[d]
            target_chars = a.chunk_chars_vi if lang == "vi" else a.chunk_chars_zh
            qt = qt_vi if lang == "vi" else qt_zh
            
            paras = paragraphs(tx, a.min_chunk)
            scored = []
            for s, e in paras:
                ct = set(tokenize(tx[s:e], lang))
                scored.append(len(qt & ct) / (max(len(ct), 1) ** a.len_pow))
            if not paras: continue
            best = max(range(len(paras)), key=lambda i: scored[i])
            lo = hi = best
            while paras[hi][1] - paras[lo][0] < target_chars and (lo > 0 or hi < len(paras) - 1):
                left = scored[lo - 1] if lo > 0 else -1
                right = scored[hi + 1] if hi < len(paras) - 1 else -1
                if right >= left: hi += 1
                else: lo -= 1
            cands.append((scored[best], d, tx[paras[lo][0]:paras[hi][1]], lang))
        
        cands.sort(key=lambda x: -x[0])
        per_doc, chunks = {}, []
        for sc, d, ctext, lang in cands:
            if per_doc.get(d, 0) >= a.chunk_per_doc: continue
            per_doc[d] = per_doc.get(d, 0) + 1
            chunks.append({"doc_id": d, "chunk_text": ctext})
        pred.append({"id": qid, "relevant_docs": sel_docs[qid], "relevant_chunks": chunks})
    
    ids = [int(i) for i in q.id]
    pred = complete(pred, ids)
    errs = validate(pred, ids, doc_lookup={d: v[1] for d, v in texts.items()})
    if errs:
        print("VALIDATION ERRORS", errs[:5]); sys.exit(1)
    
    for p in pred:
        p["relevant_docs"] = [int(x) for x in p["relevant_docs"]]
        for c in p["relevant_chunks"]:
            c["doc_id"] = int(c["doc_id"])
            
    if hasattr(a, "dump_preds") and a.dump_preds:
        with open(a.dump_preds, "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False)
        print(f"Dumped predictions to {a.dump_preds}")
            
    write_zip(pred, a.out)
    json.dump({k: v for k, v in vars(a).items() if k != "func"}, open(str(a.out) + ".cfg.json", "w"))
    print(f"wrote {a.out}: docs/q {np.mean([len(p['relevant_docs']) for p in pred]):.2f}, "
          f"chunks/q {np.mean([len(p['relevant_chunks']) for p in pred]):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build_zh")
    p = sub.add_parser("predict")
    p.add_argument("--out", required=True)
    p.add_argument("--doc-max-vi", type=int, default=20)
    p.add_argument("--doc-max-zh", type=int, default=40)
    p.add_argument("--chunk-per-doc", type=int, default=1)
    p.add_argument("--min-chunk", type=int, default=120)
    p.add_argument("--len-pow", type=float, default=0.3)
    p.add_argument("--chunk-chars-vi", type=int, default=800)
    p.add_argument("--chunk-chars-zh", type=int, default=500)
    p.add_argument("--use-plan", action="store_true", default=True)
    p.add_argument("--plan-w", type=float, default=0.5)
    p.add_argument("--dump-preds", help="Dump raw predictions to JSON")
    a = ap.parse_args()
    if a.cmd == "build_zh": build_zh()
    else: predict(a)
