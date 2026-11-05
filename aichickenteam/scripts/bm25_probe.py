"""Thăm dò giả thuyết POOLING: gold = doc mà các hệ của BTC trả về (vd. BM25 baseline, precision .88) được LLM duyệt.
Nếu cấu hình BM25 trùng với BTC, top-k của nó gần như toàn nhãn DƯƠNG đã biết, rất ít nhãn ÂM đã biết.
    python scripts/bm25_probe.py --tok word --field title_text --out data/cache/bm25p_word_tt.json
In: tỉ lệ dương/âm/chưa thấy trong top-k (k=10,20,42) trên 600 câu public; so với thứ tự LTR v62.
"""
import argparse, glob, json, re, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import bm25s
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402


def stats(name, top, pos, seen, ks=(10, 20, 42)):
    out = []
    for k in ks:
        p = n = u = 0
        for q in seen:
            for d in top.get(q, [])[:k]:
                if d in pos[q]: p += 1
                elif d in seen[q]: n += 1
                else: u += 1
        t = max(p + n + u, 1); out.append(f"top{k}: dương {p/t:.3f} âm {n/t:.3f} chưa-thấy {u/t:.3f}")
    print(f"{name:28s} " + " | ".join(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="word", choices=["word", "word2", "pyvi"]); ap.add_argument("--field", default="title_text", choices=["title_text", "text", "title"])
    ap.add_argument("--k1", type=float, default=1.5); ap.add_argument("--b", type=float, default=0.75); ap.add_argument("--method", default="lucene")
    ap.add_argument("--maxchars", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--topk", type=int, default=100)
    a = ap.parse_args(); t0 = time.time()
    pos, seen = labels()
    ids, texts = [], []
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "lang", "title", "text"]).to_pandas()
        t = t[t.lang.fillna("vi") == "vi"]
        tt = t.title.fillna("") + "\n" + t.text.fillna("") if a.field == "title_text" else t[a.field].fillna("")
        if a.maxchars: tt = tt.str[:a.maxchars]
        ids += t.doc_id.astype(str).tolist(); texts += tt.tolist()
    print("docs", len(ids), f"{time.time()-t0:.0f}s", flush=True)
    pat = r"(?u)\b\w+\b" if a.tok == "word" else r"(?u)\b\w\w+\b"
    if a.tok == "pyvi":
        from pyvi import ViTokenizer
        texts = [ViTokenizer.tokenize(x) for x in texts]; pat = r"(?u)\b\w+\b"
    tok = bm25s.tokenize(texts, lower=True, token_pattern=pat, stopwords=None, show_progress=False, return_ids=True)
    del texts
    ret = bm25s.BM25(method=a.method, k1=a.k1, b=a.b); ret.index(tok, show_progress=False)
    print("indexed", f"{time.time()-t0:.0f}s", flush=True)
    Q = pd.read_parquet("data/ViBioMIR/query.parquet")
    qt = Q["query"].tolist()
    if a.tok == "pyvi":
        qt = [ViTokenizer.tokenize(x) for x in qt]
    vocab = tok.vocab
    qtok = [[vocab[w] for w in re.findall(pat, x.lower()) if w in vocab] for x in qt]
    res, sc = ret.retrieve(qtok, k=a.topk, show_progress=False, n_threads=2)
    top = {int(q): [ids[i] for i in r] for q, r in zip(Q.id, res)}
    json.dump({str(q): [[float(s), d] for s, d in zip(sc[j], top[int(q)])] for j, q in enumerate(Q.id)}, open(a.out, "w"))
    stats(f"bm25 {a.tok}/{a.field}/k1={a.k1},b={a.b}", top, pos, seen)
    od = json.load(open("data/cache/ltrx4_out/ltr_order_oof31.json")); stats("LTR v62 (oof31)", {int(q): v for q, v in od.items()}, pos, seen)
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
