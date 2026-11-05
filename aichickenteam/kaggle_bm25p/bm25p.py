"""Kaggle CPU: thăm dò giả thuyết POOLING (gold = doc các hệ của BTC trả về rồi LLM duyệt; BM25 baseline precision .88).
Mỗi cấu hình BM25 (tách token × trường × k1/b/method) → top-k trên 600 câu public → tỉ lệ dương/âm/chưa-thấy theo nhãn đã biết.
Cấu hình trùng BTC ⇒ top-42 gần như toàn DƯƠNG, rất ít ÂM. Output: bm25p_<cfg>.json (top-100/câu, cả 1200 câu) + results.txt"""
import glob, json, multiprocessing as mp, os, re, subprocess, sys, time
subprocess.run("pip install -q bm25s pyvi 2>&1 | tail -1", shell=True)
import numpy as np, pandas as pd, pyarrow.parquet as pq, bm25s
t0 = time.time(); W = "/kaggle/working"
G = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)
L = json.load(open(G("labels.json")[0])); pos = {int(q): set(v[0]) for q, v in L.items()}; neg = {int(q): set(v[1]) for q, v in L.items()}
LTR = {int(q): v for q, v in json.load(open(G("ltr_top100.json")[0])).items()}
Q = pd.read_parquet(G("query.parquet")[0])
ids, title, text = [], [], []
for f in sorted(G("corpus-*.parquet")):
    t = pq.read_table(f, columns=["doc_id", "title", "text"]).to_pandas()
    ids += t.doc_id.astype(str).tolist(); title += t.title.fillna("").tolist(); text += t.text.fillna("").tolist()
print("docs", len(ids), f"{time.time()-t0:.0f}s", flush=True)
out = open(f"{W}/results.txt", "w")


def stats(name, top):
    s = []
    for k in (10, 20, 42, 100):
        p = n = u = 0
        for q in pos:
            for d in top.get(q, [])[:k]:
                if d in pos[q]: p += 1
                elif d in neg[q]: n += 1
                else: u += 1
        t = max(p + n + u, 1); s.append(f"@{k} +{p/t:.3f} -{n/t:.3f} ?{u/t:.3f}")
    line = f"{name:40s} " + " | ".join(s); print(line, flush=True); out.write(line + "\n"); out.flush()


stats("LTR v62 (oof31)", LTR)


def pyvi_chunk(xs):
    from pyvi import ViTokenizer
    return [ViTokenizer.tokenize(x) for x in xs]


def run(tokname, field, maxc, variants):
    tt = [(a + "\n" + b) if field == "title_text" else (b if field == "text" else a) for a, b in zip(title, text)]
    if maxc: tt = [x[:maxc] for x in tt]
    qt = Q["query"].tolist()
    pat = r"(?u)\b\w\w+\b" if tokname == "word2" else r"(?u)\b\w+\b"
    if tokname == "pyvi":
        with mp.Pool(4) as pool:
            parts = pool.map(pyvi_chunk, [tt[i:i + 5000] for i in range(0, len(tt), 5000)])
        tt = [x for p in parts for x in p]; qt = pyvi_chunk(qt)
    tok = bm25s.tokenize(tt, lower=True, token_pattern=pat, stopwords=None, show_progress=False); del tt
    qtok = [[w for w in re.findall(pat, x.lower()) if w in tok.vocab] for x in qt]
    qids = [[tok.vocab[w] for w in x] for x in qtok]
    for method, k1, b in variants:
        r = bm25s.BM25(method=method, k1=k1, b=b); r.index(tok, show_progress=False)
        res, sc = r.retrieve(qids, k=100, show_progress=False, n_threads=4)
        top = {int(q): [ids[i] for i in rr] for q, rr in zip(Q.id, res)}
        name = f"{tokname}/{field}/{maxc or 'full'}/{method}/k1={k1},b={b}"
        stats(name, top)
        json.dump({str(q): top[int(q)] for q in Q.id}, open(f"{W}/bm25p_{name.replace('/', '_')}.json", "w"))
        del r
    print(f"[{time.time()-t0:.0f}s] done {tokname}/{field}", flush=True)


V = [("lucene", 1.5, 0.75), ("robertson", 1.5, 0.75), ("lucene", 1.2, 0.75), ("lucene", 0.9, 0.4)]
run("word", "title_text", 0, V)
run("word", "text", 0, V[:2])
run("word2", "title_text", 0, V[:2])
run("word", "title_text", 2000, V[:2])
run("pyvi", "title_text", 3000, V[:2])
print("ALLDONE", f"{time.time()-t0:.0f}s")
