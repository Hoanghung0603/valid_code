"""Submissions for sub3: rerank scores with a penalty on Chinese/English docs, optional RRF with the whole-doc BM25
ranking, and long chunks (the best s400 chunk widened by `win` neighbours on each side, cut from the doc text) for the
first `chunk_docs` docs only (optionally the bare s400 chunk for the docs after them, up to `plain_docs`), so the zip stays under the ~100MB upload limit.

    python make_sub3.py --run D:/r2ai_data/runs/sub3 --chunks <s400 dirs in run order> --docs <doc dirs>
        --pen 0 2 100 --fuse 0 1 --n-docs 120 150 --chunk-docs 40 60 --win 1
Names: s3_p<pen>_f<fuse>_d<n>_k<chunk_docs>_w<win>[_c<plain_docs>].zip
"""
import argparse
import itertools
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--chunks", type=Path, nargs="+", required=True)
ap.add_argument("--docs", type=Path, nargs="+", required=True)
ap.add_argument("--extra", type=Path, nargs="*", default=[], help="more rerank score pkls (rerank_more.py), merged in")
ap.add_argument("--bm25doc", type=Path, help="bm25doc.pkl, read only when --fuse > 0 (default <run>/bm25doc/bm25doc.pkl)")
ap.add_argument("--queries", type=Path, default=Path(os.environ.get("R2AI_DATA", "D:/r2ai_data")) / "raw/query.parquet")
ap.add_argument("--pen", type=float, nargs="+", default=[0.0], help="subtracted from zh/en rerank scores")
ap.add_argument("--fuse", type=float, nargs="+", default=[0.0], help="RRF weight of the whole-doc BM25 ranking")
ap.add_argument("--n-docs", type=int, nargs="+", default=[120])
ap.add_argument("--chunk-docs", type=int, nargs="+", default=[40])
ap.add_argument("--win", type=int, nargs="+", default=[1])
ap.add_argument("--plain-docs", type=int, nargs="+", default=[0], help="docs past chunk_docs, up to this rank, get the bare best s400 chunk")
ap.add_argument("--pad", type=int, nargs="+", default=[0], help="if > 0: chunk = best s400 chunk +- pad chars (instead of win)")
ap.add_argument("--thr", type=float, nargs="+", default=[None], help="per-query cutoff: docs with score >= thr, at least --min-docs, at most n_docs")
ap.add_argument("--min-docs", type=int, default=100)
ap.add_argument("--cpd", type=int, default=1, help="chunks per doc for the first chunk_docs docs: windows around the cpd best-scored, non-adjacent s400 chunks")
ap.add_argument("--prf", type=Path, help="prf.py output; adds prf_w * z(topic cosine) to z(rerank) for its pooled docs")
ap.add_argument("--prf-k", type=int, default=20)
ap.add_argument("--prf-w", type=float, nargs="+", default=[0.0])
ap.add_argument("--chunk-cut", action="store_true", help="chunks only for the submitted docs (after the per-query cut)")
ap.add_argument("--bz2", action="store_true", help="bzip2 instead of deflate: about a third smaller, room for more chunk text")
ap.add_argument("--out", type=Path)
args = ap.parse_args()
out = args.out or args.run / "submit"
out.mkdir(parents=True, exist_ok=True)

files = [f for d in args.chunks for f in sorted(d.glob("*.parquet"))]
meta = pq.read_table(files, columns=["doc_id", "lang", "char_start", "char_end"])
doc_of = meta.column("doc_id").to_numpy()
lang = np.array([x != "vi" for x in meta.column("lang").to_pylist()])  # True for zh/en chunks
c_start, c_end = meta.column("char_start").to_numpy(), meta.column("char_end").to_numpy()
del meta
rr = pickle.loads((args.run / "rerank.pkl").read_bytes())["scores"]
for p in args.extra:
    for q, r in pickle.loads(p.read_bytes()).items():
        for ql, sc in r.items():
            rr.setdefault(q, {}).setdefault(ql, {}).update(sc)
bm_rank = {}
if any(args.fuse):
    bm = pickle.load(open(args.bm25doc or args.run / "bm25doc" / "bm25doc.pkl", "rb"))
    bm_rank = {int(q): [int(d) for d in row] for q, row in zip(bm["qids"], bm["doc"])}
qids = pq.read_table(args.queries, columns=["id"]).column("id").to_pylist()

# best chunk and score per doc (max over the Vietnamese and translated query)
best: dict[int, dict[int, tuple[float, int]]] = {}
ranked_ch: dict[int, dict[int, list[int]]] = {}  # query -> doc -> its scored chunks, best first
for q in qids:
    s = dict(rr[q]["vi"])
    for i, z in rr[q].get("t", {}).items():
        s[i] = max(s.get(i, z), z)
    b: dict[int, tuple[float, int]] = {}
    rc: dict[int, list[int]] = {}
    for i, z in sorted(s.items(), key=lambda x: -x[1]):
        rc.setdefault(int(doc_of[i]), []).append(i)
    ranked_ch[q] = rc
    for i, z in s.items():
        d = int(doc_of[i])
        if d not in b or z > b[d][0]:
            b[d] = (z, i)
    best[q] = b


prf = pickle.loads(args.prf.read_bytes())[args.prf_k] if args.prf else {}


def ranking(q: int, pen: float, fuse: float, pw: float = 0.0) -> list[int]:
    sc = {d: z - (pen if lang[i] else 0.0) for d, (z, i) in best[q].items()}
    r = sorted(sc, key=sc.__getitem__, reverse=True)
    if pw and q in prf:  # re-order the pooled docs by z(rerank) + pw * z(topic cosine); the rest keep their order after
        P = prf[q]
        top = [d for d in r if d in P]
        a = np.array([sc[d] for d in top]); b = np.array([P[d] for d in top])
        f = (a - a.mean()) / (a.std() + 1e-9) + pw * (b - b.mean()) / (b.std() + 1e-9)
        top = [top[j] for j in np.argsort(-f)]
        r = top + [d for d in r if d not in P]
    if not fuse:
        return r
    f = {d: 1 / (60 + k) for k, d in enumerate(r, 1)}
    for k, d in enumerate(bm_rank[q], 1):
        f[d] = f.get(d, 0) + fuse / (60 + k)
    return sorted(f, key=f.__getitem__, reverse=True)


combos = list(itertools.product(args.pen, args.fuse, args.prf_w))
ranks = {(p, f, w): {q: ranking(q, p, f, w) for q in qids} for p, f, w in combos}
max_n, max_k = max(args.n_docs), max(args.chunk_docs + args.plain_docs)
need = {d for R in ranks.values() for r in R.values() for d in r[:max_k]}
print(f"{len(need)} docs need text", flush=True)

# s400 spans of those docs (for neighbours) and the doc texts
spans: dict[int, list[tuple[int, int, int]]] = {}  # doc -> [(start, end, chunk index)]
for i in np.flatnonzero(np.isin(doc_of, np.array(sorted(need), np.int64))):
    spans.setdefault(int(doc_of[i]), []).append((int(c_start[i]), int(c_end[i]), int(i)))
for v in spans.values():
    v.sort()
text: dict[int, str] = {}
na = pa.array(sorted(need), pa.int64())
for f in [f for d in args.docs for f in sorted(d.glob("*.parquet"))]:
    t = pq.read_table(f, columns=["id", "text"])
    t = t.filter(pc.is_in(t.column("id"), value_set=na))
    text.update(zip(t.column("id").to_pylist(), t.column("text").to_pylist()))
print(f"{len(text)} doc texts loaded", flush=True)


def window(q: int, d: int, win: int, pad: int = 0, i: int | None = None) -> str | None:
    sp, tx = spans.get(d), text.get(d)
    if not sp or not tx:
        return None
    if i is None:
        i = best[q][d][1] if d in best[q] else None
    k = next((j for j, s in enumerate(sp) if s[2] == i), 0)
    if pad:
        return tx[max(0, sp[k][0] - pad):sp[k][1] + pad]
    return tx[sp[max(0, k - win)][0]:sp[min(len(sp) - 1, k + win)][1]]


def cut(q: int, p: float, f: float, pw: float, n: int, thr: float | None) -> list[int]:
    r = ranks[(p, f, pw)][q][:n]
    if thr is None or f:
        return r
    keep = [d for d in r if best[q][d][0] - (p if lang[best[q][d][1]] else 0.0) >= thr]
    return r[:max(len(keep), args.min_docs)]


for (p, f, pw), n, kd, w, kp, pad, thr in itertools.product(combos, args.n_docs, args.chunk_docs, args.win, args.plain_docs,
                                                       args.pad, args.thr):
    sub = []
    for q in qids:
        docs = cut(q, p, f, pw, n, thr)
        cdocs = docs if args.chunk_cut else ranks[(p, f, pw)][q]  # by default chunks follow the ranking, not the cut
        ch = []
        for r, d in enumerate(cdocs[:max(kd, kp)]):
            x = window(q, d, w if r < kd else 0, pad if r < kd else 0)
            if x:
                ch.append({"doc_id": d, "chunk_text": x})
            if r < kd and args.cpd > 1:  # more windows, around the next best chunks that the first window misses
                pos = {c[2]: j for j, c in enumerate(spans.get(d, []))}
                used = [pos.get(best[q][d][1])] if d in best[q] else []
                for i in ranked_ch[q].get(d, [])[1:]:
                    if len(used) >= args.cpd or i not in pos:
                        continue
                    if all(u is None or abs(pos[i] - u) > 2 * w for u in used):
                        used.append(pos[i])
                        y = window(q, d, w, pad, i)
                        if y:
                            ch.append({"doc_id": d, "chunk_text": y})
        sub.append({"id": int(q), "relevant_docs": docs, "relevant_chunks": ch})
    name = f"s3_p{p:g}_f{f:g}" + (f"_prf{args.prf_k}w{pw:g}" if pw else "") + f"_d{n}_k{kd}_w{w}" + (f"_c{kp}" if kp > kd else "") + (f"_pad{pad}" if pad else "") + (f"_cpd{args.cpd}" if args.cpd > 1 else "")         + (f"_t{thr:g}" if thr is not None else "") + ("_cc" if args.chunk_cut else "") + ("_bz" if args.bz2 else "")
    js = out / f"{name}.json"
    js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    with zipfile.ZipFile(out / f"{name}.zip", "w", zipfile.ZIP_BZIP2 if args.bz2 else zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        z.write(js, arcname=js.name)
    zh = np.mean([lang[best[s["id"]][d][1]] for s in sub for d in s["relevant_docs"] if d in best[s["id"]]])
    print(f"{name}: {np.mean([len(x['relevant_docs']) for x in sub]):.0f} docs/q, zh/en share {zh:.2f}, {(out / f'{name}.zip').stat().st_size / 1e6:.1f}MB", flush=True)
    js.unlink()
