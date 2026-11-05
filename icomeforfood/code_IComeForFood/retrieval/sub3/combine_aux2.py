"""Aux score for make_sub3 --prf over every doc that has at least one signal: sum of w_k * z_k, where z_k is the
per-query z-score of source k over the docs it covers and 0 (its mean) for docs it does not cover.
    python combine_aux2.py <out.pkl> <pkl>:<key>:<w> ...   (sources in prf.py layout {key: {qid: {doc: score}}};
    a source given as qemb:<pkl>:<w> reads a qemb.py output {qids, doc, score})"""
import pickle, sys
import numpy as np
srcs = []
for spec in sys.argv[2:]:
    if spec.startswith("qemb:"):
        _, p, w = spec.split(":", 2)
        r = pickle.load(open(p, "rb"))
        d = {int(q): dict(zip(r["doc"][j].tolist(), r["score"][j].tolist())) for j, q in enumerate(r["qids"])}
    else:
        p, k, w = spec.rsplit(":", 2)
        d = pickle.load(open(p, "rb"))[int(k)]
    srcs.append((d, float(w)))
out = {}
for q in set().union(*[set(d) for d, _ in srcs]):
    tot = {}
    for d, w in srcs:
        S = d.get(q, {})
        if not S:
            continue
        v = np.array(list(S.values()), dtype=np.float64); m, s = v.mean(), v.std() + 1e-9
        for doc, x in S.items():
            tot[doc] = tot.get(doc, 0.0) + w * (x - m) / s
    out[q] = tot
pickle.dump({0: out}, open(sys.argv[1], "wb"))
print(len(out), np.mean([len(v) for v in out.values()]))
