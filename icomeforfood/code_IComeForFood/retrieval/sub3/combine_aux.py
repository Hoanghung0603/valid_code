"""One aux score for make_sub3 --prf: z(topic cosine) + a * z(Qwen3-8B cosine) per query, over docs that have both.
    python combine_aux.py <prf.pkl> <prf K> <q8.pkl> <a> <out.pkl>   (out: {0: {qid: {doc: score}}})"""
import pickle, sys
import numpy as np
prf = pickle.load(open(sys.argv[1], "rb"))[int(sys.argv[2])]
q8 = pickle.load(open(sys.argv[3], "rb"))[0]
a = float(sys.argv[4])
out = {}
for q, P in prf.items():
    Q = q8.get(q, {})
    ds = [d for d in P if d in Q]
    if not ds:
        continue
    x = np.array([P[d] for d in ds]); y = np.array([Q[d] for d in ds])
    z = (x - x.mean()) / (x.std() + 1e-9) + a * (y - y.mean()) / (y.std() + 1e-9)
    out[q] = dict(zip(ds, z.tolist()))
pickle.dump({0: out}, open(sys.argv[5], "wb"))
print(len(out), np.mean([len(v) for v in out.values()]))
