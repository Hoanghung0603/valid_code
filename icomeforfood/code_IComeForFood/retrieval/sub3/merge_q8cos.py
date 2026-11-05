"""Qwen3-Embedding-8B cosines from both runs in one aux source: the full-corpus top 500 (qemb.py) and the pooled top 300
(qemb_pool.py); same model and doc text, so the values agree where both have a doc.
    python merge_q8cos.py <q8pool300.pkl> <qemb8.pkl> <out.pkl>   (out: {0: {qid: {doc: cosine}}})"""
import pickle
import sys

pool = pickle.load(open(sys.argv[1], "rb"))[0]
full = pickle.load(open(sys.argv[2], "rb"))
out = {}
for j, q in enumerate(full["qids"]):
    d = dict(zip(full["doc"][j].tolist(), full["score"][j].tolist()))
    d.update(pool.get(int(q), {}))
    out[int(q)] = d
pickle.dump({0: out}, open(sys.argv[3], "wb"))
print(len(out), "queries")
