"""Rerank more chunks of each pooled doc: for every (query, doc) in the pool, up to `per_doc` s400 chunks of the doc
that best overlap the query (by distinct query tokens they contain), skipping chunks already scored.

    python rerank_more.py <pool.pkl> <out.pkl> <per_doc> <chunk dirs in run order...>
pool.pkl: {qid: {doc_id: [already scored chunk indices]}}. out.pkl: {qid: {"vi": {chunk: score}, "t": {chunk: score}}},
the same layout as pipeline.py rerank (t = translated query, zh/en chunks only).
"""
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "code/retrieval"))
from pipeline import chunk_text, make_reranker, zh_tokens  # noqa: E402
from translate_queries import key_of, load as load_translations  # noqa: E402

pool_p, out_p, per_doc = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
files = [f for d in sys.argv[4:] for f in sorted(Path(d).glob("*.parquet"))]
_W = re.compile(r"\w+", re.U)


def toks(s: str, lang: str) -> set[str]:
    return set(zh_tokens(s)) if lang == "zh" else set(_W.findall((s or "").lower()))


pool = pickle.loads(pool_p.read_bytes())
need = pa.array(sorted({d for v in pool.values() for d in v}), pa.int64())
c = {"text": {}, "heading_path": {}, "lang": {}}
by_doc: dict[int, list[int]] = {}
base = 0
for f in files:
    t = pq.read_table(f, columns=["doc_id", "lang", "text", "heading_path"])
    rows = np.flatnonzero(pc.is_in(t.column("doc_id"), value_set=need).to_numpy(zero_copy_only=False))
    sub = t.take(pa.array(rows, pa.int64())).to_pydict()
    for r, d, lg, x, h in zip(rows, sub["doc_id"], sub["lang"], sub["text"], sub["heading_path"]):
        i = base + int(r)
        by_doc.setdefault(int(d), []).append(i)
        c["lang"][i], c["text"][i], c["heading_path"][i] = lg, x, h
    base += t.num_rows
print(f"{len(by_doc)} docs, {len(c['text'])} chunks loaded", flush=True)

qt = pq.read_table(W / "data/raw/query.parquet", columns=["id", "query"]).to_pydict()
qtext = dict(zip(qt["id"], qt["query"]))
trans = load_translations(W / "data/qtrans/v1/translations.jsonl")
jobs, pairs = [], []
for qid, docs in pool.items():
    q = qtext[qid]
    tr = trans[key_of(q)]
    qs = {"vi": toks(q, "vi"), "en": toks(tr["en"], "en"), "zh": toks(tr["zh"], "zh")}
    for d, done in docs.items():
        cand = [i for i in by_doc.get(d, []) if i not in set(done)]
        if not cand:
            continue
        lg = c["lang"][cand[0]]
        qq = qs["vi"] | qs.get(lg, set())
        cand.sort(key=lambda i: -len(qq & toks(chunk_text(c, i), lg)))
        for i in cand[:per_doc]:
            jobs.append((qid, i, "vi"))
            pairs.append((q, chunk_text(c, i)))
            if lg in ("en", "zh"):
                jobs.append((qid, i, "t"))
                pairs.append((tr[lg], chunk_text(c, i)))
print(f"{len(pool)} queries, {len(pairs)} pairs", flush=True)
model = make_reranker("bge", 512)
t0 = time.time()
scores = model.score(pairs, batch_size=128)
print(f"reranked in {time.time() - t0:.0f}s ({len(pairs) / (time.time() - t0):.0f} pairs/s)", flush=True)
out: dict[int, dict] = {}
for (qid, i, ql), s in zip(jobs, scores):
    out.setdefault(qid, {}).setdefault(ql, {})[i] = float(s)
pickle.dump(out, open(out_p, "wb"))
print("RERANK_MORE_DONE", flush=True)
