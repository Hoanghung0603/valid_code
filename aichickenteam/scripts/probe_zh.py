"""Bài THĂM DÒ ZH: mỗi câu PUBLIC nộp top-N doc ZH (bộ 235k) theo --by (ce0 | ce2 | dense), chunk = title\\n text[:--chars] (nguyên văn).
Câu private: rỗng. Mục đích: đo tỉ lệ gold ZH theo hạng để hiệu chỉnh khi trộn vào bài chính.
    python scripts/probe_zh.py --pairs data/cache/zhfeat_out/zh_pairs.parquet --docs data/cache/kg_zhdocs/zh_docs.parquet --n 50 --out submissions/v69_probe_zh.zip
"""
import argparse, json, sys, zipfile
import pandas as pd
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--docs", required=True)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--by", default="ce0"); ap.add_argument("--chars", type=int, default=3000); ap.add_argument("--out", required=True)
a = ap.parse_args(); _, seen = labels()
P = pd.read_parquet(a.pairs); P["doc_id"] = P.doc_id.astype(str)
D = pd.read_parquet(a.docs); D["doc_id"] = D.doc_id.astype(str); TX = dict(zip(D.doc_id, D.text.fillna(""))); TI = dict(zip(D.doc_id, D.title.fillna("")))
res, n = [], 0
for q in sorted(int(x) for x in pd.read_parquet("data/ViBioMIR/query.parquet").id):
    if q not in seen:
        res.append({"id": q, "relevant_docs": [], "relevant_chunks": []}); continue
    g = P[P.qid == q].sort_values(a.by, ascending=False).head(a.n)
    docs = g.doc_id.tolist(); ch = [{"doc_id": int(d), "chunk_text": TX[d][:a.chars]} for d in docs if len(TX.get(d, "")) > 30]
    n += len(ch); res.append({"id": q, "relevant_docs": [int(d) for d in docs], "relevant_chunks": ch})
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.writestr("submission.json", json.dumps(res, ensure_ascii=False))
print("wrote", a.out, "chunks", n)
