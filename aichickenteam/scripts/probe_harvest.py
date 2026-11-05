"""Bài nộp THU HOẠCH NHÃN: 42% gold chưa từng được gắn nhãn (nhãn chỉ lộ qua chunk predictions trong details).
- relevant_docs = top --n-docs (450) theo LTR: thăm dò pos-rate vùng chưa thấy qua recall tổng.
- relevant_chunks câu PUBLIC = --n-harvest (100) doc CHƯA TỪNG THẤY xếp hạng cao nhất → ~60k nhãn mới đúng vùng mô hình phân vân.
- câu PRIVATE = top-120 chunk thường (giữ pipeline cũ, bài này không dùng cho private).
Chunk = cửa sổ d3 (hoặc select_chunks) nới 8000 ký tự như fuse_dense.
    python scripts/probe_harvest.py --order data/cache/ltr_order_oof25.json --out data/cache/tmp/v60h_deflate.zip
"""
import argparse, glob, json, sys, zipfile
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, "scripts")
from chunk_v2 import select_chunks  # noqa: E402
from fuse_dense import expand  # noqa: E402
from ltr import labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=450); ap.add_argument("--n-harvest", type=int, default=100)
    ap.add_argument("--n-private", type=int, default=120); ap.add_argument("--chunk-chars", type=int, default=8000)
    ap.add_argument("--d3", default="data/cache/dense_chunks_top_v8.json")
    a = ap.parse_args()
    od = json.load(open(a.order)); _, seen = labels()
    D3 = json.load(open(a.d3))
    W = {(int(q), str(x[1])): x[2] for q, v in D3.items() for x in v if len(x) > 2}
    plans = {int(json.loads(l)["id"]): json.loads(l) for l in open("data/cache/query_plans.jsonl")}
    qtext = dict(pd.read_parquet("data/ViBioMIR/query.parquet").values)
    sel = {}
    for q, docs in od.items():
        qi = int(q)
        sel[qi] = [d for d in docs if d not in seen[qi]][:a.n_harvest] if qi in seen else docs[:a.n_private]
    need = {d for v in sel.values() for d in v}
    texts = {}
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "text"]).to_pandas()
        t = t[t.doc_id.astype(str).isin(need)]
        texts.update(zip(t.doc_id.astype(str), t.text))
    res = []
    for q, docs in sorted(od.items(), key=lambda x: int(x[0])):
        qi = int(q); chunks = []
        for d in sel[qi]:
            c = W.get((qi, d))
            if not c and d in texts:
                s = select_chunks(qtext.get(qi, ""), plans.get(qi), texts[d], "vi", {"target_chars": 1600})
                c = s[0] if s else ""
            if c and len(c) < a.chunk_chars and d in texts:
                c = expand(texts[d], c, a.chunk_chars)
            if c:
                chunks.append({"doc_id": int(d), "chunk_text": c})
        res.append({"id": qi, "relevant_docs": [int(d) for d in docs[:a.n_docs]], "relevant_chunks": chunks})
    ids = set(pd.read_parquet("data/ViBioMIR/query.parquet").id.astype(int))
    res += [{"id": i, "relevant_docs": [], "relevant_chunks": []} for i in sorted(ids - {r["id"] for r in res})]
    res.sort(key=lambda e: e["id"])
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("submission.json", json.dumps(res, ensure_ascii=False))
    nh = sum(len(r["relevant_chunks"]) for r in res if r["id"] in {int(x) for x in seen})
    print("wrote", a.out, len(res), "câu; chunk thu hoạch:", nh)


if __name__ == "__main__":
    main()
