"""Chọn lại chunk cho các doc CHỈ có từ dense (chưa được GreenNode chấm đoạn) bằng GreenNode.

    python scripts/chunk_rerank.py --sub submissions/v10_dense_w2.zip --preds data/cache/vi_preds_300.jsonl \
        --out submissions/v13_chunkrr.zip --windows 4
Cửa sổ ứng viên mỗi doc: tối đa --windows cửa sổ liên tiếp ~1600 ký tự (gộp đoạn theo '\\n\\n') + chunk hiện tại.
"""
import argparse
import glob
import json
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MODEL = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"


def windows(text, target=1600, max_w=4):
    paras, pos = [], 0
    for part in text.split("\n\n"):
        s = text.find(part, pos); e = s + len(part); pos = e
        if part.strip():
            paras.append((s, e))
    out, i = [], 0
    while i < len(paras) and len(out) < max_w:
        s, e = paras[i]
        j = i
        while e - s < target and j + 1 < len(paras):
            j += 1; e = paras[j][1]
        out.append(text[s:e]); i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True); ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--top", type=int, default=200); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    sub = json.loads(zipfile.ZipFile(a.sub).read(zipfile.ZipFile(a.sub).namelist()[0]))
    t = open(a.preds).read()
    try:
        preds = json.loads(t)
    except json.JSONDecodeError:
        preds = [json.loads(l) for l in t.splitlines() if l.strip()]
    bm = {int(e["id"]): set(map(str, e["relevant_docs"][:a.top])) for e in preds}
    need = {(e["id"], str(c["doc_id"])) for e in sub for c in e["relevant_chunks"] if str(c["doc_id"]) not in bm[e["id"]]}
    docs = {d for _, d in need}
    texts = {}
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        tb = pq.read_table(f, columns=["doc_id", "text"]).to_pandas()
        tb = tb[tb.doc_id.astype(str).isin(docs)]
        texts.update(zip(tb.doc_id.astype(str), tb.text))
    q = dict(pd.read_parquet("data/ViBioMIR/query.parquet").values)
    cand = []                     # (qid, doc, window_text)
    for e in sub:
        for c in e["relevant_chunks"]:
            d = str(c["doc_id"])
            if (e["id"], d) in need and d in texts:
                ws = windows(texts[d], max_w=a.windows)
                for w in dict.fromkeys(ws + [c["chunk_text"]]):
                    cand.append((e["id"], d, w))
    print("docs to fix", len(need), "windows", len(cand), flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 512
    uq = sorted({c[0] for c in cand})
    qe = dict(zip(uq, m.encode([q[i] for i in uq], normalize_embeddings=True, batch_size=32)))
    we = m.encode([c[2] for c in cand], normalize_embeddings=True, batch_size=a.batch, show_progress_bar=True)
    sc = (we * np.stack([qe[c[0]] for c in cand])).sum(1)
    best = {}
    for (qid, d, w), s in zip(cand, sc):
        if (qid, d) not in best or s > best[(qid, d)][0]:
            best[(qid, d)] = (s, w)
    changed = 0
    for e in sub:
        for c in e["relevant_chunks"]:
            k = (e["id"], str(c["doc_id"]))
            if k in best and best[k][1] != c["chunk_text"]:
                c["chunk_text"] = best[k][1]; changed += 1
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("submission.json", json.dumps(sub, ensure_ascii=False))
    print("changed", changed, "wrote", a.out)


if __name__ == "__main__":
    main()
