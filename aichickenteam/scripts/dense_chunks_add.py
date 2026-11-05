"""Dense cấp đoạn (GreenNode, như kaggle/dense_chunks.py) cho các doc MỚI rồi gộp vào kết quả cũ.

    python scripts/dense_chunks_add.py --old data/cache/kaggle_out/dense_chunks_top.json \
        --index data/cache/dense_gn --out data/cache/dense_chunks_top_v2.json
Doc mới = doc VI có trong data/docs nhưng không xuất hiện trong --known (ids của lần chạy Kaggle: dataset r2ai-vi-docs).
"""
import argparse
import glob
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "kaggle")
from dense_chunks import windows  # noqa: E402

MODEL = "GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--known", default="/tmp/claude-1000/-home-phong-Desktop-hsk-test2/0c85d412-25bd-4023-b8a4-93fe29ebc96d/scratchpad/kaggle_ds")
    ap.add_argument("--topn", type=int, default=300)
    a = ap.parse_args()
    known = set()
    for f in glob.glob(f"{a.known}/corpus-*.parquet"):
        known |= set(pq.read_table(f, columns=["doc_id"]).column("doc_id").to_pylist())
    W, D, seen = [], [], set()
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "lang", "text"]).to_pandas()
        t = t[t.lang == "vi"]
        for did, tx in zip(t.doc_id.astype(str), t.text.fillna("")):
            if did in known or did in seen or not tx:
                continue
            seen.add(did)
            for w in windows(tx):
                W.append(w); D.append(did)
    print("new docs", len(seen), "windows", len(W), flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float16}); m.max_seq_length = 512
    q = pd.read_parquet("data/ViBioMIR/query.parquet")
    Q = torch.tensor(m.encode(q["query"].tolist(), normalize_embeddings=True, batch_size=32)).float()
    E = torch.tensor(m.encode(W, normalize_embeddings=True, batch_size=32, show_progress_bar=True)).float()
    S = (Q @ E.T).numpy()
    old = json.load(open(a.old))
    out = {}
    for r, qid in enumerate(q.id):
        best = {}
        for s, d, w in old.get(str(qid), []):
            best[str(d)] = (s, w)
        o = np.argsort(-S[r])[:a.topn * 3]
        for i in o:
            d = D[i]
            if d not in best or S[r, i] > best[d][0]:
                best[d] = (float(S[r, i]), W[i])
        top = sorted(best.items(), key=lambda kv: -kv[1][0])[:a.topn]
        out[str(int(qid))] = [[s, d, w] for d, (s, w) in top]
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
