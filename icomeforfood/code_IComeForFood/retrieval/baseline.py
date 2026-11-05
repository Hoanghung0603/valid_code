"""First retrieval baseline (KE_HOACH.md 5, 28.2): BM25 on Vietnamese chunks + BGE-M3 dense on all chunks,
fused with RRF; a doc scores as its best chunk (BioASQ style).

    encode   BGE-M3 dense vectors of a chunk file -> <out>/dense.npy (fp16, normalized) + chunk ids
    run      rank chunks for a query set, write a submission JSON per setting and recall@k

BM25 sees only Vietnamese chunks: queries are Vietnamese and nothing translates them yet, so Chinese and
English docs are reached through the dense branch only. Vietnamese is tokenized into syllables plus
syllable bigrams, a stand-in until textproc/ (T2.2) gives word segmentation.

Usage (env with torch + FlagEmbedding + bm25s):
    python -I retrieval/baseline.py encode --chunks D:/r2ai_data/chunks40k/p150 --out D:/r2ai_data/index40k/p150
    python -I retrieval/baseline.py run --chunks D:/r2ai_data/chunks40k/p150 --index D:/r2ai_data/index40k/p150 \
        --queries D:/r2ai_data/devset/v0/devset.jsonl --gold D:/r2ai_data/devset/v0/dev_gold.json \
        --out D:/r2ai_data/runs/p150_baseline
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from r2ai_eval import normalize  # noqa: E402
from score_submission import score_query  # noqa: E402

_WORD = re.compile(r"\w+")
RRF_K = 60
# the PyTorch weights only; the repo also holds a 2.2GB ONNX copy we do not use
BGE_M3_FILES = ["*.json", "*.bin", "*.pt", "*.model", "1_Pooling/*"]


def bge_m3():
    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download
    return BGEM3FlagModel(snapshot_download("BAAI/bge-m3", allow_patterns=BGE_M3_FILES), use_fp16=True)


def load_chunks(path: str) -> dict[str, list]:
    cols = ["chunk_id", "doc_id", "text", "lang", "heading_path"]
    t = pq.read_table(sorted(Path(path).glob("*.parquet")), columns=cols) if Path(path).is_dir() \
        else pq.read_table(path, columns=cols)
    return {c: t.column(c).to_pylist() for c in cols}


def embed_text(c: dict, i: int) -> str:
    """What the dense model reads: the heading path gives the chunk its topic."""
    head = c["heading_path"][i] or ""
    return f"{head}\n{c['text'][i]}" if head else c["text"][i]


def vi_tokens(text: str) -> list[str]:
    syl = _WORD.findall(normalize(text))
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


def cmd_encode(args) -> None:
    c = load_chunks(args.chunks)
    texts = [embed_text(c, i) for i in range(len(c["text"]))]
    order = np.argsort([len(t) for t in texts])[::-1]  # long first: padding waste stays low
    model = bge_m3()
    t0 = time.time()
    vecs = model.encode([texts[i] for i in order], batch_size=args.batch, max_length=args.max_length,
                        return_dense=True, return_sparse=False)["dense_vecs"]
    dense = np.empty_like(vecs)
    dense[order] = vecs
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dense.npy", dense.astype(np.float16))
    (out / "chunk_ids.json").write_text(json.dumps(c["chunk_id"]), encoding="utf-8")
    print(f"{len(texts)} chunks in {time.time() - t0:.0f}s -> {out}")


def rrf(rankings: list[np.ndarray], n: int) -> np.ndarray:
    score = np.zeros(n, dtype=np.float32)
    for r in rankings:
        score[r] += 1.0 / (RRF_K + np.arange(1, len(r) + 1))
    return score


def cmd_run(args) -> None:
    import bm25s
    import torch

    c = load_chunks(args.chunks)
    ids = json.loads((Path(args.index) / "chunk_ids.json").read_text(encoding="utf-8"))
    assert ids == c["chunk_id"], "index was built from another chunk file"
    n = len(ids)
    doc_of = np.array(c["doc_id"])
    queries = [json.loads(l) for l in Path(args.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = {g["id"]: g for g in json.loads(Path(args.gold).read_text(encoding="utf-8"))}
    queries = [q for q in queries if q["id"] in gold]

    vi_idx = np.array([i for i, lang in enumerate(c["lang"]) if lang == "vi"])
    bm = bm25s.BM25()
    bm.index([vi_tokens(embed_text(c, i)) for i in vi_idx], show_progress=False)

    dense = torch.from_numpy(np.load(Path(args.index) / "dense.npy")).cuda()
    model = bge_m3()
    qv = model.encode([q["query"] for q in queries], batch_size=32, max_length=512)["dense_vecs"]
    qv = torch.from_numpy(np.asarray(qv, dtype=np.float16)).cuda()

    k = args.depth
    runs = {name: [] for name in ("bm25", "dense", "rrf")}
    for qi, q in enumerate(queries):
        docs_b, _ = bm.retrieve([vi_tokens(q["query"])], k=k, show_progress=False)
        r_bm = vi_idx[docs_b[0]]
        r_de = torch.topk(dense @ qv[qi], k).indices.cpu().numpy()
        fused = rrf([r_bm, r_de], n)
        r_rrf = np.argsort(-fused)[:k]
        for name, r in (("bm25", r_bm), ("dense", r_de), ("rrf", r_rrf)):
            runs[name].append(r)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, ranked in runs.items():
        rec = {}
        for kk in (10, 50, 100):
            hits = [len(set(gold[q["id"]]["relevant_docs"]) & set(doc_list(doc_of, r, kk))) > 0
                    for q, r in zip(queries, ranked)]
            rec[f"doc_hit@{kk}"] = round(float(np.mean(hits)), 4)
        for n_docs in args.n_docs:
            for per_doc in args.per_doc:
                sub = [submission(q["id"], r, doc_of, c["text"], n_docs, per_doc) for q, r in zip(queries, ranked)]
                s = [score_query(p, gold[p["id"]], "ws") for p in sub]
                rec[f"d{n_docs}_c{per_doc}"] = {m: round(float(np.mean([x[m] for x in s])), 4)
                                               for m in ("doc_r", "doc_f2", "chunk_r", "chunk_f2")}
                (out / f"{name}_d{n_docs}_c{per_doc}.json").write_text(json.dumps(sub, ensure_ascii=False),
                                                                    encoding="utf-8")
        report[name] = rec
    by_lang = {}
    for lang in sorted({q["doc_lang"] for q in queries}):
        sel = [i for i, q in enumerate(queries) if q["doc_lang"] == lang]
        by_lang[lang] = {name: round(float(np.mean([
            len(set(gold[queries[i]["id"]]["relevant_docs"]) & set(doc_list(doc_of, runs[name][i], 100))) > 0
            for i in sel])), 4) for name in runs}
    report["doc_hit@100_by_lang"] = by_lang
    report["queries"] = len(queries)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


def doc_list(doc_of: np.ndarray, ranked: np.ndarray, k: int) -> list[int]:
    """First k distinct docs in chunk rank order."""
    seen: dict[int, None] = {}
    for i in ranked:
        seen.setdefault(int(doc_of[i]), None)
        if len(seen) == k:
            break
    return list(seen)


def submission(qid: int, ranked: np.ndarray, doc_of: np.ndarray, texts: list[str], n_docs: int,
               per_doc: int) -> dict:
    docs = doc_list(doc_of, ranked, n_docs)
    keep = set(docs)
    taken: dict[int, int] = {}
    chunks = []
    for i in ranked:
        d = int(doc_of[i])
        if d in keep and taken.get(d, 0) < per_doc:
            taken[d] = taken.get(d, 0) + 1
            chunks.append({"doc_id": d, "chunk_text": texts[i]})
    return {"id": qid, "relevant_docs": docs, "relevant_chunks": chunks}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "run"])
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--queries")
    ap.add_argument("--gold")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=1000, help="chunks kept per branch")
    ap.add_argument("--n-docs", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--per-doc", type=int, nargs="+", default=[1, 3])
    args = ap.parse_args()
    {"encode": cmd_encode, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
