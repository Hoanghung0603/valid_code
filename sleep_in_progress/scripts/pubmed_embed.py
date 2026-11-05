"""Embed doc PubMed + truy vấn VI/EN bằng bge-m3 (dense, CLS, fp16) trên GPU, tìm top-100 mỗi câu.

Tái dùng embed_run.read_docs/embed, dense_topk.load_embeddings/topk_scan/format_hits, fuse.rrf_fuse_multi.
- Doc: text = title + "\\n" + abstract (đã xuất bởi pubmed_docs.py, {id, idx, text}); resume theo khối blk_*.npz.
- (a) top-100 theo truy vấn VI gốc, (b) top-100 theo truy vấn EN, (c) RRF k60 của (a)+(b).
- Cả 3 cache: {qid: [[doc_id, idx, score], ...]} giảm dần theo score - cùng định dạng search_cache.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from dense_topk import format_hits, load_embeddings, topk_scan
from embed_all import merge_topk  # noqa: F401 - dense_topk import sẵn, giữ tường minh
from embed_run import BLOCK, embed, read_docs
from fuse import rrf_fuse_multi


def load_queries(path: str, field: str) -> tuple:
    """JSONL -> (qids, texts); dòng thiếu field/field rỗng bị bỏ qua (vd câu chưa dịch được)."""
    qids, texts = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = r.get(field) or r.get("query") or ""
        if not str(t).strip():
            continue
        qids.append(r["id"])
        texts.append(str(t))
    return qids, texts


def embed_docs(docs, work: Path, model, tok, doc_len: int, batch: int) -> dict:
    """Embed mọi doc vào work/blk_*.npz (khối đã ghi thì bỏ qua); trả {id: idx}."""
    work.mkdir(parents=True, exist_ok=True)
    done: set = set()
    for f in sorted(work.glob("blk_*.npz")):
        done.update(int(x) for x in np.load(f)["ids"])
    bis = [int(f.stem.split("_")[1]) for f in work.glob("blk_*.npz")]
    bi = max(bis, default=-1) + 1
    idx_map: dict = {}
    block, n_new, n_skip, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal block, bi
        if not block:
            return
        f = work / f"blk_{bi:05d}.npz"
        e = embed(model, tok, [r["text"] for r in block], doc_len, batch)
        np.savez(f, ids=np.asarray([r["id"] for r in block], dtype=np.int64), emb=e)
        print(f"khối {bi}: {len(block)} doc, {time.time() - t0:.0f}s", flush=True)
        block, bi = [], bi + 1

    for r in docs:
        idx_map[int(r["id"])] = int(r["idx"])
        if int(r["id"]) in done:
            n_skip += 1
            continue
        done.add(int(r["id"]))
        block.append(r)
        n_new += 1
        if len(block) >= BLOCK:
            flush()
    flush()
    print(f"embed doc: {n_new} mới, {n_skip} tái dùng, idx_map {len(idx_map)} ({time.time() - t0:.0f}s)",
          flush=True)
    return idx_map


def topk_for(embs_ids, embs, qids, texts, model, tok, top, chunk, query_len, batch, idx_map):
    """top-k cosine cho một tập truy vấn -> {qid: [[doc_id, idx, cosine], ...]}."""
    import torch
    ids = embs_ids
    qe = embed(model, tok, texts, query_len, batch)
    Q = torch.from_numpy(qe).cuda()

    def score_fn(e, q):
        E = torch.from_numpy(np.ascontiguousarray(e)).cuda()
        sc = (E.float() @ Q.float().T).cpu().numpy()
        del E
        return sc

    t0 = time.time()
    best_s, best_i = topk_scan(embs, qe, top, chunk, score_fn)
    out = format_hits(best_s, best_i, ids, idx_map, qids)
    print(f"top-{top}: {len(qids)} truy vấn ({time.time() - t0:.0f}s)", flush=True)
    return out


def expand_docs(paths: list) -> list:
    """Nhận file hoặc thư mục (thư mục -> *.jsonl.zst bên trong), trả danh sách file đã sắp."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.glob("*.jsonl.zst")))
        else:
            out.append(p)
    if not out:
        raise SystemExit(f"không thấy file docs nào trong {paths}")
    return out


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, nargs="+", help="docs_pm.part*.jsonl.zst {id, idx, text}")
    ap.add_argument("--queries-vi", required=True, help="JSONL {id, query}")
    ap.add_argument("--queries-en", required=True, help="JSONL {id, query, en}")
    ap.add_argument("--work", required=True, help="thư mục blk_*.npz (resume)")
    ap.add_argument("--out-vi", required=True)
    ap.add_argument("--out-en", required=True)
    ap.add_argument("--out-rrf", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--doc-len", type=int, default=512)
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--k", type=int, default=60, help="RRF k")
    ap.add_argument("--chunk", type=int, default=131072)
    a = ap.parse_args()
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    doc_paths = expand_docs(a.docs)
    print(f"{len(doc_paths)} file docs", flush=True)
    idx_map = embed_docs((r for p in doc_paths for r in read_docs(p)),
                         Path(a.work), model, tok, a.doc_len, a.batch)

    ids, embs = load_embeddings([a.work])
    print(f"doc embedding: {embs.shape}", flush=True)

    vi_ids, vi_texts = load_queries(a.queries_vi, "query")
    en_ids, en_texts = load_queries(a.queries_en, "en")
    out_vi = topk_for(ids, embs, vi_ids, vi_texts, model, tok, a.top, a.chunk, a.query_len, 8, idx_map)
    out_en = topk_for(ids, embs, en_ids, en_texts, model, tok, a.top, a.chunk, a.query_len, 8, idx_map)

    out_rrf = {}
    for q in vi_ids:
        q = str(q)
        fused = rrf_fuse_multi([(out_vi.get(q, []), 1.0), (out_en.get(q, []), 1.0)], a.k)
        out_rrf[q] = fused[: a.top]
    for p, obj in ((a.out_vi, out_vi), (a.out_en, out_en), (a.out_rrf, out_rrf)):
        Path(p).write_text(json.dumps(obj), encoding="utf-8")
        print(f"ghi {p}: {len(obj)} câu", flush=True)
    print(f"tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
