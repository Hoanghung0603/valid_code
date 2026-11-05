"""Tìm top-k cosine cho truy vấn trên TOÀN KHO doc từ embedding ĐÃ CÓ (blk_*.npz), KHÔNG embed lại doc.

Đọc embedding doc trong các thư mục --emb (blk_*.npz của embed_run.py/embed_all.py — chỉ đọc, không ghi),
nhúng truy vấn bằng bge-m3 trên GPU (--model) hoặc đọc --query-emb (.npz emb, hàng theo --queries),
rồi tìm top-k cosine toàn kho theo lát --chunk (matmul GPU cuda hoặc CPU numpy).
idx của doc lấy từ --docs (JSONL.zst {id, idx, text}) hoặc --idx-from (JSON {id: idx}).

Đầu ra --out: {qid: [[doc_id, idx, cosine], ...]} giảm dần theo cosine — cùng định dạng embed_all.py,
dùng được cho fuse.py/rerank_run.py.

Ví dụ (z87):
  python scripts/dense_topk.py --emb D:\\r2ai\\work\\emb D:\\r2ai\\work\\emb_all --docs part*.jsonl.zst \
      --queries D:\\r2ai\\work\\queries.jsonl --out dense_full_top400.json --top 400 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from embed_all import merge_topk


def read_jsonl_zst(path):
    """Đọc JSONL.zst {id, idx, text} từng bản ghi (zstandard nạp lười)."""
    import zstandard
    with open(path, "rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as r:
            buf = b""
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if ln:
                        yield json.loads(ln)
            if buf.strip():
                yield json.loads(buf)


def load_embeddings(dirs) -> tuple:
    """Nối blk_*.npz trong các thư mục -> (ids int64[n], emb float16[n, d]); id trùng báo lỗi."""
    ids, embs = [], []
    for d in dirs:
        for f in sorted(Path(d).glob("blk_*.npz")):
            z = np.load(f)
            ids.append(np.asarray(z["ids"], dtype=np.int64))
            embs.append(np.asarray(z["emb"], dtype=np.float16))
    if not ids:
        raise SystemExit(f"không thấy blk_*.npz trong {dirs}")
    ids = np.concatenate(ids)
    embs = np.concatenate(embs)
    if len(np.unique(ids)) != len(ids):
        raise SystemExit(f"id doc trùng giữa các blk_*.npz ({len(ids)} id)")
    return ids, embs


def load_idx_map(docs=(), idx_from=()) -> dict:
    """{doc_id: idx} từ JSONL.zst {id, idx, text} (--docs) và/hoặc JSON {id: idx} (--idx-from)."""
    out: dict = {}
    for p in docs:
        for r in read_jsonl_zst(p):
            out[int(r["id"])] = int(r["idx"])
    for p in idx_from:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
        for k, v in obj.items():
            out[int(k)] = int(v)
    return out


def topk_scan(embs: np.ndarray, qe: np.ndarray, top: int, chunk: int = 131072,
              score_fn=None, progress=None) -> tuple:
    """Top-k cosine mọi truy vấn, duyệt doc theo lát --chunk; score_fn(E, Q) -> [n, nq] (mặc định numpy).

    Trả (best_s, best_i) [nq, min(top, n)] giảm dần theo cosine; best_i là hàng trong embs.
    """
    n, nq = embs.shape[0], qe.shape[0]
    top = min(top, n)
    best_s = np.full((nq, top), -np.inf, dtype=np.float32)
    best_i = np.full((nq, top), -1, dtype=np.int64)
    fn = score_fn or (lambda e, q: e.astype(np.float32) @ q.astype(np.float32).T)
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        rows = np.arange(s0, s1, dtype=np.int64)
        sc = np.asarray(fn(embs[s0:s1], qe), dtype=np.float32)
        for j in range(nq):
            merge_topk(best_s, best_i, sc[:, j], rows, j)
        if progress:
            progress(s1, n)
    return best_s, best_i


def format_hits(best_s: np.ndarray, best_i: np.ndarray, ids: np.ndarray, idx_map: dict,
                qids: list) -> dict:
    """{qid: [[doc_id, idx, cosine], ...]} giảm dần cosine; doc thiếu idx trong idx_map báo lỗi."""
    out: dict = {}
    missing: set = set()
    for k, qid in enumerate(qids):
        res = []
        for j in np.argsort(-best_s[k], kind="stable"):
            r = int(best_i[k, j])
            if r < 0:
                continue
            d = int(ids[r])
            i = idx_map.get(d)
            if i is None:
                missing.add(d)
                continue
            res.append([d, i, round(float(best_s[k, j]), 4)])
        out[str(qid)] = res
    if missing:
        raise SystemExit(f"{len(missing)} doc top-k thiếu idx (vd {sorted(missing)[:5]})")
    return out


def embed_queries(model_name: str, texts: list, max_len: int, batch: int) -> np.ndarray:
    """Nhúng truy vấn bằng bge-m3 trên GPU (CLS, fp16, đã chuẩn hóa) — cùng embed_run.embed."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    enc = tok(texts, truncation=True, max_length=max_len)["input_ids"]
    lens = [len(x) for x in enc]
    order = sorted(range(len(enc)), key=lambda i: lens[i])
    out = np.zeros((len(enc), model.config.hidden_size), dtype=np.float16)
    pad = tok.pad_token_id
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            b = order[s: s + batch]
            m = max(lens[i] for i in b)
            ids = torch.full((len(b), m), pad, dtype=torch.long)
            att = torch.zeros((len(b), m), dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, : lens[i]] = torch.tensor(enc[i])
                att[r, : lens[i]] = 1
            h = model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h.float(), dim=-1)
            out[b] = h.half().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, nargs="+", help="thư mục blk_*.npz ĐÃ CÓ (chỉ đọc)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", nargs="*", default=[], help="JSONL.zst {id, idx, text} để lấy idx")
    ap.add_argument("--idx-from", nargs="*", default=[], help="JSON {id: idx} để lấy idx")
    ap.add_argument("--query-emb", default=None, help=".npz emb truy vấn (hàng theo --queries), bỏ nhúng model")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8, help="batch nhúng truy vấn")
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=131072, help="số doc mỗi lát khi quét top-k")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="matmul top-k; cpu cần --query-emb")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N câu đầu (thử)")
    a = ap.parse_args()
    if not a.docs and not a.idx_from:
        ap.error("cần --docs hoặc --idx-from để lấy idx doc")
    if a.device == "cpu" and not a.query_emb:
        ap.error("--device cpu cần --query-emb có sẵn (không nhúng model trên CPU)")

    t0 = time.time()
    ids, embs = load_embeddings(a.emb)
    idx_map = load_idx_map(a.docs, a.idx_from)
    print(f"doc embedding: {embs.shape}, idx_map {len(idx_map)} ({time.time() - t0:.0f}s)", flush=True)

    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit > 0:
        qs = qs[: a.limit]
    if a.query_emb:
        qe = np.asarray(np.load(a.query_emb)["emb"], dtype=np.float16)
    else:
        qe = embed_queries(a.model, [q["query"] for q in qs], a.query_len, a.batch)
    if qe.shape[0] != len(qs):
        raise SystemExit(f"emb truy vấn {qe.shape[0]} hàng != {len(qs)} câu")
    print(f"query embedding: {qe.shape} ({time.time() - t0:.0f}s)", flush=True)

    score_fn = None
    if a.device == "cuda":
        import torch
        Q = torch.from_numpy(qe).cuda()

        def score_fn(e, q):
            E = torch.from_numpy(np.ascontiguousarray(e)).cuda()
            sc = (E.float() @ Q.float().T).cpu().numpy()
            del E
            return sc

    def on_slice(s1, n):
        print(f"top-k: {s1}/{n} doc ({time.time() - t0:.0f}s)", flush=True)

    best_s, best_i = topk_scan(embs, qe, a.top, a.chunk, score_fn, progress=on_slice)
    out = format_hits(best_s, best_i, ids, idx_map, [q["id"] for q in qs])
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"ghi {a.out}: {len(out)} câu x {best_s.shape[1]}; tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
