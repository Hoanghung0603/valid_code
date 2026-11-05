"""ZH dense trên tập giao (id ZH kho mới × doc ZH đã embed) + RRF k60 giữa truy vấn VI và ZH.

Chạy trên z87 GPU (bge-m3, cùng cách #5002/embed_all.py: CLS fp16, doc-len 512, query-len 256):
  1. Nạp mọi blk_*.npz {ids, emb} trong --emb, GIỮ các dòng có id ∈ --zh-ids (tập giao).
  2. Embed truy vấn VI (--queries) và truy vấn ZH (--zh-queries, bản dịch; id thiếu/rỗng -> dùng câu VI).
  3. Cosine top-100 mỗi truy vấn trên tập giao, cho cả hai nhánh.
  4. RRF k=60 trọng số 1:1 -> thứ hạng hợp nhất; ghi cache {qid: [[doc_id, idx, score], ...]}
     (idx = -1, vá sau bằng zhnew_report.py trên b660).

Dạng:
  python zhnew_dense.py --emb D:\\r2ai\\work\\zh\\emb --zh-ids new_zh_ids.json --queries queries.jsonl \
      --zh-queries zh_queries.jsonl --out-vi vi.json --out-zh zh.json --out-rrf rrf.json --out-meta meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


def load_zh_ids(path: str) -> list[int]:
    """Danh sách id ZH kho mới: JSON list[int] hoặc {id: ...} hay npz 'ids'."""
    p = Path(path)
    if p.suffix == ".npz":
        return sorted(int(x) for x in np.load(p)["ids"])
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        obj = obj.get("ids", list(obj.keys()))
    return sorted(int(x) for x in obj)


def load_queries(path: str, field: str = "query") -> dict[int, str]:
    """{id: text} từ JSONL theo field; bỏ qua bản ghi text rỗng, bản ghi trùng id: bản sau thắng."""
    out: dict[int, str] = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        text = (r.get(field) or "").strip()
        if not text:
            continue
        out[int(r["id"])] = text
    return out


def fallback_queries(vi: dict[int, str], zh: dict[int, str]) -> tuple[dict[int, str], list[int]]:
    """Truy vấn ZH cho mọi câu VI; câu không có bản dịch ZH -> dùng câu VI. Trả (map, id fallback)."""
    merged = {}
    fell = []
    for qid, q in sorted(vi.items()):
        z = (zh.get(qid) or "").strip()
        if z:
            merged[qid] = z
        else:
            merged[qid] = q
            fell.append(qid)
    return merged, fell


def load_emb_blocks(emb_dir: str, allowed: set[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """(ids, emb) chỉ gồm id ∈ allowed; trả thêm tổng id trong các khối. fp16 giữ nguyên."""
    ids_all, emb_all = [], []
    n_total = 0
    for f in sorted(Path(emb_dir).glob("blk_*.npz")):
        z = np.load(f)
        ids = z["ids"].astype(np.int64)
        n_total += len(ids)
        m = np.fromiter((int(x) in allowed for x in ids), dtype=bool, count=len(ids))
        if m.any():
            ids_all.append(ids[m])
            emb_all.append(z["emb"][m])
    if not ids_all:
        return np.zeros(0, np.int64), np.zeros((0, 1), np.float16), n_total
    return np.concatenate(ids_all), np.concatenate(emb_all), n_total


def rrf_fuse(a_hits: list, b_hits: list, k: int = 60) -> list:
    """RRF k giữa hai thứ hạng [doc_id, idx, score]: fused = 1/(k+rank_a) + 1/(k+rank_b).

    Doc vắng ở nguồn nào thì nguồn đó không đóng góp. Đồng điểm giữ thứ tự xuất hiện (a trước b).
    """
    def ranks(hits: list) -> dict:
        out: dict[int, int] = {}
        for i, h in enumerate(hits):
            out.setdefault(int(h[0]), i + 1)
        return out

    ra, rb = ranks(a_hits), ranks(b_hits)
    rows, seen = [], set()
    for h in list(a_hits) + list(b_hits):
        d = int(h[0])
        if d in seen:
            continue
        seen.add(d)
        sc = 0.0
        if d in ra:
            sc += 1.0 / (k + ra[d])
        if d in rb:
            sc += 1.0 / (k + rb[d])
        rows.append([d, -1, round(sc, 8)])
    rows.sort(key=lambda r: -r[2])
    return rows


def topk_cosine(qe: np.ndarray, ids: np.ndarray, emb: np.ndarray, top: int,
                chunk: int = 131072, log=print) -> dict[int, list]:
    """{qid_index: [[doc_id, -1, cosine], ...]} giảm dần; qe (n_q, d) fp16, emb (n, d) fp16.

    Chấm theo lát GPU như embed_all.merge_topk (GPU nếu torch dùng được, không thì numpy CPU).
    """
    from embed_all import merge_topk
    n, n_q = len(ids), len(qe)
    best_s = np.full((n_q, top), -np.inf, dtype=np.float32)
    best_i = np.full((n_q, top), -1, dtype=np.int64)
    try:
        import torch
        Q = torch.from_numpy(qe).cuda()
        use_gpu = True
    except Exception:
        Q = qe.astype(np.float32)
        use_gpu = False
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        rows = np.arange(s0, s1, dtype=np.int64)
        E = torch.from_numpy(emb[s0:s1]).cuda() if use_gpu else emb[s0:s1].astype(np.float32)
        for q0 in range(0, n_q, 64):
            q1 = min(q0 + 64, n_q)
            if use_gpu:
                sc = (E.float() @ Q[q0:q1].float().T).cpu().numpy()
            else:
                sc = E @ Q[q0:q1].T
            for j in range(q1 - q0):
                merge_topk(best_s, best_i, np.ascontiguousarray(sc[:, j]), rows, q0 + j)
        if use_gpu:
            del E
        log(f"top-k: {s1}/{n} doc")
    out: dict[int, list] = {}
    for q in range(n_q):
        res = []
        for j in np.argsort(-best_s[q], kind="stable"):
            r = int(best_i[q, j])
            if r < 0:
                continue
            res.append([int(ids[r]), -1, round(float(best_s[q, j]), 4)])
        out[q] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="thư mục blk_*.npz {ids, emb} của doc ZH đã embed")
    ap.add_argument("--zh-ids", required=True, help="id ZH kho mới (JSON list hoặc npz)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query} truy vấn VI")
    ap.add_argument("--zh-queries", required=True, help="JSONL {id, query, zh} bản dịch ZH")
    ap.add_argument("--out-vi", required=True)
    ap.add_argument("--out-zh", required=True)
    ap.add_argument("--out-rrf", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--limit-docs", type=int, default=0, help="chỉ giữ N doc giao đầu (thử nghiệm)")
    a = ap.parse_args()
    t0 = time.time()

    allowed_list = load_zh_ids(a.zh_ids)
    allowed = set(allowed_list)
    ids, emb, n_total = load_emb_blocks(a.emb, allowed)
    if a.limit_docs:
        ids, emb = ids[: a.limit_docs], emb[: a.limit_docs]
    print(f"embedding: {n_total} doc trong khối, giao {len(ids)}/{len(allowed)} id ZH mới "
          f"({time.time() - t0:.0f}s)", flush=True)
    if not len(ids):
        raise SystemExit("tập giao rỗng")

    vi = load_queries(a.queries, field="query")
    zh = load_queries(a.zh_queries, field="zh")
    zh_fallback, fell = fallback_queries(vi, zh)
    print(f"truy vấn: VI {len(vi)}, ZH {len(zh_fallback)} (fallback VI cho {len(fell)} id: {fell[:10]})",
          flush=True)

    from embed_run import embed
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype="auto").cuda().eval()
    qids = sorted(vi)
    qe_vi = embed(model, tok, [vi[q] for q in qids], a.query_len, a.batch)
    qe_zh = embed(model, tok, [zh_fallback[q] for q in qids], a.query_len, a.batch)
    print(f"query embedding: VI {qe_vi.shape}, ZH {qe_zh.shape} ({time.time() - t0:.0f}s)", flush=True)

    hits_vi = topk_cosine(qe_vi, ids, emb, a.top, a.chunk, log=lambda m: print(m, flush=True))
    hits_zh = topk_cosine(qe_zh, ids, emb, a.top, a.chunk, log=lambda m: print(m, flush=True))

    out_vi, out_zh, out_rrf = {}, {}, {}
    for i, q in enumerate(qids):
        out_vi[str(q)] = hits_vi[i]
        out_zh[str(q)] = hits_zh[i]
        out_rrf[str(q)] = rrf_fuse(hits_vi[i], hits_zh[i], a.k)
    Path(a.out_vi).write_text(json.dumps(out_vi), encoding="utf-8")
    Path(a.out_zh).write_text(json.dumps(out_zh), encoding="utf-8")
    Path(a.out_rrf).write_text(json.dumps(out_rrf), encoding="utf-8")

    ov = [len({h[0] for h in hits_vi[i][:30]} & {h[0] for h in hits_zh[i][:30]}) for i in range(len(qids))]
    meta = {
        "n_emb_block_ids": n_total,
        "n_zh_new_ids": len(allowed),
        "n_intersection": len(ids),
        "n_queries": len(qids),
        "zh_fallback_ids": fell,
        "top": a.top,
        "rrf_k": a.k,
        "top30_vi_zh_overlap_mean": round(sum(ov) / len(ov), 2) if ov else 0.0,
        "seconds": round(time.time() - t0, 1),
    }
    Path(a.out_meta).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ghi {a.out_vi}, {a.out_zh}, {a.out_rrf}; overlap top-30 VI/ZH TB {meta['top30_vi_zh_overlap_mean']}; "
          f"tổng {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
