"""Chấm lại pool ứng viên bằng đầu ColBERT (multi-vector) + sparse + dense của BAAI/bge-m3.

Công thức khớp FlagEmbedding BGEM3FlagModel
(FlagEmbedding/finetune/embedder/encoder_only/m3/modeling.py + inference/m3.py):
  - dense:   cosine CLS last_hidden_state[:, 0] đã normalize (giống embed_run.embed cũ)
  - colbert: colbert_vecs = normalize(colbert_linear(last_hidden_state[:, 1:]) * attention_mask[:, 1:])
             điểm = sum_i max_j (q_i · d_j) / số token q (bỏ CLS, bỏ padding)
  - sparse:  w(t) = max theo vị trí của relu(sparse_linear(last_hidden_state)) cho token t
             (bỏ cls/eos/pad/unk, bỏ w <= 0); điểm = sum trên token id chung w_q(t) * w_d(t)
  - all:     dense + 0.3 * sparse + colbert (giống ensemble_score của FlagEmbedding)

Mỗi doc trong pool được embed ĐÚNG MỘT LẦN (không lưu vector token doc), rồi chấm cho mọi
câu có doc đó trong pool. Resume theo khối: khối đã có parts/part_XXXXX.json thì bỏ qua.

Dữ liệu vào: pool JSON {qid: [[doc_id, idx, logit], ...]}, doc text từ part JSONL.zst
{id, idx, text}, truy vấn JSONL {id, query}.

Dầu ra --out: {qid: [[doc_id, idx, dense, sparse, colbert], ...]} đủ mọi doc của pool,
thứ tự như pool. Chế độ --export-caches sinh thêm cache_m3_sparse.json / cache_m3_colbert.json /
cache_m3_all.json cùng định dạng các cache khác {qid: [[doc_id, idx, score], ...]} giảm dần.

Ví dụ (z87):
  python scripts/m3_multi_score.py --pool D:\\r2ai\\work\\cms\\cache_pool_rr.json \
      --docs D:\\r2ai\\work\\all\\docs_vi_all.part000.jsonl.zst ... --queries D:\\r2ai\\work\\queries.jsonl \
      --model D:\\r2ai\\hf\\hub\\models--BAAI--bge-m3\\snapshots\\5617a9f61b028005a4858fdac845db406aefb181 \
      --out D:\\r2ai\\work\\cms\\m3_scores.json --parts-dir D:\\r2ai\\work\\cms\\m3_parts \
      --doc-len 512 --query-len 256 --batch 16 --chunk-docs 4096 --status D:\\r2ai\\work\\cms\\m3_full.status
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


# ---------- phần tính điểm (numpy, không cần torch) ----------

def dense_score(q_vec, d_vec) -> float:
    """Cosine 2 vector đã chuẩn hóa = dot product."""
    q = np.asarray(q_vec, dtype=np.float64)
    d = np.asarray(d_vec, dtype=np.float64)
    return float(np.dot(q, d))


def sparse_score(q_w: dict, d_w: dict) -> float:
    """Điểm sparse = tổng trên token id chung w_q(t) * w_d(t)."""
    if len(q_w) > len(d_w):
        q_w, d_w = d_w, q_w
    s = 0.0
    for t, w in q_w.items():
        w2 = d_w.get(t)
        if w2 is not None:
            s += float(w) * float(w2)
    return s


def colbert_score(q_vecs, d_vecs) -> float:
    """ColBERT MaxSim = sum_i max_j (q_i · d_j) / số token q.

    q_vecs [Lq, D], d_vecs [Ld, D] (đã normalize, đã bỏ CLS/padding).
    """
    q = np.asarray(q_vecs, dtype=np.float64)
    d = np.asarray(d_vecs, dtype=np.float64)
    if q.ndim != 2 or d.ndim != 2:
        raise ValueError("q_vecs/d_vecs phải là mảng 2 chiều [L, D]")
    if q.shape[0] == 0 or d.shape[0] == 0:
        raise ValueError("q_vecs/d_vecs không được rỗng")
    if q.shape[1] != d.shape[1]:
        raise ValueError("dim q và d khác nhau")
    sims = q @ d.T
    return float(sims.max(axis=1).sum() / q.shape[0])


def sparse_token_weights(token_ids, weights, unused_tokens=()) -> dict:
    """w(t) = max theo vị trí của weight (đã relu) cho token t; bỏ special token và w <= 0."""
    unused = set(int(x) for x in unused_tokens)
    out: dict = {}
    for idx, w in zip(token_ids, weights):
        idx = int(idx)
        w = float(w)
        if idx in unused or w <= 0.0:
            continue
        if idx not in out or w > out[idx]:
            out[idx] = w
    return out


def ensemble_score(dense, sparse, colbert) -> float:
    """Tổng hợp theo FlagEmbedding ensemble_score: dense + 0.3 * sparse + colbert."""
    return float(dense) + 0.3 * float(sparse) + float(colbert)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Hạng 0-based, xử lý tie bằng trung bình hạng."""
    order = np.argsort(x, kind="stable")
    sx = x[order]
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0
        r[order[i: j + 1]] = avg
        i = j + 1
    return r


def spearman(a, b) -> float:
    """Hệ số tương quan Spearman của 2 danh sách điểm (cùng doc)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a và b phải cùng hình dạng 1 chiều")
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    if den == 0.0:
        return float("nan")
    return float((ra * rb).sum() / den)


def rank_cache(rows: list, pick) -> list:
    """[[doc_id, idx, dense, sparse, colbert], ...] -> [[doc_id, idx, score], ...] giảm dần.

    Điểm bằng nhau giữ thứ tự gốc (sort ổn định).
    """
    out = [[r[0], r[1], round(float(pick(r)), 6)] for r in rows]
    return sorted(out, key=lambda r: -r[2])


# ---------- đọc dữ liệu ----------

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


def load_pool(path, limit_queries: int = 0, top_docs: int = 0) -> tuple:
    """Đọc pool {qid: [[doc_id, idx, logit], ...]} -> (pool, doc_to_qids).

    doc_to_qids: {doc_id: [(qid, vị_trí_trong_pool[q]), ...]}. Lỗi nếu (qid, doc) trùng.
    """
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    qids = list(pool.keys())
    if limit_queries > 0:
        qids = qids[:limit_queries]
    out: dict = {}
    doc_to_qids: dict = {}
    seen: set = set()
    for q in qids:
        rows = pool[q]
        if top_docs > 0:
            rows = rows[:top_docs]
        out[q] = rows
        for pos, row in enumerate(rows):
            doc = int(row[0])
            key = (q, doc)
            if key in seen:
                raise SystemExit(f"pool có cặp (qid, doc) trùng: qid={q} doc={doc}")
            seen.add(key)
            doc_to_qids.setdefault(doc, []).append((q, pos))
    return out, doc_to_qids


# ---------- model + embed (torch) ----------

def load_model(model_dir: str):
    """Load bge-m3 fp16 + colbert_linear/sparse_linear.pt; trả (model, tok, cl, sl, unused_ids)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).cuda().eval()
    linears = {}
    for name in ("colbert_linear", "sparse_linear"):
        p = os.path.join(model_dir, f"{name}.pt")
        sd = torch.load(p, map_location="cpu", weights_only=True)
        w = sd["weight"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias="bias" in sd)
        lin.load_state_dict(sd)
        linears[name] = lin.half().cuda().eval()
        print(f"{name}: weight {tuple(w.shape)} dtype {w.dtype}", flush=True)
    unused = [x for x in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id)
              if x is not None]
    return model, tok, linears["colbert_linear"], linears["sparse_linear"], unused


def encode_texts(model, tok, colbert_linear, sparse_linear, texts, max_len: int, unused_ids) -> list:
    """Embed một batch text -> [(dense [H], colbert [L, Dc], sparse dict), ...] theo thứ tự vào."""
    import torch
    import torch.nn.functional as F

    enc = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
    lens = [len(x) for x in enc]
    m = max(lens)
    ids = torch.full((len(enc), m), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(enc), m), dtype=torch.long)
    for r, x in enumerate(enc):
        ids[r, : lens[r]] = torch.tensor(x, dtype=torch.long)
        att[r, : lens[r]] = 1
    ids = ids.cuda()
    att = att.cuda()
    h = model(input_ids=ids, attention_mask=att).last_hidden_state
    dense = F.normalize(h[:, 0].float(), dim=-1).cpu().numpy()
    c = colbert_linear(h[:, 1:])
    c = F.normalize(c.float() * att[:, 1:].unsqueeze(-1).float(), dim=-1).cpu().numpy()
    sp = torch.relu(sparse_linear(h)).squeeze(-1).float().cpu().numpy()
    out = []
    for r in range(len(enc)):
        nt = lens[r]
        out.append((
            dense[r],
            c[r, : nt - 1],
            sparse_token_weights(enc[r], sp[r, :nt].tolist(), unused_ids),
        ))
    return out


def encode_stream(model, tok, colbert_linear, sparse_linear, items, max_len: int, unused_ids,
                  batch: int, get_text, on_result):
    """Embed từng batch theo thứ tự độ dài (gọi on_result(idx, kết_quả)), tự giảm batch khi OOM."""
    import torch

    order = sorted(range(len(items)), key=lambda i: len(get_text(items[i])))
    i = 0
    while i < len(order):
        idxs = order[i: i + batch]
        texts = [get_text(items[j]) for j in idxs]
        try:
            outs = encode_texts(model, tok, colbert_linear, sparse_linear, texts, max_len, unused_ids)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(idxs) == 1:
                raise
            torch.cuda.empty_cache()
            batch = max(1, len(idxs) // 2)
            print(f"OOM -> batch {batch}", flush=True)
            continue
        for j, o in zip(idxs, outs):
            on_result(j, o)
        del outs
        i += len(idxs)


# ---------- gộp kết quả + xuất cache ----------

def merge_and_write(pool: dict, parts_dir, out_path) -> int:
    """Gộp parts/part_*.json -> {qid: [[doc_id, idx, dense, sparse, colbert], ...]} theo thứ tự pool.

    Trả về số dòng; báo lỗi nếu thiếu điểm hoặc có NaN/inf.
    """
    pos: dict = {}
    res: dict = {}
    for q, rows in pool.items():
        res[q] = [[int(r[0]), int(r[1]), None, None, None] for r in rows]
        for i, r in enumerate(rows):
            pos[(q, int(r[0]))] = i
    n_part = 0
    for pf in sorted(Path(parts_dir).glob("part_*.json")):
        n_part += 1
        data = json.loads(pf.read_text(encoding="utf-8"))
        for doc_s, entries in data["scores"].items():
            doc = int(doc_s)
            for q, dense, sparse, colbert in entries:
                i = pos.get((q, doc))
                if i is None:
                    raise SystemExit(f"part {pf.name} có doc {doc} không thuộc pool của qid {q}")
                res[q][i][2] = dense
                res[q][i][3] = sparse
                res[q][i][4] = colbert
    bad = 0
    n = 0
    for q, rows in res.items():
        for r in rows:
            n += 1
            for v in r[2:]:
                if v is None or not math.isfinite(v):
                    bad += 1
    if bad:
        raise SystemExit(f"{bad} điểm None/NaN/inf trong kết quả ({n_part} part)")
    Path(out_path).write_text(json.dumps(res), encoding="utf-8")
    print(f"ghi {out_path}: {len(res)} câu x {n} dòng ({n_part} part)", flush=True)
    return n


def export_caches(scores_path, out_dir) -> list:
    """Xuất 3 cache xếp hạng giảm dần từ m3_scores.json; kiểm tra NaN.

    all = dense + 0.3 * sparse + colbert (trọng số FlagEmbedding ensemble_score).
    """
    scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    kinds = {
        "cache_m3_sparse.json": lambda r: r[3],
        "cache_m3_colbert.json": lambda r: r[4],
        "cache_m3_all.json": lambda r: ensemble_score(r[2], r[3], r[4]),
    }
    out_paths = []
    for name, pick in kinds.items():
        out = {}
        for q, rows in scores.items():
            for r in rows:
                for v in r[2:]:
                    if v is None or not math.isfinite(v):
                        raise SystemExit(f"điểm None/NaN/inf ở qid {q} doc {r[0]}")
            out[q] = rank_cache(rows, pick)
        p = Path(out_dir) / name
        p.write_text(json.dumps(out), encoding="utf-8")
        out_paths.append(p)
        print(f"ghi {p}: {len(out)} câu", flush=True)
    return out_paths


def bench_compare(scores: dict, old_path) -> tuple:
    """So dense mới vs dense cũ: (danh_sách_spearman, danh_sách_trùng_top30) trên doc chung."""
    old = json.loads(Path(old_path).read_text(encoding="utf-8"))
    sps, ovs = [], []
    for q, rows in scores.items():
        old_rows = old.get(q)
        if not old_rows:
            continue
        old_map = {int(r[0]): float(r[2]) for r in old_rows}
        new_map = {int(r[0]): float(r[2]) for r in rows}
        common = [d for d in new_map if d in old_map]
        if len(common) < 2:
            continue
        sps.append(spearman([old_map[d] for d in common], [new_map[d] for d in common]))
        top_old = sorted(common, key=lambda d: -old_map[d])[:30]
        top_new = sorted(common, key=lambda d: -new_map[d])[:30]
        ovs.append(len(set(top_old) & set(top_new)))
    return sps, ovs


def write_status(path, obj) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(obj), encoding="utf-8")


# ---------- chạy chính ----------

def run(args) -> None:
    t0 = time.time()
    pool, doc_to_qids = load_pool(args.pool, args.limit_queries, args.top_docs)
    n_pairs = sum(len(v) for v in pool.values())
    print(f"pool: {len(pool)} câu, {n_pairs} cặp (q,d), {len(doc_to_qids)} doc duy nhất", flush=True)

    parts_dir = Path(args.parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    qtext: dict = {}
    for rec in Path(args.queries).read_text(encoding="utf-8").splitlines():
        if rec.strip():
            r = json.loads(rec)
            qtext[str(r["id"])] = r["query"]
    missing_q = [q for q in pool if q not in qtext]
    if missing_q:
        raise SystemExit(f"{len(missing_q)} qid của pool thiếu trong queries.jsonl (vd {missing_q[:5]})")

    model, tok, cl, sl, unused = load_model(args.model)
    import torch

    q_dense: dict = {}
    q_colbert: dict = {}
    q_sparse: dict = {}
    qids = list(pool.keys())

    def on_query(j, o):
        q_dense[qids[j]], q_colbert[qids[j]], q_sparse[qids[j]] = o

    with torch.inference_mode():
        encode_stream(model, tok, cl, sl, qids, args.query_len, unused, args.batch,
                      lambda q: qtext[q], on_query)
        print(f"embed {len(qids)} truy vấn ({time.time() - t0:.0f}s)", flush=True)

        need = set(doc_to_qids)
        n_done_parts = 0
        for pf in sorted(parts_dir.glob("part_*.json")):
            data = json.loads(pf.read_text(encoding="utf-8"))
            n_done_parts += len(data["scores"])
        print(f"resume: {len(list(parts_dir.glob('part_*.json')))} part đã có, "
              f"{n_done_parts} doc đã chấm", flush=True)

        n_scored = 0
        t_score = time.time()
        window: list = []
        window_ci = 0
        seen: set = set()
        max_alloc = 0

        def flush_chunk(ci: int, items: list):
            nonlocal n_scored, max_alloc
            scores: dict = {}
            for doc, dense, colbert, sparse in items:
                rows = []
                for q, _pos in doc_to_qids[doc]:
                    rows.append([q,
                                 round(dense_score(q_dense[q], dense), 6),
                                 round(sparse_score(q_sparse[q], sparse), 6),
                                 round(colbert_score(q_colbert[q], colbert), 6)])
                scores[str(doc)] = rows
            final = parts_dir / f"part_{ci:05d}.json"
            tmp = parts_dir / f"part_{ci:05d}.json.tmp"
            tmp.write_text(json.dumps({"scores": scores}), encoding="utf-8")
            os.replace(tmp, final)
            n_scored += len(items)
            max_alloc = max(max_alloc, torch.cuda.max_memory_allocated())
            dt = time.time() - t_score
            rate = n_scored / dt if dt > 0 else 0.0
            left = (len(need) - n_done_parts - n_scored) / rate if rate > 0 else 0.0
            print(f"chunk {ci}: +{len(items)} doc, tổng mới {n_scored} "
                  f"({rate:.1f} doc/s, còn ~{left / 60:.0f} phút, VRAM {max_alloc / 1e9:.2f}GB)", flush=True)
            write_status(args.status, {
                "phase": "score", "done_docs": n_done_parts + n_scored, "total_docs": len(need),
                "docs_per_s": round(rate, 2), "elapsed_s": round(time.time() - t0, 1),
                "vram_gb": round(max_alloc / 1e9, 2),
            })

        def process_window(ci: int):
            pending = []

            def on_doc(j, o):
                pending.append((window[j][0],) + tuple(o))

            encode_stream(model, tok, cl, sl, window, args.doc_len, unused, args.batch,
                          lambda it: it[1], on_doc)
            flush_chunk(ci, pending)

        # duyệt doc theo thứ tự gặp trong part files; khối c = nhóm chunk_docs doc liên tiếp
        for p in sorted({str(x) for x in args.docs}):
            for rec in read_jsonl_zst(p):
                doc = int(rec["id"])
                if doc not in need or doc in seen:
                    continue
                seen.add(doc)
                ci = (len(seen) - 1) // args.chunk_docs
                if (parts_dir / f"part_{ci:05d}.json").exists():
                    continue
                if not window:
                    window_ci = ci
                window.append((doc, str(rec.get("text") or "")))
                if len(window) >= args.chunk_docs:
                    process_window(window_ci)
                    window = []
        if window:
            process_window(window_ci)

        if len(seen) != len(need):
            raise SystemExit(f"thiếu {len(need) - len(seen)} doc trong part files "
                             f"(gặp {len(seen)}/{len(need)})")

    merge_and_write(pool, parts_dir, args.out)
    if args.bench_vs:
        scores = json.loads(Path(args.out).read_text(encoding="utf-8"))
        sps, ovs = bench_compare(scores, args.bench_vs)
        if sps:
            print(f"BENCH dense moi vs cu: spearman mean {sum(sps) / len(sps):.4f} "
                  f"min {min(sps):.4f} ({len(sps)} câu); "
                  f"trung top30 {sum(ovs) / len(ovs):.1f}/30 min {min(ovs)}", flush=True)
        else:
            print("BENCH: khong co doc chung nao de so sanh", flush=True)
    print(f"xong {len(pool)} câu, {n_scored} doc chấm lại, tổng {time.time() - t0:.0f}s", flush=True)
    write_status(args.status, {
        "phase": "done", "done_docs": len(need), "total_docs": len(need),
        "elapsed_s": round(time.time() - t0, 1),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="JSON {qid: [[doc_id, idx, logit], ...]}")
    ap.add_argument("--docs", nargs="*", default=[], help="part JSONL.zst {id, idx, text}")
    ap.add_argument("--queries", help="JSONL {id, query}")
    ap.add_argument("--model", help="thư mục model bge-m3 (snapshot)")
    ap.add_argument("--out", help="JSON kết quả m3_scores.json")
    ap.add_argument("--parts-dir", help="thư mục parts resume")
    ap.add_argument("--doc-len", type=int, default=512)
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk-docs", type=int, default=4096, help="số doc mỗi khối resume")
    ap.add_argument("--limit-queries", type=int, default=0, help="chỉ chạy N câu đầu (bench)")
    ap.add_argument("--top-docs", type=int, default=0, help="chỉ chấm top N doc đầu mỗi câu")
    ap.add_argument("--bench-vs", default=None, help="dense cũ JSON để sanity dense mới")
    ap.add_argument("--status", default=None, help="ghi trạng thái 1 dòng JSON")
    ap.add_argument("--export-caches", action="store_true", help="chỉ xuất 3 cache từ --scores")
    ap.add_argument("--scores", default=None, help="m3_scores.json cho --export-caches")
    ap.add_argument("--out-dir", default=None, help="thư mục ghi cache khi --export-caches")
    a = ap.parse_args()
    if a.export_caches:
        if not a.scores or not a.out_dir:
            ap.error("--export-caches cần --scores và --out-dir")
        export_caches(a.scores, a.out_dir)
        return
    for flag in ("pool", "docs", "queries", "model", "out", "parts_dir"):
        if not getattr(a, flag):
            ap.error(f"cần --{flag.replace('_', '-')}")
    run(a)


if __name__ == "__main__":
    main()
