"""Embed TOÀN BỘ doc VI bằng bge-m3 (dense, CLS, fp16) trên GPU rồi lấy top-200 cosine mỗi truy vấn.

Đọc các part JSONL.zst {"id","idx","text"} từ --docs; id đã có embedding trong --work-old (blk_*.npz cũ)
được TÁI DÙNG, phần còn lại embed vào --work/blk_NNNNN.npz (khối 4096, resume: khối đã ghi thì bỏ qua).
Sau đó nạp mọi embedding (cũ + mới, ~613k x 1024 fp16), embed truy vấn (max_len 256) và tìm top-200
toàn kho theo lát GPU. Đầu ra --out: {qid: [[doc_id, idx, cosine], ...]} giảm dần theo cosine; idx là
chỉ số doc trong chỉ mục VI lấy từ bản ghi xuất.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np


def load_done(dirs: list[Path]) -> set[int]:
    done: set[int] = set()
    for d in dirs:
        for f in sorted(Path(d).glob("blk_*.npz")):
            done.update(int(x) for x in np.load(f)["ids"])
    return done


def next_block_index(work: Path) -> int:
    bis = [int(f.stem.split("_")[1]) for f in work.glob("blk_*.npz")]
    return max(bis, default=-1) + 1


def iter_docs(paths: list[Path], read_docs):
    for p in paths:
        yield from read_docs(str(p))


def merge_topk(best_s: np.ndarray, best_i: np.ndarray, col: np.ndarray, rows: np.ndarray, q: int) -> None:
    """Gộp điểm một lát doc (col = điểm các hàng rows) vào top đang giữ của truy vấn q."""
    top = best_s.shape[1]
    k = min(top, len(col))
    t = np.argpartition(col, -k)[-k:]
    cs = np.concatenate([best_s[q], col[t]])
    ci = np.concatenate([best_i[q], rows[t]])
    sel = np.argsort(-cs, kind="stable")[:top]
    best_s[q], best_i[q] = cs[sel], ci[sel]


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from embed_run import BLOCK, embed, read_docs
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, help="thư mục chứa *.jsonl.zst {id, idx, text}")
    ap.add_argument("--docs-old", help="JSONL.zst cũ {id, idx, text} (dự phòng cho idx)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--work-old", required=True, help="thư mục blk_*.npz cũ (chỉ đọc, tái dùng)")
    ap.add_argument("--work", required=True, help="thư mục blk_*.npz mới (resume)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--doc-len", type=int, default=512)
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=131072, help="số doc mỗi lát khi tìm top-k trên GPU")
    ap.add_argument("--limit", type=int, default=0, help="chỉ embed N doc mới đầu (thử nghiệm)")
    a = ap.parse_args()

    work, work_old = Path(a.work), Path(a.work_old)
    work.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.docs).glob("*.jsonl.zst"))
    if not paths:
        raise SystemExit(f"không thấy part nào trong {a.docs}")

    t0 = time.time()
    done = load_done([work_old, work])
    bi = next_block_index(work)
    print(f"{len(paths)} part, {len(done)} doc đã có embedding, khối mới từ {bi}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()

    idx_map: dict[int, int] = {}
    n_new = n_skip = n_flush = 0
    block: list[dict] = []
    seen = set(done)

    def flush():
        nonlocal block, bi, n_flush
        if not block:
            return
        f = work / f"blk_{bi:05d}.npz"
        e = embed(model, tok, [r["text"] for r in block], a.doc_len, a.batch)
        np.savez(f, ids=np.asarray([r["id"] for r in block], dtype=np.int64), emb=e)
        n_flush += len(block)
        el = time.time() - t0
        print(f"khối {bi}: {n_new} doc mới, {n_skip} tái dùng, {el:.0f}s, {n_flush / max(el, 1):.1f} doc/s", flush=True)
        block, bi = [], bi + 1

    for r in iter_docs(paths, read_docs):
        did = int(r["id"])
        idx_map[did] = int(r["idx"])
        if did in seen:
            n_skip += 1
            continue
        seen.add(did)
        block.append(r)
        n_new += 1
        if len(block) >= BLOCK:
            flush()
        if a.limit and n_new >= a.limit:
            break
    flush()
    if a.docs_old:
        for r in read_docs(a.docs_old):
            idx_map.setdefault(int(r["id"]), int(r["idx"]))
    print(f"embed xong: {n_new} doc mới, {n_skip} tái dùng từ {work_old.name}, idx_map {len(idx_map)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    embs, ids = [], []
    for f in sorted(work_old.glob("blk_*.npz")) + sorted(work.glob("blk_*.npz")):
        z = np.load(f)
        ids.append(z["ids"])
        embs.append(z["emb"])
    ids = np.concatenate(ids)
    embs = np.concatenate(embs)
    n = len(ids)
    print(f"doc embedding: {embs.shape}", flush=True)

    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    qe = embed(model, tok, [q["query"] for q in qs], a.query_len, 8)
    print(f"query embedding: {qe.shape} ({time.time() - t0:.0f}s)", flush=True)

    top = a.top
    best_s = np.full((len(qs), top), -np.inf, dtype=np.float32)
    best_i = np.full((len(qs), top), -1, dtype=np.int64)
    Q = torch.from_numpy(qe).cuda()
    for s0 in range(0, n, a.chunk):
        s1 = min(s0 + a.chunk, n)
        rows = np.arange(s0, s1, dtype=np.int64)
        E = torch.from_numpy(embs[s0:s1]).cuda()
        for q0 in range(0, len(qs), 64):
            q1 = min(q0 + 64, len(qs))
            sc = (E.float() @ Q[q0:q1].float().T).cpu().numpy()
            for j in range(q1 - q0):
                merge_topk(best_s, best_i, sc[:, j], rows, q0 + j)
        del E
        print(f"top-k: {s1}/{n} doc ({time.time() - t0:.0f}s)", flush=True)

    out = {}
    for k, q in enumerate(qs):
        res = []
        for j in np.argsort(-best_s[k], kind="stable"):
            r = int(best_i[k, j])
            if r < 0:
                continue
            d = int(ids[r])
            res.append([d, idx_map[d], round(float(best_s[k, j]), 4)])
        out[str(q["id"])] = res
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"ghi {a.out}: {len(out)} câu x {top}; tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
