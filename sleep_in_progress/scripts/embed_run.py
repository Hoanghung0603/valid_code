"""Embed doc + truy vấn bằng bge-m3 (dense, CLS, fp16) trên GPU rồi tính cosine cho các cặp (câu, doc ứng viên).

Chạy lại được: embedding doc lưu theo khối trong --work (blk_00000.npz ...); khối đã có thì bỏ qua.
Chỉ đặt tải GPU; CPU chỉ để tokenize. Đầu ra: --out JSON {qid: [[doc_id, idx, cosine], ...]} (cùng thứ tự ứng viên).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import zstandard
from transformers import AutoModel, AutoTokenizer

BLOCK = 4096


def read_docs(path: str):
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


@torch.inference_mode()
def embed(model, tok, texts: list[str], max_len: int, batch: int) -> np.ndarray:
    enc = tok(texts, truncation=True, max_length=max_len)["input_ids"]
    lens = [len(x) for x in enc]
    order = sorted(range(len(enc)), key=lambda i: lens[i])
    out = np.zeros((len(enc), model.config.hidden_size), dtype=np.float16)
    pad = tok.pad_token_id
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
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--doc-len", type=int, default=512)
    ap.add_argument("--query-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="chỉ embed N doc đầu (thử nghiệm)")
    a = ap.parse_args()

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    t0 = time.time()

    ids_all: list[int] = []
    block, bi, n_done = [], 0, 0

    def flush():
        nonlocal block, bi, n_done
        if not block:
            return
        f = work / f"blk_{bi:05d}.npz"
        ids_all.extend(r["id"] for r in block)
        if not f.exists():
            e = embed(model, tok, [r["text"] for r in block], a.doc_len, a.batch)
            np.savez(f, ids=np.asarray([r["id"] for r in block], dtype=np.int64), emb=e)
        n_done += len(block)
        el = time.time() - t0
        print(f"khối {bi}: {n_done} doc, {el:.0f}s, {n_done / max(el, 1):.1f} doc/s", flush=True)
        block, bi = [], bi + 1

    for r in read_docs(a.docs):
        block.append(r)
        if len(block) >= BLOCK:
            flush()
        if a.limit and n_done + len(block) >= a.limit:
            break
    flush()

    embs, ids = [], []
    for f in sorted(work.glob("blk_*.npz")):
        z = np.load(f)
        ids.append(z["ids"])
        embs.append(z["emb"])
    ids = np.concatenate(ids)
    embs = np.concatenate(embs)
    row = {int(d): k for k, d in enumerate(ids)}
    print(f"doc embedding: {embs.shape}", flush=True)

    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    qe = embed(model, tok, [q["query"] for q in qs], a.query_len, 8)
    qrow = {str(q["id"]): k for k, q in enumerate(qs)}
    print(f"query embedding: {qe.shape} ({time.time() - t0:.0f}s)", flush=True)

    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    E = torch.from_numpy(embs).cuda()
    out, miss = {}, 0
    for qid, lst in cands.items():
        if qid not in qrow:
            continue
        rows = [row.get(d, -1) for d, _ in lst]
        ok = [k for k, r in enumerate(rows) if r >= 0]
        miss += len(rows) - len(ok)
        q = torch.from_numpy(qe[qrow[qid]]).cuda()
        sc = (E[[rows[k] for k in ok]].float() @ q.float()).cpu().numpy()
        res = [[lst[k][0], lst[k][1], None] for k in range(len(lst))]
        for k, s in zip(ok, sc):
            res[k][2] = round(float(s), 4)
        out[qid] = res
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"ghi {a.out}: {len(out)} câu, cặp thiếu embedding {miss}; tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
