"""Rerank ứng viên BM25 bằng cross-encoder bge-reranker-v2-m3 (fp16) trên GPU rồi ghi điểm logit.

Chạy lại được: kết quả từng phần (PART_SIZE câu) lưu JSON trong --resume-dir; phần đã có thì bỏ qua.
Chỉ đặt tải GPU; CPU chỉ để tokenize. Đầu ra: --out JSON {qid: [[doc_id, idx, logit], ...]} (cùng thứ tự
ứng viên BM25 như dense_top200.json của embed_run.py, dùng được cho fuse.py). Logit thô (chưa sigmoid).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PART_SIZE = 50  # số câu mỗi phần khi lưu resume
LOG_EVERY = 25  # in tiến độ mỗi N batch


def read_docs(path: str):
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


def load_docs(paths, doc_chars: int = 0, wanted: set | None = None) -> dict:
    """Đọc một hoặc nhiều file docs -> {id: text}; doc_chars > 0 thì cắt text; wanted chỉ giữ id trong tập đó."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    docs = {}
    for p in paths:
        for r in read_docs(p):
            i = r["id"]
            if wanted is not None and i not in wanted:
                continue
            t = r["text"]
            if doc_chars > 0:
                t = t[: doc_chars]
            docs[i] = t
    return docs


def split_parts(items: list, size: int) -> list[list]:
    """Chia danh sách thành các phần <= size (phần cuối có thể ngắn hơn)."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def part_file(resume_dir: Path, pi: int) -> Path:
    return resume_dir / f"part_{pi:05d}.json"


def plan_parts(qids: list, size: int, resume_dir: Path) -> list[tuple]:
    """Trả [(part_idx, [qid...], done)]; done=True nếu file phần đã có (bỏ qua khi chạy)."""
    return [
        (pi, chunk, part_file(resume_dir, pi).exists())
        for pi, chunk in enumerate(split_parts(qids, size))
    ]


def build_pairs(qtext: str, cand: list, docs: dict, top: int) -> tuple[list, list]:
    """Cặp (query, text doc) cho top-N ứng viên có text; trả (pairs, vị trí trong cand tương ứng)."""
    pairs, pos = [], []
    for k, item in enumerate(cand[:top]):
        t = docs.get(item[0])
        if t is None:
            continue
        pairs.append((qtext, t))
        pos.append(k)
    return pairs, pos


def merge_scores(cand: list, top: int, pos: list[int], scores: list[float]) -> list:
    """Ghép điểm vào đúng thứ tự ứng viên BM25; doc thiếu điểm -> None."""
    res = [[item[0], item[1], None] for item in cand[:top]]
    for k, s in zip(pos, scores):
        res[k][2] = round(float(s), 4)
    return res


def merge_parts(plan: list, resume_dir: Path) -> dict:
    """Gộp file các phần (theo thứ tự phần + thứ tự câu) thành {qid: kết quả}."""
    out = {}
    for pi, chunk, _ in plan:
        f = part_file(resume_dir, pi)
        if not f.exists():
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for qid in chunk:
            if qid in data:
                out[qid] = data[qid]
    return out


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda().eval()
    return tok, model


def score_pairs(model, tok, pairs: list, max_len: int, batch: int, progress=None) -> list:
    """Điểm logit thô (logits[:,0]) của từng cặp (query, doc); sắp theo độ dài để giảm padding."""
    import torch
    qs = [p[0] for p in pairs]
    ds = [p[1] for p in pairs]
    enc = tok(qs, ds, truncation="only_second", max_length=max_len)["input_ids"]
    lens = [len(x) for x in enc]
    order = sorted(range(len(enc)), key=lambda i: lens[i])  # giam padding
    out = [0.0] * len(enc)
    pad = tok.pad_token_id
    done, nb = 0, 0
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            m = max(lens[i] for i in b)
            ids = torch.full((len(b), m), pad, dtype=torch.long)
            att = torch.zeros((len(b), m), dtype=torch.long)
            for r, i in enumerate(b):
                ids[r, : lens[i]] = torch.tensor(enc[i])
                att[r, : lens[i]] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, 0]
            for r, i in enumerate(b):
                out[i] = float(logits[r].item())
            done += len(b)
            nb += 1
            if progress and (nb % LOG_EVERY == 0 or done == len(enc)):
                progress(done, len(enc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, nargs="+", help="JSONL.zst {id, idx, text} (một hoặc nhiều file)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--cands", required=True, help="JSON {qid: [[doc_id, idx], ...]} theo thứ tự BM25")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top", type=int, default=200, help="rerank N ứng viên đầu mỗi câu")
    ap.add_argument("--limit-queries", type=int, default=0, help="chỉ chạy N câu đầu (0 = tất cả; benchmark)")
    ap.add_argument("--doc-chars", type=int, default=0, help="cắt text doc (0 = không cắt)")
    ap.add_argument("--resume-dir", required=True)
    a = ap.parse_args()

    resume_dir = Path(a.resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    qtext = {str(q["id"]): q["query"] for q in qs}
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    qids = [q for q in cands if q in qtext]
    if a.limit_queries > 0:
        qids = qids[: a.limit_queries]
    wanted = {item[0] for q in qids for item in cands[q]}
    docs = load_docs(a.docs, a.doc_chars, wanted)
    print(f"{len(qids)} câu, {len(docs)} doc, tải dữ liệu {time.time() - t0:.0f}s", flush=True)

    plan = plan_parts(qids, PART_SIZE, resume_dir)
    todo = [(pi, chunk) for pi, chunk, done in plan if not done]
    for pi, chunk, done in plan:
        if done:
            print(f"phần {pi}: bỏ qua (đã có {part_file(resume_dir, pi).name})", flush=True)
    total_pairs = sum(
        len(build_pairs(qtext[q], cands[q], docs, a.top)[0]) for _, chunk in todo for q in chunk
    )
    print(f"{len(todo)} phần cần chạy, {total_pairs} cặp", flush=True)

    pairs_done = 0
    if todo:
        tok, model = load_model(a.model)
        import torch
        print(f"load_model {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated() / 2**20:.0f}MB",
              flush=True)
        for pi, chunk in todo:
            pmap, plist, ppos = {}, [], []
            for q in chunk:
                pr, pos = build_pairs(qtext[q], cands[q], docs, a.top)
                pmap[q] = (cands[q], pos)
                base = len(plist)
                plist.extend(pr)
                ppos.append((q, base, base + len(pr)))
            t1 = time.time()

            def on_batch(done_p, np_p, _pi=pi):
                el = time.time() - t0
                done = pairs_done + done_p
                rate = done / max(el, 1)
                eta = (total_pairs - done) / max(rate, 1e-9)
                print(f"phần {_pi}: {done}/{total_pairs} cặp, {el:.0f}s, {rate:.1f} cặp/s, ETA {eta:.0f}s",
                      flush=True)

            sc = score_pairs(model, tok, plist, a.max_len, a.batch, progress=on_batch)
            out_part = {}
            for q, lo, hi in ppos:
                cand, pos = pmap[q]
                nmiss = len(cand[: a.top]) - len(pos)
                out_part[q] = merge_scores(cand, a.top, pos, sc[lo:hi])
                if nmiss:
                    print(f"câu {q}: thiếu text {nmiss} doc", flush=True)
            part_file(resume_dir, pi).write_text(json.dumps(out_part), encoding="utf-8")
            pairs_done += len(plist)
            el = time.time() - t1
            print(f"phần {pi}: ghi {part_file(resume_dir, pi).name}, {len(plist)} cặp, "
                  f"{el:.0f}s, {len(plist) / max(el, 1):.1f} cặp/s", flush=True)

    out = merge_parts(plan, resume_dir)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"ghi {a.out}: {len(out)} câu; tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
