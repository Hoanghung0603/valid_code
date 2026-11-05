"""Chấm cửa sổ mức DOC bằng cross-encoder bge-reranker-v2-m3 (fp16) trên GPU, chỉ đặt tải GPU.

Mỗi câu x mỗi doc trong top-N ứng viên (cands từ export_docs --no-title): cắt TEXT DOC thành
cửa sổ --window-words từ, bước --window-stride (từ = token tách khoảng trắng), ghi offset ký tự
(start, end) của mỗi cửa sổ trong text doc, chấm (query, cửa sổ) bằng rerank_run.score_pairs.
KHÔNG chunk ở bước này — cửa sổ là NGUYÊN VĂN của text doc; việc gán cửa sổ vào chunk theo
nhiều max_units làm sau ở CPU (chunks_from_docwin.py).

Chạy lại được: kết quả từng phần (PART_SIZE câu) lưu JSON trong --resume-dir; phần đã có thì bỏ qua.
Đầu ra --out: {qid: {doc_id: [[start_char, end_char, logit], ...]}} — logit thô làm tròn 4 chữ số
(nếu JSON > 90MB thì làm tròn 3 chữ số); luôn ghi nén zstd (.json.zst) để SFTP (≤100MiB/file).

Dùng:
  python scripts/docwin_run.py --docs chunkdocs100.part000.jsonl.zst ... --queries queries.jsonl \
    --cands cands_chunkdocs100.json --out docwin_top100.json.zst --resume-dir dw_parts
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from rerank_run import (
    load_docs,
    load_model,
    merge_parts,
    part_file,
    plan_parts,
    score_pairs,
    split_parts,
)

PART_SIZE = 50  # số câu mỗi phần khi lưu resume (như rerank_run)
WORD_RE = re.compile(r"\S+")
OUT_JSON_LIMIT = 90_000_000  # JSON lớn hơn 90MB thì làm tròn logit 3 chữ số


def window_spans(text: str, window_words: int, window_stride: int) -> list[tuple[int, int]]:
    """Offset ký tự (start, end) của mỗi cửa sổ window_words từ, bước window_stride.

    Từ = token tách khoảng trắng; cửa sổ là text[start:end] NGUYÊN VĂN, bám biên từ.
    Cửa sổ cuối phủ tới từ cuối cùng (có thể ngắn hơn W từ); text ≤ W từ thì 1 cửa sổ
    bao trùm mọi từ. Yêu cầu 1 ≤ window_stride ≤ window_words (bước ≤ W để mọi từ thuộc ≥ 1 cửa sổ).
    """
    if window_words < 1:
        raise ValueError("window_words phải ≥ 1")
    if window_stride < 1 or window_stride > window_words:
        raise ValueError("window_stride phải trong [1, window_words]")
    spans = [m.span() for m in WORD_RE.finditer(text)]
    n = len(spans)
    if n == 0:
        return []
    if n <= window_words:
        return [(spans[0][0], spans[-1][1])]
    out = []
    start = 0
    while True:
        end = min(start + window_words, n)
        out.append((spans[start][0], spans[end - 1][1]))
        if end >= n:
            break
        start += window_stride
    return out


def build_docwin_pairs(qtext: str, cand: list, docs: dict, top_docs: int,
                       window_words: int, window_stride: int,
                       span_cache: dict | None = None) -> tuple[list, list]:
    """Cặp (query, cửa sổ) cho top-docs doc; trả (pairs, groups).

    groups: [(doc_id, [(start_char, end_char), ...]), ...] theo thứ tự doc trong cand — mỗi doc có
    text; pairs phẳng theo groups (mỗi doc len(spans) cặp). Doc thiếu text bị bỏ qua nhưng thứ tự
    doc_rank của doc có text vẫn giữ đúng vị trí trong cand. span_cache {doc_id: [(s, e), ...]}
    được dùng nếu có (không cắt lại); cache miss thì cắt từ docs[doc_id] và ghi vào cache.
    """
    pairs, groups = [], []
    for k, item in enumerate(cand[:top_docs]):
        did = item[0]
        t = docs.get(did)
        if t is None:
            continue
        spans = span_cache.get(did) if span_cache is not None else None
        if spans is None:
            spans = window_spans(t, window_words, window_stride)
            if span_cache is not None:
                span_cache[did] = spans
        if not spans:
            continue
        groups.append((did, spans))
        pairs.extend((qtext, t[s:e]) for s, e in spans)
    return pairs, groups


def pack_doc_scores(groups: list, scores: list, ndigits: int = 4) -> dict:
    """Ghép logit theo groups: {doc_id: [[start_char, end_char, logit], ...]} (logit làm tròn)."""
    out = {}
    i = 0
    for did, spans in groups:
        rows = []
        for s, e in spans:
            rows.append([s, e, round(float(scores[i]), ndigits)])
            i += 1
        out[did] = rows
    return out


def write_zst(out: str, data: dict, limit: int = OUT_JSON_LIMIT) -> Path:
    """Ghi JSON nén zstd ra <out>.json.zst (hoặc <out>.zst); quá limit byte thì logit 3 chữ số.

    data: {qid: {doc_id: [[s, e, logit], ...]}}. Trả đường dẫn đã ghi.
    """
    import zstandard
    p = Path(out)
    if not p.name.endswith(".zst"):
        p = p.with_name(p.name + ("" if p.name.endswith(".json") else ".json") + ".zst")
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if len(raw) > limit:
        slim = {
            q: {d: [[s, e, round(lg, 3)] for s, e, lg in rows] for d, rows in by_d.items()}
            for q, by_d in data.items()
        }
        raw = json.dumps(slim, ensure_ascii=False).encode("utf-8")
        print(f"JSON > {limit / 1e6:.0f}MB: làm tròn logit 3 chữ số ({len(raw) / 1e6:.0f}MB)", flush=True)
    with open(p, "wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as w:
            w.write(raw)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, nargs="+", help="JSONL.zst {id, idx, text} (một hoặc nhiều file)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--cands", required=True, help="JSON {qid: [[doc_id, idx], ...]} theo thứ tự ứng viên")
    ap.add_argument("--out", required=True, help=".json.zst (tự thêm .zst nếu thiếu)")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top-docs", type=int, default=100, help="cửa sổ từ N doc đầu mỗi câu")
    ap.add_argument("--window-words", type=int, default=300)
    ap.add_argument("--window-stride", type=int, default=250)
    ap.add_argument("--limit-queries", type=int, default=0, help="chỉ chấm N câu đầu (0 = tất cả; benchmark)")
    ap.add_argument("--resume-dir", required=True)
    a = ap.parse_args()
    if a.window_words < 1 or not (1 <= a.window_stride <= a.window_words):
        ap.error("cần window_words ≥ 1 và 1 ≤ window_stride ≤ window_words")

    resume_dir = Path(a.resume_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip()]
    qtext = {str(q["id"]): q["query"] for q in qs}
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    qids_all = [q for q in cands if q in qtext]
    wanted = {item[0] for q in qids_all for item in cands[q][: a.top_docs]}
    docs = load_docs(a.docs, wanted=wanted)
    print(f"{len(qids_all)} câu, {len(docs)} doc, tải dữ liệu {time.time() - t0:.0f}s", flush=True)

    span_cache: dict = {}

    def q_pairs(q: str) -> list:
        return build_docwin_pairs(qtext[q], cands[q], docs, a.top_docs,
                                  a.window_words, a.window_stride, span_cache)[0]

    total_pairs = 0
    for q in qids_all:
        total_pairs += len(q_pairs(q))
    print(f"ước {total_pairs} cặp ({total_pairs / max(len(qids_all), 1):.0f} cặp/câu, "
          f"{len(span_cache)} doc đã cắt cửa sổ {a.window_words}/{a.window_stride}, "
          f"{time.time() - t0:.0f}s)", flush=True)

    qids = qids_all[: a.limit_queries] if a.limit_queries > 0 else qids_all
    plan = plan_parts(qids, PART_SIZE, resume_dir)
    todo = [(pi, chunk) for pi, chunk, done in plan if not done]
    for pi, chunk, done in plan:
        if done:
            print(f"phần {pi}: bỏ qua (đã có {part_file(resume_dir, pi).name})", flush=True)
    todo_pairs = sum(len(q_pairs(q)) for _, chunk in todo for q in chunk)
    print(f"{len(todo)} phần cần chạy, {todo_pairs} cặp", flush=True)

    pairs_done = 0
    if todo:
        tok, model = load_model(a.model)
        import torch
        print(f"load_model {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated() / 2**20:.0f}MB",
              flush=True)
        for pi, chunk in todo:
            pmap, plist, ppos = {}, [], []
            for q in chunk:
                pr, groups = build_docwin_pairs(qtext[q], cands[q], docs, a.top_docs,
                                                a.window_words, a.window_stride, span_cache)
                pmap[q] = groups
                base = len(plist)
                plist.extend(pr)
                ppos.append((q, base, base + len(pr)))
            t1 = time.time()

            def on_batch(done_p, np_p, _pi=pi):
                el = time.time() - t0
                done = pairs_done + done_p
                rate = done / max(el, 1)
                eta = (todo_pairs - done) / max(rate, 1e-9)
                print(f"phần {_pi}: {done}/{todo_pairs} cặp, {el:.0f}s, {rate:.1f} cặp/s, ETA {eta:.0f}s",
                      flush=True)

            sc = score_pairs(model, tok, plist, a.max_len, a.batch, progress=on_batch)
            out_part = {}
            for q, lo, hi in ppos:
                out_part[q] = pack_doc_scores(pmap[q], sc[lo:hi])
            part_file(resume_dir, pi).write_text(json.dumps(out_part, ensure_ascii=False), encoding="utf-8")
            pairs_done += len(plist)
            el = time.time() - t1
            print(f"phần {pi}: ghi {part_file(resume_dir, pi).name}, {len(plist)} cặp, "
                  f"{el:.0f}s, {len(plist) / max(el, 1):.1f} cặp/s", flush=True)

    merged = merge_parts(plan, resume_dir)
    p = write_zst(a.out, merged)
    el = time.time() - t0
    print(f"ghi {p} ({p.stat().st_size / 1e6:.0f}MB): {len(merged)} câu; {pairs_done} cặp mới, "
          f"{pairs_done / max(el, 1):.1f} cặp/s; tổng {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
