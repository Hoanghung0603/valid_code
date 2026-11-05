"""Rerank mức CHUNK bằng cross-encoder bge-reranker-v2-m3 (fp16) trên GPU, chỉ đặt tải GPU.

Mỗi câu: chunk từng doc trong top-N theo thứ tự ứng viên (cache_pool_rr), tạo cặp (query, chunk_text),
chấm logit thô bằng rerank_run.score_pairs, giữ top-K chunk theo logit giảm dần. Chunk là NGUYÊN VĂN
từ text doc (chunking.chunk_document, lang 'vi', max_units=600 — cùng pick_chunks của bigpipeline;
text đã clean, KHÔNG prepend title).

Cửa sổ (--window-words W --window-stride S, mặc định 0 = hành vi cũ): cross-encoder max_len 512 chỉ
thấy ~300-400 từ đầu chunk 600 từ → khi W>0 mỗi chunk (chunk_text nộp vẫn là chunk đầy đủ) cắt thành
các cửa sổ W từ, bước S (cửa sổ cuối phủ tới hết chunk; chunk ≤ W từ thì 1 cửa sổ), chấm (query, cửa sổ),
điểm chunk = max logit các cửa sổ. Cửa sổ là đoạn NGUYÊN VĂN của chunk.

Chạy lại được: kết quả từng phần (PART_SIZE câu) lưu JSON trong --resume-dir; phần đã có thì bỏ qua.
Đầu ra: --out JSON {qid: [[doc_id, doc_rank, chunk_idx, logit, chunk_text], ...]} — doc_rank tính từ 1
theo thứ tự ứng viên; nếu JSON > 90MB thì ghi nén zstd ra .json.zst (SFTP giới hạn 100MiB/file).
Khi window-words > 0: rows trong --resume-dir kèm cột 6 = logit cửa sổ đầu; --out giữ format chính
5 cột, cột phụ ghi --first-out (mặc định <out>.wfirst.json: {qid: [[doc_id, doc_rank, chunk_idx, first_logit], ...]}).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from chunking import chunk_document
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
LOG_EVERY = 25  # in tiến độ mỗi N batch (score_pairs)
CHUNK_UNITS = 600  # từ/chunk — cùng cấu hình chunks_pool_rr (units-vi 600)
OUT_JSON_LIMIT = 90_000_000  # JSON lớn hơn 90MB thì ghi .json.zst
_WORD = re.compile(r"\S+")


def chunk_doc_texts(text: str, max_units: int = CHUNK_UNITS) -> list[str]:
    """Chunk NGUYÊN VĂN của một doc (chunk_document lang 'vi', như pick_chunks)."""
    return [c.text for c in chunk_document(text, "vi", max_units=max_units)]


def window_texts(chunk_text: str, window_words: int, window_stride: int) -> list[str]:
    """Cắt chunk thành các cửa sổ window_words từ, bước window_stride; trả list cửa sổ NGUYÊN VĂN.

    Cửa sổ cuối phủ tới hết chunk (có thể ngắn hơn W từ); chunk ≤ W từ thì đúng 1 cửa sổ = nguyên chunk.
    window_words ≤ 0: [chunk_text] (hành vi cũ, không cắt). Yêu cầu 1 ≤ window_stride ≤ window_words
    (bước ≤ W để mọi từ luôn thuộc ≥ 1 cửa sổ).
    """
    if window_words <= 0:
        return [chunk_text]
    if window_stride < 1 or window_stride > window_words:
        raise ValueError("window_stride phải trong [1, window_words]")
    spans = [m.span() for m in _WORD.finditer(chunk_text)]
    n = len(spans)
    if n <= window_words:
        return [chunk_text]
    out = []
    start = 0
    while True:
        end = min(start + window_words, n)
        out.append(chunk_text[spans[start][0]:spans[end - 1][1]])
        if end >= n:
            break
        start += window_stride
    return out


def max_pool(window_scores: list, window_counts: list) -> tuple[list, list]:
    """Gộp logit cửa sổ theo chunk: trả (logit_chunk = max các cửa sổ, logit cửa sổ đầu).

    window_counts[i] = số cửa sổ của chunk i (≥ 1); tổng counts phải bằng len(window_scores).
    """
    if sum(window_counts) != len(window_scores):
        raise ValueError("tổng window_counts phải bằng số window_scores")
    chunk_scores, first_scores = [], []
    i = 0
    for c in window_counts:
        if c < 1:
            raise ValueError("mỗi chunk phải có ≥ 1 cửa sổ")
        seg = window_scores[i : i + c]
        chunk_scores.append(max(seg))
        first_scores.append(seg[0])
        i += c
    return chunk_scores, first_scores


def build_window_pairs(qtext: str, cand: list, docs: dict, top_docs: int,
                       window_words: int = 0, window_stride: int = 0,
                       max_units: int = CHUNK_UNITS, chunk_cache: dict | None = None,
                       ) -> tuple[list, list, list]:
    """Cặp (query, cửa sổ) cho chunk của top-docs doc; trả (pairs, metas, counts).

    meta mỗi chunk: [doc_id, doc_rank (1-based theo cand), chunk_idx, chunk_text]; counts[i] = số cửa sổ
    của chunk i; pairs gom theo thứ tự chunk (mỗi chunk counts[i] cặp). Doc thiếu text bị bỏ qua
    nhưng doc_rank vẫn giữ đúng vị trí trong cand. chunk_cache {doc_id: [chunk_text]} dùng nếu có.
    """
    pairs, metas, counts = [], [], []
    for k, item in enumerate(cand[:top_docs]):
        did = item[0]
        cts = chunk_cache.get(did) if chunk_cache is not None else None
        if cts is None:
            t = docs.get(did)
            if t is None:
                continue
            cts = chunk_doc_texts(t, max_units)
            if chunk_cache is not None:
                chunk_cache[did] = cts
        for ci, ct in enumerate(cts):
            wins = window_texts(ct, window_words, window_stride)
            metas.append([did, k + 1, ci, ct])
            counts.append(len(wins))
            pairs.extend((qtext, w) for w in wins)
    return pairs, metas, counts


def build_chunk_pairs(qtext: str, cand: list, docs: dict, top_docs: int,
                      max_units: int = CHUNK_UNITS, chunk_cache: dict | None = None) -> tuple[list, list]:
    """Cặp (query, chunk_text) cho chunk của top-docs doc ứng viên theo thứ tự; trả (pairs, metas).

    meta = [doc_id, doc_rank (1-based theo cand), chunk_idx, chunk_text]. Doc thiếu text bị bỏ qua
    nhưng doc_rank vẫn giữ đúng vị trí trong cand. chunk_cache {doc_id: [chunk_text]} được dùng nếu có
    (không phải chunk lại); cache miss thì chunk từ docs[doc_id] và ghi vào cache.
    """
    pairs, metas, _ = build_window_pairs(qtext, cand, docs, top_docs, 0, 0, max_units, chunk_cache)
    return pairs, metas


def select_top(metas: list, scores: list, top_chunks: int = 40, first_scores: list | None = None) -> list:
    """Ghép logit vào meta rồi giữ top_chunks theo logit giảm dần.

    Mỗi dòng: [doc_id, doc_rank, chunk_idx, logit (4 chữ số), chunk_text]; đồng logit thì doc_rank,
    chunk_idx tăng dần. first_scores (logit cửa sổ đầu) nếu có thì thêm cột 6 (format chính giữ 5 cột).
    """
    rows = []
    for i, (m, s) in enumerate(zip(metas, scores)):
        r = [m[0], m[1], m[2], round(float(s), 4), m[3]]
        if first_scores is not None:
            r.append(round(float(first_scores[i]), 4))
        rows.append(r)
    rows.sort(key=lambda r: (-r[3], r[1], r[2]))
    return rows[:top_chunks]


def split_first_col(rows: list) -> tuple[list, list]:
    """Tách rows 6 cột -> (rows 5 cột format chính, first_rows [doc_id, doc_rank, chunk_idx, first_logit])."""
    main_rows = [r[:5] for r in rows]
    first_rows = [[r[0], r[1], r[2], r[5]] for r in rows]
    return main_rows, first_rows


def out_path(out: str, nbytes: int, limit: int = OUT_JSON_LIMIT) -> Path:
    """Đường dẫn ghi: .json, hoặc .json.zst khi nội dung vượt limit byte."""
    p = Path(out)
    if nbytes <= limit:
        return p
    name = p.name
    if name.endswith(".json"):
        return p.with_name(name[:-5] + ".json.zst")
    return p.with_name(name + ".zst")


def write_out(out: str, data: dict, limit: int = OUT_JSON_LIMIT) -> Path:
    """Ghi JSON (ensure_ascii=False); quá limit byte thì nén zstd. Trả đường dẫn đã ghi."""
    raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
    p = out_path(out, len(raw), limit)
    if p.name.endswith(".zst"):
        import zstandard
        with open(p, "wb") as f:
            with zstandard.ZstdCompressor(level=6).stream_writer(f) as w:
                w.write(raw)
    else:
        p.write_bytes(raw)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, nargs="+", help="JSONL.zst {id, idx, text} (một hoặc nhiều file)")
    ap.add_argument("--queries", required=True, help="JSONL {id, query}")
    ap.add_argument("--cands", required=True, help="JSON {qid: [[doc_id, idx], ...]} theo thứ tự ứng viên")
    ap.add_argument("--out", required=True, help=".json (tự ghi .json.zst nếu >90MB)")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--top-docs", type=int, default=40, help="chunk từ N doc đầu mỗi câu")
    ap.add_argument("--top-chunks", type=int, default=40, help="giữ N chunk theo logit mỗi câu")
    ap.add_argument("--chunk-units", type=int, default=CHUNK_UNITS)
    ap.add_argument("--window-words", type=int, default=0,
                    help="W>0: chấm cửa sổ W từ mỗi chunk, điểm chunk = max logit (0 = chấm nguyên chunk)")
    ap.add_argument("--window-stride", type=int, default=0, help="bước cửa sổ (1..W; cần khi --window-words>0)")
    ap.add_argument("--first-out", default="", help="JSON logit cửa sổ đầu (mặc định <out>.wfirst.json khi có cửa sổ)")
    ap.add_argument("--limit-queries", type=int, default=0, help="chỉ chấm N câu đầu (0 = tất cả; benchmark)")
    ap.add_argument("--resume-dir", required=True)
    a = ap.parse_args()
    if a.window_words > 0 and not (1 <= a.window_stride <= a.window_words):
        ap.error("--window-words>0 cần 1 ≤ --window-stride ≤ --window-words")

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

    def q_pairs(q: str) -> list:
        return build_window_pairs(qtext[q], cands[q], docs, a.top_docs,
                                  a.window_words, a.window_stride,
                                  a.chunk_units, chunk_cache)[0]

    chunk_cache: dict = {}
    total_pairs = 0
    for q in qids_all:
        total_pairs += len(q_pairs(q))
    wtag = f"cửa sổ {a.window_words}/{a.window_stride}" if a.window_words > 0 else "nguyên chunk"
    print(f"ước {total_pairs} cặp ({wtag}, {total_pairs / max(len(qids_all), 1):.0f} cặp/câu, "
          f"{len(chunk_cache)} doc đã chunk, {time.time() - t0:.0f}s)", flush=True)

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
                pr, meta, counts = build_window_pairs(qtext[q], cands[q], docs, a.top_docs,
                                                      a.window_words, a.window_stride,
                                                      a.chunk_units, chunk_cache)
                pmap[q] = (meta, counts)
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
                meta, counts = pmap[q]
                csc, fsc = max_pool(sc[lo:hi], counts)
                out_part[q] = select_top(meta, csc, a.top_chunks,
                                         first_scores=fsc if a.window_words > 0 else None)
            part_file(resume_dir, pi).write_text(json.dumps(out_part, ensure_ascii=False), encoding="utf-8")
            pairs_done += len(plist)
            el = time.time() - t1
            print(f"phần {pi}: ghi {part_file(resume_dir, pi).name}, {len(plist)} cặp, "
                  f"{el:.0f}s, {len(plist) / max(el, 1):.1f} cặp/s", flush=True)

    merged = merge_parts(plan, resume_dir)
    out = {}
    first_out = {}
    for q, rows in merged.items():
        if a.window_words > 0:
            out[q], fo = split_first_col(rows)
            first_out[q] = fo
        else:
            out[q] = rows
    p = write_out(a.out, out)
    if a.window_words > 0:
        fp = Path(a.first_out) if a.first_out else p.with_name(p.name.replace(".json.zst", ".json")
                                                               .replace(".json", "") + ".wfirst.json")
        fp.write_text(json.dumps(first_out, ensure_ascii=False), encoding="utf-8")
        print(f"ghi {fp} (logit cửa sổ đầu)", flush=True)
    el = time.time() - t0
    print(f"ghi {p} ({p.stat().st_size / 1e6:.0f}MB): {len(out)} câu; {pairs_done} cặp mới, "
          f"{pairs_done / max(el, 1):.1f} cặp/s; tổng {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
