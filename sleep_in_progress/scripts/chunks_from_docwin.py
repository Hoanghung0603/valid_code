"""Tạo chunk từ docwin (điểm cửa sổ mức doc) cho nhiều max_units — chạy CPU local.

Đầu vào: docwin_run.py --out ({qid: {doc_id: [[start_char, end_char, logit], ...]}}),
text doc (chunkdocs export --no-title --chars 200000), cands (thứ tự ứng viên top-100 để biết doc_rank).
Với mỗi U: chunk mỗi doc bằng chunking.chunk_document(text, 'vi', max_units=U); điểm chunk =
max logit các cửa sổ có ≥50% số ký tự nằm trong [start, end) của chunk; nếu không có cửa sổ nào
thoả thì lấy cửa sổ giao nhiều nhất (đồng overlap thì logit cao hơn). Chunk là NGUYÊN VĂN từ text.

Xuất mỗi U: {qid: [[doc_id, doc_rank, chunk_idx, logit, chunk_text], ...]} — định dạng y hệt
chunk_rerank — sắp logit giảm dần (đồng logit: doc_rank, chunk_idx tăng dần), giữ top-K chunk/câu.
Kiểm mọi chunk_text là substring nguyên văn của text doc. Sanity: top-K của U=600 giới hạn doc_rank
≤ --sanity-rank đối chiếu --sanity-ref (chunk_rerank_w_top50.json.zst).

Dùng:
  python scripts/chunks_from_docwin.py --docwin data/docwin_top100.json.zst \
    --docs .../chunkdocs100.part000.jsonl.zst ... --cands .../cands_chunkdocs100.json \
    --units 600 900 1200 --top-chunks 120 --out-dir data --sanity-ref data/chunk_rerank_w_top50.json.zst
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from chunk_rerank import select_top, write_out
from chunking import chunk_document
from rerank_run import load_docs

DEFAULT_UNITS = [600, 900, 1200]


def load_json(path: str) -> dict:
    """Đọc JSON (.json hoặc .json.zst)."""
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with open(p, "rb") as f:
            raw = zstandard.ZstdDecompressor().stream_reader(f).read()
        return json.loads(raw)
    return json.loads(p.read_text(encoding="utf-8"))


def overlap_chars(ws: int, we: int, cs: int, ce: int) -> int:
    """Số ký tự giao nhau của [ws, we) và [cs, ce)."""
    return max(0, min(we, ce) - max(ws, cs))


def chunk_score(wins: list, cs: int, ce: int) -> float | None:
    """Điểm chunk [cs, ce): max logit cửa sổ có ≥50% số ký tự trong [cs, ce).

    wins: [[start_char, end_char, logit], ...]. Không có cửa sổ nào thoả (hoặc wins rỗng ngoài
    trường hợp tất cả overlap = 0): lấy cửa sổ giao nhiều nhất; đồng overlap thì logit cao hơn.
    Trả None nếu wins rỗng.
    """
    if not wins:
        return None
    qual, best_ov, best_logit = [], -1, None
    for ws, we, lg in wins:
        ov = overlap_chars(ws, we, cs, ce)
        if ov * 2 >= we - ws:
            qual.append(lg)
        if ov > best_ov or (ov == best_ov and (best_logit is None or lg > best_logit)):
            best_ov, best_logit = ov, lg
    return max(qual) if qual else best_logit


def score_doc_chunks(chunks: list, wins: list) -> list:
    """Gán điểm cửa sổ cho các chunk đã cắt: [(chunk_idx, start, end, logit, chunk_text), ...].

    Chunk không có cửa sổ nào (wins rỗng) bị bỏ qua. chunk_text là NGUYÊN VĂN (substring của text).
    """
    out = []
    for ci, c in enumerate(chunks):
        sc = chunk_score(wins, c.start, c.end)
        if sc is None:
            continue
        out.append((ci, c.start, c.end, float(sc), c.text))
    return out


def build_q_rows(cand: list, docs: dict, wins_q: dict, max_units: int,
                 top_docs: int = 100, chunk_cache: dict | None = None) -> list:
    """Dòng chunk_rerank [doc_id, doc_rank, chunk_idx, logit, chunk_text] cho một câu (chưa sort).

    wins_q: {doc_id (str hoặc int): [[s, e, logit], ...]} của câu; doc thiếu text hoặc thiếu wins
    bị bỏ qua. doc_rank = vị trí 1-based trong cand[:top_docs] (giữ đúng vị trí khi bỏ qua).
    chunk_cache {(doc_id, max_units): [Chunk]} được dùng nếu có (không chunk lại).
    """
    rows = []
    for k, item in enumerate(cand[:top_docs]):
        did = item[0]
        t = docs.get(did)
        if t is None:
            continue
        wins = wins_q.get(str(did), wins_q.get(did))
        if not wins:
            continue
        key = (did, max_units)
        chunks = chunk_cache.get(key) if chunk_cache is not None else None
        if chunks is None:
            chunks = chunk_document(t, "vi", max_units=max_units)
            if chunk_cache is not None:
                chunk_cache[key] = chunks
        for ci, _cs, _ce, sc, ct in score_doc_chunks(chunks, wins):
            rows.append([did, k + 1, ci, sc, ct])
    return rows


def select_q_rows(rows: list, top_chunks: int) -> list:
    """Sắp logit giảm dần + cắt top_chunks (định dạng chunk_rerank, logit 4 chữ số)."""
    metas = [[r[0], r[1], r[2], r[4]] for r in rows]
    return select_top(metas, [r[3] for r in rows], top_chunks=top_chunks)


def overlap_topk(rows_a: list, rows_b: list, k: int) -> int:
    """Số dòng trùng (doc_id, chunk_idx) giữa top-k của hai danh sách đã sắp logit giảm dần."""
    ka = {(r[0], r[2]) for r in rows_a[:k]}
    kb = {(r[0], r[2]) for r in rows_b[:k]}
    return len(ka & kb)


def verify_verbatim(out: dict, docs: dict) -> int:
    """Đếm dòng có chunk_text KHÔNG phải substring nguyên văn của text doc (phải là 0)."""
    bad = 0
    for rows in out.values():
        for r in rows:
            t = docs.get(r[0])
            if t is None or r[4] not in t:
                bad += 1
    return bad


def logit_at_ranks(out: dict, ranks: list) -> dict:
    """{rank: logit TB} — trung bình logit tại hạng đó (1-based) trên các câu có đủ dòng."""
    sums = {rk: 0.0 for rk in ranks}
    ns = {rk: 0 for rk in ranks}
    for rows in out.values():
        for rk in ranks:
            if len(rows) >= rk:
                sums[rk] += rows[rk - 1][3]
                ns[rk] += 1
    return {rk: (sums[rk] / ns[rk] if ns[rk] else float("nan")) for rk in ranks}


def doc_rank_dist(out: dict, top: int = 40, lo: int = 1, hi: int = 100) -> dict:
    """Phân bố doc_rank của top-`top` dòng/câu: {"1-25": %, "26-50": %, "51-75": %, "76-100": %}."""
    step = (hi - lo + 1) // 4
    edges = [(lo, lo + step - 1), (lo + step, lo + 2 * step - 1),
             (lo + 2 * step, lo + 3 * step - 1), (lo + 3 * step, hi)]
    keys = [f"{a}-{b}" for a, b in edges]
    counts = dict.fromkeys(keys, 0)
    total = 0
    for rows in out.values():
        for r in rows[:top]:
            total += 1
            for key, (a, b) in zip(keys, edges):
                if a <= r[1] <= b:
                    counts[key] += 1
                    break
    return {key: (counts[key] / total if total else 0.0) for key in keys}


def stats(out: dict, ranks: list, top: int = 40) -> dict:
    """Thống kê một file chunk_dw: số câu, TB chunk/câu, TB từ/chunk, logit TB theo hạng, doc_rank."""
    nq = len(out)
    rows_all = [r for rows in out.values() for r in rows]
    nc = [len(rows) for rows in out.values()]
    words = [len(r[4].split()) for r in rows_all]
    return {
        "câu": nq,
        "TB chunk/câu": sum(nc) / nq if nq else 0.0,
        "TB từ/chunk": sum(words) / len(words) if words else 0.0,
        "logit TB hạng": logit_at_ranks(out, ranks),
        f"doc_rank top-{top}": doc_rank_dist(out, top),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docwin", required=True, help=".json/.json.zst của docwin_run.py")
    ap.add_argument("--docs", required=True, nargs="+", help="JSONL.zst chunkdocs (một hoặc nhiều part)")
    ap.add_argument("--cands", required=True, help="JSON {qid: [[doc_id, idx], ...]} theo thứ tự ứng viên")
    ap.add_argument("--units", type=int, nargs="+", default=DEFAULT_UNITS)
    ap.add_argument("--top-chunks", type=int, default=120, help="giữ N chunk mỗi câu")
    ap.add_argument("--top-docs", type=int, default=100)
    ap.add_argument("--out-dir", default="data", help="ghi chunk_dw_u{U}.json.zst vào thư mục này")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 15, 40, 60, 80, 100, 120],
                    help="hạng tính logit TB")
    ap.add_argument("--sanity-ref", default="", help="chunk_rerank để so top-K (U=600)")
    ap.add_argument("--sanity-k", type=int, default=15)
    ap.add_argument("--sanity-rank", type=int, default=50, help="chỉ so dòng doc_rank ≤ N")
    a = ap.parse_args()
    t0 = time.time()

    docwin = load_json(a.docwin)
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    wanted = {item[0] for q in cands for item in cands[q][: a.top_docs]}
    docs = load_docs(a.docs, wanted=wanted)
    print(f"{len(docwin)} câu docwin, {len(docs)} doc, tải dữ liệu {time.time() - t0:.0f}s", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {}
    chunk_cache: dict = {}
    for u in a.units:
        t1 = time.time()
        out = {}
        for q, wins_q in docwin.items():
            if q not in cands:
                continue
            rows = build_q_rows(cands[q], docs, wins_q, u, a.top_docs, chunk_cache)
            out[q] = select_q_rows(rows, a.top_chunks)
        bad = verify_verbatim(out, docs)
        p = write_out(str(out_dir / f"chunk_dw_u{u}.json"), out)
        outs[u] = out
        st = stats(out, a.ranks)
        print(f"U={u}: ghi {p} ({p.stat().st_size / 1e6:.0f}MB) {time.time() - t1:.0f}s; "
              f"không nguyên văn {bad}", flush=True)
        print(f"  {json.dumps(st, ensure_ascii=False)}", flush=True)

    if a.sanity_ref:
        ref = load_json(a.sanity_ref)
        u = 600
        if u not in outs:
            print(f"sanity: thiếu U={u} trong --units", flush=True)
        else:
            tot, nq = 0, 0
            for q, rows in outs[u].items():
                if q not in ref:
                    continue
                sub = [r for r in rows if r[1] <= a.sanity_rank]
                tot += overlap_topk(sub, ref[q], a.sanity_k)
                nq += 1
            print(f"sanity U=600 top-{a.sanity_k} (doc_rank ≤ {a.sanity_rank}) so với "
                  f"{Path(a.sanity_ref).name}: trùng TB {tot / max(nq, 1):.2f}/{a.sanity_k} trên {nq} câu",
                  flush=True)


if __name__ == "__main__":
    main()
