"""Thống kê độ dài chunk (token) của file nộp + ước lượng chồng lấn với cửa sổ vàng giả định.

Phần 1 (luôn chạy): mỗi chunk_text tách token theo khoảng trắng sau chuẩn hoá
(evaluate.tokenize, --tokenizer ws|cjk); báo p5/p25/p50/p75/p95/max, tỉ lệ > 1280 / > 1.024 /
> 900 / < 205 token, TB chunk/câu, TB token/chunk và /câu, histogram bin 100.

Phần 2 (--docs): giả thuyết chunk vàng = cửa sổ 512 âm tiết, bước 462 (overlap 50), cắt từ đầu
text doc. Mỗi chunk nộp tìm offset nguyên văn bằng text.find(chunk), quy đổi sang offset âm tiết,
tính giao [start,end) với mọi cửa sổ vàng -> tỉ lệ chunk nộp "có thể khớp" luật cũ (giao/|g| >= 0.4)
và luật mới (thêm giao/|c| >= 0.4). Chunk rỗng / không tìm được offset bị loại khỏi ước lượng
(có báo số). Cửa sổ cuối của doc có thể ngắn hơn 512 (cắt hết text).

Dùng:
  python scripts/chunk_len_stats.py submission.json --out stats.json [--tokenizer ws|cjk]
      [--docs a.jsonl.zst b.jsonl.zst ...] [--win 512] [--step 462] [--threshold 0.4]
      [--sample-queries 0] [--lock D:/r2ai/locks/heavy_job.lock] [--index D:/r2ai/data/index/vi24]
File .zip bài nộp đọc được thẳng (json trong zip, không giải nén ra đĩa).
"""
import argparse
import io
import json
import math
import re
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import normalize, tokenize  # noqa: E402

WIN = 512
STEP = 462
THRESHOLD = 0.4
BIN = 100
_WS_RUN = re.compile(r"\S+")


def chunk_len(text: str, tokenizer: str = "ws") -> int:
    """Số token của chunk sau chuẩn hoá, tách theo evaluate.tokenize."""
    return len(tokenize(text, tokenizer))


def percentile(v_sorted: list, p: float) -> float:
    """Phân vị thứ p (0-100) theo nearest-rank bậc thấp trên mảng đã sắp."""
    if not v_sorted:
        return 0
    i = max(0, min(len(v_sorted) - 1, math.ceil(p / 100.0 * len(v_sorted)) - 1))
    return v_sorted[i]


def length_stats(lengths: list) -> dict:
    """p5/p25/p50/p75/p95/max/mean + tỉ lệ >1280, >1024, >900, <205 token."""
    v = sorted(lengths)
    n = len(v) or 1
    return {
        "count": len(v),
        "p5": percentile(v, 5), "p25": percentile(v, 25), "p50": percentile(v, 50),
        "p75": percentile(v, 75), "p95": percentile(v, 95), "max": v[-1] if v else 0,
        "mean": round(sum(v) / n, 2),
        "ratio_gt_1280": round(sum(x > 1280 for x in v) / n, 4),
        "ratio_gt_1024": round(sum(x > 1024 for x in v) / n, 4),
        "ratio_gt_900": round(sum(x > 900 for x in v) / n, 4),
        "ratio_lt_205": round(sum(x < 205 for x in v) / n, 4),
    }


def histogram(lengths: list, bin_size: int = BIN) -> dict:
    """{\"0-99\": n, \"100-199\": n, ...} theo bin bin_size token (bỏ bin rỗng)."""
    if not lengths:
        return {}
    nb = max(lengths) // bin_size + 1
    counts = [0] * nb
    for x in lengths:
        counts[min(x // bin_size, nb - 1)] += 1
    return {f"{i * bin_size}-{i * bin_size + bin_size - 1}": c for i, c in enumerate(counts) if c}


def syl_starts(text: str) -> list:
    """Offset ký tự bắt đầu mỗi âm tiết (chạy không khoảng trắng) trong text."""
    return [m.start() for m in _WS_RUN.finditer(text)]


def syl_span(starts: list, cs: int, ce: int) -> tuple:
    """Khoảng âm tiết [s, e) bao phủ đoạn ký tự [cs, ce); ít nhất 1 âm tiết."""
    from bisect import bisect_left
    s = bisect_left(starts, cs)
    e = bisect_left(starts, ce)
    if e <= s:
        e = s + 1
    return s, e


def window_hit(s: int, e: int, n_syl: int, win: int = WIN, step: int = STEP,
               threshold: float = THRESHOLD) -> tuple:
    """(cũ, mới): có cửa sổ vàng [k*step, min(k*step+win, n_syl)) nào mà chunk [s,e) giao đủ ngưỡng.

    Luật cũ: giao/|g| >= threshold. Luật mới: thêm giao/|c| >= threshold (|c| = e - s).
    """
    if e <= s or n_syl <= 0:
        return False, False
    c_len = e - s
    k = max(0, (s - win) // step)
    old = False
    while k * step < n_syl:
        gs = k * step
        ge = min(gs + win, n_syl)
        inter = min(e, ge) - max(s, gs)
        if inter > 0 and inter / (ge - gs) >= threshold:
            old = True
            if inter / c_len >= threshold:
                return True, True
        k += 1
    return old, False


def load_submission(path: str) -> list:
    """Đọc bài nộp (.json hoặc .zip chứa .json)."""
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
            return json.loads(z.read(names[0]))
    return json.loads(p.read_text(encoding="utf-8"))


def read_docs_jsonl(paths, wanted=None) -> dict:
    """{str(id): [text, ...]} từ các file JSONL(.zst) {\"id\": ..., \"text\": ...}.

    GIỮ NHIỀU biến thể text của cùng doc (mỗi export có thể làm sạch khác chút);
    chunk là substring nguyên văn của biến thể lúc nó được cắt.
    """
    docs = {}
    wanted = set(map(str, wanted)) if wanted is not None else None
    for p in paths:
        p = Path(p)
        f = p.open("rb")
        raw = f
        if p.name.endswith(".zst"):
            import zstandard
            raw = zstandard.ZstdDecompressor().stream_reader(f)
        with io.TextIOWrapper(raw, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                i = str(r["id"])
                if wanted is not None and i not in wanted:
                    continue
                lst = docs.setdefault(i, [])
                if r["text"] not in lst:
                    lst.append(r["text"])
        if not p.name.endswith(".zst"):
            f.close()
    return docs


def load_docs_index(missing, index_dir) -> dict:
    """Doc text cho id thiếu từ chỉ mục + shard (export_docs.iter_all_texts, mode docwin).

    Chậm (quét shard); chỉ chạy cho id không có trong các file --docs. Text = bp.clean,
    cắt 200.000 ký tự, không title — đúng như lúc export docs để cắt chunk.
    """
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    idx = BigIndex(Path(index_dir))
    bp = BoilerplateFilter.from_json((Path(index_dir) / "boilerplate.json").read_text(encoding="utf-8"))
    pos_of = {int(idx.doc_id(i)): i for i in range(idx.n)}
    wanted = {int(d): pos_of[int(d)] for d in missing if int(d) in pos_of}
    out = {}
    for did, _i, body in iter_all_texts(idx, wanted, bp, 200000, no_title=True,
                                        log=lambda m: print(m, flush=True)):
        out[str(did)] = body
    return out


def estimate(sub: list, docs: dict, win: int = WIN, step: int = STEP,
             threshold: float = THRESHOLD) -> dict:
    """Ước lượng tỉ lệ chunk nộp có thể khớp luật cũ/mới với cửa sổ vàng giả định.

    docs: {str(doc_id): text | [text, ...]} (nhiều biến thể text/để doc). Chunk là substring
    nguyên văn của text doc; offset tìm bằng text.find(chunk) (vị trí đầu tiên), quy đổi sang
    âm tiết, giao với mọi cửa sổ vàng.
    """
    t0 = time.time()
    total = empty = miss_doc = miss_txt = found = 0
    n_old = n_new = 0
    starts_cache: dict = {}
    span_cache: dict = {}
    for item in sub:
        for ch in item.get("relevant_chunks", []):
            total += 1
            txt = ch.get("chunk_text") or ""
            if not txt.strip():
                empty += 1
                continue
            did = str(ch["doc_id"])
            texts = docs.get(did)
            if not texts:
                miss_doc += 1
                continue
            if isinstance(texts, str):
                texts = [texts]
            cc = span_cache.setdefault(did, {})
            if txt not in cc:
                cc[txt] = None
                for text in texts:
                    cs = text.find(txt)
                    if cs < 0:
                        continue
                    starts = syl_starts(text)
                    s, e = syl_span(starts, cs, cs + len(txt))
                    cc[txt] = (s, e, len(starts))
                    break
            hit = cc[txt]
            if hit is None:
                miss_txt += 1
                continue
            s, e, n_syl = hit
            old, new = window_hit(s, e, n_syl, win, step, threshold)
            found += 1
            n_old += old
            n_new += new
    return {
        "win": win, "step": step, "threshold": threshold,
        "chunks_total": total, "chunks_empty": empty, "chunks_doc_missing": miss_doc,
        "chunks_not_found_in_text": miss_txt, "chunks_found": found,
        "maybe_match_old": n_old, "maybe_match_new": n_new,
        "maybe_match_old_ratio": round(n_old / found, 4) if found else 0.0,
        "maybe_match_new_ratio": round(n_new / found, 4) if found else 0.0,
        "maybe_match_old_ratio_all": round(n_old / total, 4) if total else 0.0,
        "maybe_match_new_ratio_all": round(n_new / total, 4) if total else 0.0,
        "seconds": round(time.time() - t0, 1),
    }


def collect(sub: list, tokenizer: str, sample_queries: int = 0) -> tuple:
    """(lengths, tokens_per_query, by_doc) — by_doc: {str(doc_id): [chunk_text, ...]}."""
    lengths = []
    tokens_q = []
    by_doc: dict = {}
    items = sub[:sample_queries] if sample_queries > 0 else sub
    for item in items:
        qtok = 0
        for ch in item.get("relevant_chunks", []):
            txt = ch.get("chunk_text") or ""
            n = chunk_len(txt, tokenizer)
            lengths.append(n)
            qtok += n
            by_doc.setdefault(str(ch["doc_id"]), []).append(txt)
        tokens_q.append(qtok)
    return lengths, tokens_q, by_doc


def owned_lock(path: Path, label: str):
    """Tạo lock nguyên tử (mode x); lock của người khác -> FileExistsError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(label)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", choices=["ws", "cjk"], default="ws")
    ap.add_argument("--docs", nargs="*", default=[], help="JSONL(.zst) {id, text} của doc")
    ap.add_argument("--win", type=int, default=WIN)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--sample-queries", type=int, default=0, help="chỉ ước lượng trên N câu đầu (0 = hết)")
    ap.add_argument("--lock", default="", help="đường dẫn lock heavy_job; tạo trước/xoá sau")
    ap.add_argument("--index", default="", help="chỉ mục bigindex; đọc text doc còn thiếu từ shard (chậm)")
    a = ap.parse_args()

    lock = Path(a.lock) if a.lock else None
    if lock:
        try:
            owned_lock(lock, f"chunk_len_stats {a.submission}")
        except FileExistsError:
            print(f"LOCK BUSY: {lock} đã có chủ khác; dừng lại", flush=True)
            sys.exit(2)
    try:
        t0 = time.time()
        sub = load_submission(a.submission)
        print(f"tải {len(sub)} câu từ {a.submission} ({time.time() - t0:.0f}s)", flush=True)
        lengths, tokens_q, by_doc = collect(sub, a.tokenizer)
        n_q = len(sub) or 1
        out = {
            "submission": str(a.submission),
            "tokenizer": a.tokenizer,
            "n_queries": len(sub),
            "total_chunks": len(lengths),
            "len_tokens": length_stats(lengths),
            "histogram_bin100": histogram(lengths),
            "chunks_per_query_mean": round(len(lengths) / n_q, 2),
            "tokens_per_chunk_mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "tokens_per_query_mean": round(sum(tokens_q) / n_q, 2),
        }
        print("len_tokens", json.dumps(out["len_tokens"], ensure_ascii=False), flush=True)
        if a.docs:
            wanted = set(by_doc)
            docs = read_docs_jsonl(a.docs, wanted=wanted)
            print(f"tải {len(docs)}/{len(wanted)} doc từ {len(a.docs)} file ({time.time() - t0:.0f}s)", flush=True)
            if a.index and len(docs) < len(wanted):
                more = load_docs_index(wanted - set(docs), a.index)
                for d, body in more.items():
                    docs.setdefault(d, []).append(body)
                print(f"tải thêm {len(more)} doc từ chỉ mục ({time.time() - t0:.0f}s)", flush=True)
            sub_e = sub[: a.sample_queries] if a.sample_queries > 0 else sub
            out["estimate"] = estimate(sub_e, docs, a.win, a.step, a.threshold)
            print("estimate", json.dumps(out["estimate"], ensure_ascii=False), flush=True)
        else:
            out["estimate"] = None
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ghi {a.out}; tổng {time.time() - t0:.0f}s", flush=True)
    finally:
        if lock and lock.exists():
            lock.unlink()
            print(f"đã xoá lock {lock}", flush=True)


if __name__ == "__main__":
    main()
