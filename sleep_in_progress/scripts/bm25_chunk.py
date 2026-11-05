"""BM25 trên chunk 512 âm tiết overlap 50 (giả thuyết pool chunk của BTC, mentor 08/10).

Chia doc thành cửa sổ 512 âm tiết (token khoảng trắng) bước 462 (overlap 50); chunk là substring
nguyên văn [start_char, end_char]. Token hoá: syl = NFKC + lower + tách non-word (giữ dấu tiếng
Việt, ký tự Hán tách đơn) hoặc pyvi = ViTokenizer + lower (từ ghép nối '_'). Không bỏ stopwords,
không stemming, không băm term. Chỉ mục bm25s method lucene k1=1.5 b=0.75 trên MỌI chunk; truy hồi
top-3000 chunk/câu, điểm doc = max điểm chunk, giữ top-300 doc.

Đầu ra (trong --outdir): cache_bm25c_<tok>.json {qid: [[doc_id, rank 0-based, score], ...]} (giống
cache_pool_rr.json) và bestchunk_bm25c_<tok>.json.zst {qid: [[doc_id, chunk_idx, score,
chunk_text], ...]} cho top-300 doc. Pyvi: benchmark --bench-n doc đầu rồi ngoại suy, bỏ qua biến
thể nếu ước tính vượt --pyvi-max-hours. Chạy trên b660 (job nặng, nhiều RAM).

Dùng:
  python scripts/bm25_chunk.py run --docs D:/r2ai/data/sub/docs_vi_all.part*.jsonl.zst \
      --queries D:/r2ai/data/query.parquet --tok syl --outdir D:/r2ai/data/sub/bm25c
  python scripts/bm25_chunk.py report --cache .../cache_bm25c_syl.json \
      --ref D:/r2ai/data/sub/cache_pool_rr.json --n 30 --files .../subm_bm25c_syl_top30.zip
"""
import argparse
import glob
import hashlib
import itertools
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

SYLL_RE = re.compile(r"\S+")
SYL_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[^\W_\u4e00-\u9fff]+", re.U)

WIN = 512
OVERLAP = 50

_pyvi = None


def syllable_spans(text: str) -> list[tuple[int, int]]:
    """[(start_char, end_char)] của các token khoảng trắng (âm tiết) theo thứ tự trong text."""
    return [(m.start(), m.end()) for m in SYLL_RE.finditer(text)]


def chunk_spans(text: str, win: int = WIN, overlap: int = OVERLAP) -> list[tuple[int, int, int]]:
    """[(chunk_idx, start_char, end_char)] cửa sổ win âm tiết, bước win-overlap.

    Doc ngắn hơn win (kể cả 0 âm tiết) cho đúng 1 chunk; cửa sổ cuối có thể ngắn hơn.
    text[start:end] luôn là substring nguyên văn.
    """
    if win <= 0 or overlap < 0 or overlap >= win:
        raise ValueError("cần win > 0 và 0 <= overlap < win")
    spans = syllable_spans(text)
    if not spans:
        return [(0, 0, 0)]
    stride = win - overlap
    out = []
    i = 0
    while True:
        j = min(i + win, len(spans))
        out.append((len(out), spans[i][0], spans[j - 1][1]))
        if j >= len(spans):
            return out
        i += stride


def tokenize_syl(text: str) -> list[str]:
    """NFKC + lower + tách non-word (dấu _ cũng tách), ký tự Hán mỗi ký tự một token."""
    return SYL_TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def _pyvi_tokenizer():
    global _pyvi
    if _pyvi is None:
        from pyvi import ViTokenizer
        _pyvi = ViTokenizer
    return _pyvi


def tokenize_pyvi(text: str) -> list[str]:
    """pyvi ViTokenizer (từ ghép nối '_') rồi lower, tách theo khoảng trắng."""
    return _pyvi_tokenizer().tokenize(text).lower().split()


def make_tokenizer(name: str):
    if name == "syl":
        return tokenize_syl
    if name == "pyvi":
        return tokenize_pyvi
    raise ValueError(f"tokenizer lạ: {name}")


class IdSeq:
    """Trình duyệt list[int] trên mảng int32 phẳng (token id từng chunk) cho bm25s.index."""

    def __init__(self, flat: np.ndarray, offs):
        self.flat = flat
        self.offs = offs

    def __len__(self) -> int:
        return len(self.offs) - 1

    def __getitem__(self, i: int) -> list[int]:
        return self.flat[self.offs[i]:self.offs[i + 1]].tolist()

    def __iter__(self):
        for i in range(len(self.offs) - 1):
            yield self[i]


def aggregate_hits(hits: list, meta: list, top_docs: int) -> list[tuple]:
    """Gộp hit chunk -> doc: điểm doc = max điểm chunk, giữ top_docs theo (-điểm, doc_id).

    hits: [(chunk_row, score)]; meta[row] = (doc_id, chunk_idx, start, end).
    Trả [(doc_id, score, chunk_idx, start, end)] — chunk_idx/start/end của chunk tốt nhất
    (hòa điểm lấy chunk tới trước).
    """
    best = {}
    for row, sc in hits:
        m = meta[row]
        prev = best.get(m[0])
        if prev is None or sc > prev[0]:
            best[m[0]] = (sc, m)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:top_docs]
    return [(d, sc, m[1], m[2], m[3]) for d, (sc, m) in ranked]


def top_doc_ids(hits: list, n: int) -> list:
    return [h[0] for h in hits[:n]]


def overlap_mean(cache: dict, ref: dict, n: int = 30) -> float:
    """TB số doc chung trong top-n mỗi câu giữa cache và ref."""
    tot = 0
    qs = 0
    for q, hits in cache.items():
        a = set(top_doc_ids(hits, n))
        b = set(top_doc_ids(ref.get(q) or [], n))
        tot += len(a & b)
        qs += 1
    return tot / qs if qs else 0.0


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def expand_docs(patterns: list) -> list:
    """Mở rộng mỗi mẫu thành file (glob khi có ký tự magic); lỗi nếu mẫu ra rỗng/thiếu."""
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if glob.has_magic(p) else [p]
        hits = [x for x in hits if Path(x).is_file()]
        if not hits:
            raise FileNotFoundError(p)
        out += hits
    return out


def load_queries(path) -> list:
    """[(qid, text)] từ parquet (id, query) hoặc JSONL {id, query}."""
    p = str(path)
    if p.endswith(".parquet"):
        import pandas as pd
        q = pd.read_parquet(p)
        return [(int(i), str(t)) for i, t in zip(q["id"], q["query"])]
    out = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append((int(r["id"]), str(r["query"])))
    return out


def write_json_zst(obj, path) -> None:
    import zstandard
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(zstandard.ZstdCompressor(level=10).compress(data))


def read_json_zst(path) -> object:
    import zstandard
    with open(path, "rb") as fh:
        return json.loads(zstandard.ZstdDecompressor().decompress(fh.read()))


def count_docs(paths: list) -> int:
    """Đếm bản ghi JSONL.zst bằng số newline (không parse JSON)."""
    import zstandard
    n = 0
    for p in paths:
        with open(p, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as r:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    n += b.count(b"\n")
    return n


def peak_ram_mb():
    """(PeakWorkingSetSize MB, PeakPagefileUsage MB) hoặc None nếu không đo được."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return (pmc.PeakWorkingSetSize / 1e6, pmc.PeakPagefileUsage / 1e6)
    except Exception:
        return None


def bench_tokenizer(paths: list, n_docs: int, tokfn, limit_docs: int = 0) -> tuple:
    """Token hoá tối đa n_docs doc đầu; trả (số doc, số ký tự, giây)."""
    from rerank_run import read_docs
    t0 = time.time()
    chars = 0
    seen = 0
    for p in paths:
        for r in read_docs(p):
            t = r.get("text") or ""
            tokfn(t)
            chars += len(t)
            seen += 1
            if seen >= n_docs or (limit_docs and seen >= limit_docs):
                return seen, chars, time.time() - t0
    return seen, chars, time.time() - t0


def run(docs: list, queries, tok: str, outdir, win: int = WIN, overlap: int = OVERLAP,
        top_chunks: int = 3000, top_docs: int = 300, bench_n: int = 2000,
        pyvi_max_hours: float = 4.0, limit_docs: int = 0, log=print) -> dict | None:
    """Dựng chỉ mục BM25 chunk + truy hồi; ghi cache + best chunk vào outdir; trả stats.

    tok=pyvi: benchmark bench_n doc đầu, ước lượng toàn bộ; vượt pyvi_max_hours thì bỏ qua
    (trả None, KHÔNG ghi output). Chunking luôn 512/50 (âm tiết) bất kể tokenizer.
    """
    import bm25s
    from rerank_run import load_docs, read_docs

    t0 = time.time()
    paths = expand_docs(docs) if isinstance(docs, (list, tuple)) else expand_docs([str(docs)])
    items = load_queries(queries)
    tokfn = make_tokenizer(tok)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if tok == "pyvi":
        n_total = count_docs(paths)
        if limit_docs:
            n_total = min(n_total, limit_docs)
        seen, chars, el = bench_tokenizer(paths, bench_n, tokfn, limit_docs)
        rate = seen / el if el > 0 else 0.0
        est = n_total / rate if rate > 0 else float("inf")
        log(f"pyvi bench: {seen} doc / {chars} ký tự / {el:.1f}s → {rate:.2f} doc/s; "
            f"tổng {n_total} doc ước {est / 3600:.2f}h", flush=True)
        if est > pyvi_max_hours * 3600:
            log(f"SKIP tok=pyvi: ước lượng {est / 3600:.2f}h > {pyvi_max_hours}h", flush=True)
            return None

    vocab: dict = {}
    meta: list = []
    offs = [0]
    parts: list = []
    batch: list = []
    n_docs_seen = 0
    t1 = time.time()
    for p in paths:
        for r in read_docs(p):
            text = r.get("text") or ""
            did = int(r["id"])
            for cidx, s, e in chunk_spans(text, win, overlap):
                ids = []
                for t in tokfn(text[s:e]):
                    i = vocab.get(t)
                    if i is None:
                        i = vocab[t] = len(vocab)
                    ids.append(i)
                batch.append(ids)
                offs.append(offs[-1] + len(ids))
                meta.append((did, cidx, s, e))
            n_docs_seen += 1
            if len(batch) >= 50000:
                parts.append(np.fromiter(itertools.chain.from_iterable(batch), dtype=np.int32))
                batch.clear()
            if limit_docs and n_docs_seen >= limit_docs:
                break
        if limit_docs and n_docs_seen >= limit_docs:
            break
    if batch:
        parts.append(np.fromiter(itertools.chain.from_iterable(batch), dtype=np.int32))
        batch.clear()
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    del parts
    offs_arr = np.asarray(offs, dtype=np.int64)
    del offs
    n_chunks = len(meta)
    log(f"pass1: {n_docs_seen} doc → {n_chunks} chunk, vocab {len(vocab)}, {time.time() - t1:.0f}s",
        flush=True)

    t2 = time.time()
    retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    retriever.index(SimpleNamespace(ids=IdSeq(flat, offs_arr), vocab=vocab), show_progress=False)
    t_index = time.time() - t2
    del flat, offs_arr
    ram = peak_ram_mb()
    log(f"index bm25s lucene: {t_index:.0f}s, RAM peak "
        f"{f'{ram[0]:.0f}/{ram[1]:.0f}MB (WS/PF)' if ram else 'n/a'}", flush=True)

    t3 = time.time()
    q_toks = [(qid, tokfn(text)) for qid, text in items]
    nonempty = [(qid, toks) for qid, toks in q_toks if toks]
    cache: dict = {}
    picks: dict = {}
    if nonempty:
        rows, scs = retriever.retrieve([t for _, t in nonempty], k=min(top_chunks, n_chunks),
                                       show_progress=False)
        for (qid, _), row_arr, sc_arr in zip(nonempty, rows, scs):
            hits = [(int(row), float(sc)) for row, sc in zip(row_arr, sc_arr) if sc > 0]
            ranked = aggregate_hits(hits, meta, top_docs)
            cache[str(qid)] = [[d, rank, round(sc, 4)] for rank, (d, sc, *_ ) in enumerate(ranked)]
            picks[str(qid)] = ranked
    for qid, toks in q_toks:
        cache.setdefault(str(qid), [])
        picks.setdefault(str(qid), [])
    t_query = time.time() - t3
    log(f"truy hồi {len(items)} câu (top-{min(top_chunks, n_chunks)} chunk → top-{top_docs} doc): "
        f"{t_query:.0f}s", flush=True)

    t4 = time.time()
    wanted = {d for ranked in picks.values() for d, *_ in ranked}
    texts = load_docs(paths, wanted=wanted)
    best = {}
    missing = 0
    for qid, ranked in picks.items():
        rows = []
        for d, sc, cidx, s, e in ranked:
            t = texts.get(d)
            if t is None:
                missing += 1
                rows.append([d, cidx, round(sc, 4), ""])
                continue
            rows.append([d, cidx, round(sc, 4), t[s:e]])
        best[qid] = rows
    del texts

    cache_path = outdir / f"cache_bm25c_{tok}.json"
    best_path = outdir / f"bestchunk_bm25c_{tok}.json.zst"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    write_json_zst(best, best_path)
    ram = peak_ram_mb()
    stats = {"tok": tok, "docs": n_docs_seen, "chunks": n_chunks, "vocab": len(vocab),
             "queries": len(items), "pass1_s": round(t2 - t1, 1), "index_s": round(t_index, 1),
             "query_s": round(t_query, 1), "total_s": round(time.time() - t0, 1),
             "missing_texts": missing,
             "ram_peak_mb": [round(x, 1) for x in ram] if ram else None,
             "cache": str(cache_path), "bestchunk": str(best_path)}
    log(f"ghi {cache_path} ({len(cache)} câu) + {best_path} (thiếu text {missing} doc); "
        f"tổng {time.time() - t0:.0f}s, RAM peak "
        f"{f'{ram[0]:.0f}/{ram[1]:.0f}MB (WS/PF)' if ram else 'n/a'}", flush=True)
    return stats


def report(cache_path, ref_path, n: int = 30, files: list | None = None, log=print) -> None:
    """In overlap top-n so với ref và SHA256 + kích thước từng file."""
    cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    ov = overlap_mean(cache, ref, n)
    log(f"{cache_path}: overlap top-{n} với {ref_path}: TB {ov:.2f} doc/câu "
        f"({ov / max(n, 1) * 100:.1f}%) trên {len(cache)} câu", flush=True)
    for f in files or []:
        p = Path(f)
        sz = p.stat().st_size
        log(f"{p}: {sz} bytes ({sz / 1e6:.1f}MB), sha256 {sha256_file(p)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="dựng chỉ mục BM25 chunk + truy hồi + cache")
    r.add_argument("--docs", required=True, nargs="+", help="JSONL.zst {id,text} (file hoặc glob)")
    r.add_argument("--queries", required=True, help="parquet (id, query) hoặc JSONL {id, query}")
    r.add_argument("--tok", required=True, choices=["syl", "pyvi"])
    r.add_argument("--outdir", required=True)
    r.add_argument("--win", type=int, default=WIN)
    r.add_argument("--overlap", type=int, default=OVERLAP)
    r.add_argument("--top-chunks", type=int, default=3000)
    r.add_argument("--top-docs", type=int, default=300)
    r.add_argument("--bench-n", type=int, default=2000, help="số doc benchmark cho pyvi")
    r.add_argument("--pyvi-max-hours", type=float, default=4.0)
    r.add_argument("--limit-docs", type=int, default=0, help="chỉ N doc đầu (smoke)")
    p = sp.add_parser("report", help="overlap + sha256 các zip")
    p.add_argument("--cache", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.docs, a.queries, a.tok, a.outdir, a.win, a.overlap, a.top_chunks, a.top_docs,
            a.bench_n, a.pyvi_max_hours, a.limit_docs)
    else:
        report(a.cache, a.ref, a.n, a.files)


if __name__ == "__main__":
    main()
