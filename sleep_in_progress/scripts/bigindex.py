"""Chỉ mục BM25 băm term cho kho hàng trăm nghìn tới hàng triệu tài liệu, không cần nạp hết vào RAM.

Dựng theo 3 bước:
  1. (song song, mỗi shard một tác vụ) đọc shard, lọc ngôn ngữ, bỏ boilerplate, token hoá, băm term về
     2**nbits ô, ghi (term, doc, tf) đã sắp theo term ra parts/.
  2. gộp: đếm df toàn cục, chia không gian term thành các dải ~budget posting, mỗi dải đọc lát cắt từ mọi part,
     tính điểm BM25 (idf, avgdl TOÀN CỤC) rồi ghi nối tiếp vào docs.u32 / scores.f32 + indptr.npy.
  3. tìm kiếm: memmap, cộng điểm các posting của từng term truy vấn, lấy top-k.

Công thức giống bm25s (phương pháp 'lucene'): idf = ln(1 + (N - df + 0.5)/(df + 0.5)),
score = idf * tf / (tf + k1 * (1 - b + b * dl / avgdl)). Va chạm băm làm hai term khác nhau dùng chung một ô
(nbits=26 -> ~67 triệu ô); ảnh hưởng tới xếp hạng nhỏ nhưng không bằng 0.

Lưu ý: không loại trùng id giữa các shard (chỉ loại trong cùng shard); build báo số id trùng, nếu > 0 thì
người dùng chỉ mục phải tự loại trùng kết quả.
"""
import argparse
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import numpy as np

from bm25 import tokenize
from boilerplate import BoilerplateFilter
from chunking import detect_lang
from shards import read_shard

NBITS_DEFAULT = 26
_BP: BoilerplateFilter | None = None


def shard_docs(path: Path, lang: str, min_chars: int = 200) -> Iterator[dict]:
    """Bản ghi status 200, text >= min_chars, đúng ngôn ngữ; loại trùng id TRONG shard. Dùng chung cho build và tra cứu."""
    seen: set[int] = set()
    for r in read_shard(path):
        doc_id, text = r.get("id"), r.get("text") or ""
        if doc_id is None or doc_id in seen or r.get("status") != 200 or len(text) < min_chars:
            continue
        if detect_lang(text) != lang:
            continue
        seen.add(doc_id)
        yield {"id": int(doc_id), "host": r.get("host", ""), "title": r.get("title") or "", "text": text}


def _lower_priority() -> None:
    """Hạ ưu tiên tiến trình để không giành CPU với crawler chạy cùng máy."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:
        pass


def _init_worker(bp_json: str) -> None:
    global _BP
    _BP = BoilerplateFilter.from_json(bp_json)
    _lower_priority()


def _done_marker(parts_dir: Path, pi: int) -> Path:
    return parts_dir / f"p{pi:05d}.done.json"


def _build_part(task: tuple) -> tuple[int, int, int]:
    pi, path, lang, nbits, parts_dir = task
    parts_dir = Path(parts_dir)
    mask = (1 << nbits) - 1
    bp = _BP
    T, D, F, ids, dls = [], [], [], [], []
    for li, d in enumerate(shard_docs(Path(path), lang)):
        toks = tokenize(d["title"] + "\n" + bp.clean(d["host"], d["text"]), lang)
        ids.append(d["id"])
        dls.append(len(toks))
        if not toks:
            continue
        hs = np.fromiter((zlib.crc32(t.encode("utf-8")) & mask for t in toks), dtype=np.uint32, count=len(toks))
        u, c = np.unique(hs, return_counts=True)
        T.append(u)
        D.append(np.full(len(u), li, dtype=np.uint32))
        F.append(np.minimum(c, 65535).astype(np.uint16))
    if T:
        t, dd, f = np.concatenate(T), np.concatenate(D), np.concatenate(F)
        order = np.argsort(t, kind="stable")
        t, dd, f = t[order], dd[order], f[order]
    else:
        t, dd, f = np.zeros(0, np.uint32), np.zeros(0, np.uint32), np.zeros(0, np.uint16)
    pre = parts_dir / f"p{pi:05d}"
    np.save(f"{pre}.terms.npy", t)
    np.save(f"{pre}.docs.npy", dd)
    np.save(f"{pre}.tf.npy", f)
    np.save(f"{pre}.ids.npy", np.asarray(ids, dtype=np.int64))
    np.save(f"{pre}.dl.npy", np.asarray(dls, dtype=np.int32))
    _done_marker(parts_dir, pi).write_text(json.dumps({"docs": len(ids), "postings": int(len(t))}))
    return pi, len(ids), int(len(t))


def fit_boilerplate(shards: list[Path], lang: str, max_fit_docs: int = 3000, min_chars: int = 200) -> BoilerplateFilter:
    """Học dòng lặp theo host; mỗi host chỉ đếm tối đa max_fit_docs trang đầu. Đọc tuần tự."""
    count: dict[str, int] = {}

    def gen() -> Iterator[dict]:
        for p in shards:
            for r in read_shard(p):
                host, text = r.get("host", ""), r.get("text") or ""
                if count.get(host, 0) >= max_fit_docs or r.get("status") != 200 or len(text) < min_chars:
                    continue
                if detect_lang(text) != lang:
                    continue
                count[host] = count.get(host, 0) + 1
                yield {"host": host, "text": text}

    return BoilerplateFilter(max_fit_docs=max_fit_docs).fit(gen())


def merge(out: Path, nparts: int, nbits: int, lang: str, shard_paths: list[str],
          budget: int = 60_000_000, k1: float = 1.2, b: float = 0.75) -> dict:
    NB = 1 << nbits
    pdir = out / "parts"
    pre = lambda p: str(pdir / f"p{p:05d}")  # noqa: E731
    ids_l, dl_l, counts = [], [], []
    for p in range(nparts):
        ids_l.append(np.load(f"{pre(p)}.ids.npy"))
        dl_l.append(np.load(f"{pre(p)}.dl.npy"))
        counts.append(len(ids_l[-1]))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    N = int(offsets[-1])
    ids = np.concatenate(ids_l)
    dl = np.concatenate(dl_l)
    doc_part = np.repeat(np.arange(nparts, dtype=np.uint16), counts)
    avgdl = float(dl.mean()) if N else 1.0
    norm = (k1 * (1 - b + b * dl / avgdl)).astype(np.float32)

    terms = [np.load(f"{pre(p)}.terms.npy", mmap_mode="r") for p in range(nparts)]
    docs = [np.load(f"{pre(p)}.docs.npy", mmap_mode="r") for p in range(nparts)]
    tfs = [np.load(f"{pre(p)}.tf.npy", mmap_mode="r") for p in range(nparts)]

    df = np.zeros(NB, dtype=np.int64)
    for t in terms:
        if len(t):
            u, c = np.unique(np.asarray(t), return_counts=True)
            df[u] += c
    indptr = np.zeros(NB + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    total = int(indptr[-1])

    cuts = np.searchsorted(indptr[1:], np.arange(budget, total, budget), side="left") + 1
    bounds = [0] + sorted(set(int(x) for x in cuts if 0 < x < NB)) + [NB]

    written = 0
    with open(out / "docs.u32", "wb") as fd, open(out / "scores.f32", "wb") as fs:
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            TT, DD, FF = [], [], []
            for p in range(nparts):
                a, z = np.searchsorted(terms[p], [np.uint32(lo), np.uint32(hi)])
                if z > a:
                    TT.append(np.asarray(terms[p][a:z]))
                    DD.append(np.asarray(docs[p][a:z]) + np.uint32(offsets[p]))
                    FF.append(np.asarray(tfs[p][a:z]))
            if not TT:
                continue
            t, d, f = np.concatenate(TT), np.concatenate(DD), np.concatenate(FF)
            order = np.argsort(t, kind="stable")
            t, d, f = t[order], d[order], f[order]
            dft = df[t].astype(np.float64)
            idf = np.log1p((N - dft + 0.5) / (dft + 0.5)).astype(np.float32)
            f32 = f.astype(np.float32)
            s = (idf * f32 / (f32 + norm[d])).astype(np.float32)
            d.astype(np.uint32).tofile(fd)
            s.tofile(fs)
            written += len(t)
    if written != total:
        raise RuntimeError(f"ghi {written} posting nhưng df tổng = {total}")

    np.save(out / "indptr.npy", indptr)
    np.save(out / "ids.npy", ids)
    np.save(out / "dl.npy", dl)
    np.save(out / "doc_part.npy", doc_part)
    meta = {"nbits": nbits, "docs": N, "postings": total, "avgdl": avgdl, "k1": k1, "b": b, "lang": lang,
            "shards": shard_paths, "unique_ids": int(len(np.unique(ids))), "ranges": len(bounds) - 1}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def build(shards: list[Path], out: Path, lang: str = "vi", nbits: int = NBITS_DEFAULT, workers: int = 4,
          budget: int = 60_000_000, reuse_bp: bool = True, keep_parts: bool = False, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pdir = out / "parts"
    pdir.mkdir(exist_ok=True)
    shards = sorted(shards)
    bp_path = out / "boilerplate.json"
    t0 = time.time()
    if reuse_bp and bp_path.exists():
        bp_json = bp_path.read_text(encoding="utf-8")
        log(f"dùng lại boilerplate.json ({bp_path})")
    else:
        bp = fit_boilerplate(shards, lang)
        bp_json = bp.to_json()
        bp_path.write_text(bp_json, encoding="utf-8")
        log(f"boilerplate: {len(json.loads(bp_json))} host có dòng lặp ({time.time() - t0:.0f}s)")
    todo = [(i, str(p), lang, nbits, str(pdir)) for i, p in enumerate(shards) if not _done_marker(pdir, i).exists()]
    log(f"{len(shards)} shard, còn {len(todo)} cần dựng, {workers} worker")
    done_docs = done_post = 0
    if workers <= 1:
        _init_worker(bp_json)
        results = map(_build_part, todo)
    else:
        pool = Pool(workers, initializer=_init_worker, initargs=(bp_json,))
        results = pool.imap_unordered(_build_part, todo)
    for k, (pi, nd, npost) in enumerate(results, 1):
        done_docs += nd
        done_post += npost
        log(f"[{time.time() - t0:7.0f}s] part {pi:3d} xong ({k}/{len(todo)}): {nd} doc, {npost} posting")
    if workers > 1:
        pool.close()
        pool.join()
    log("gộp...")
    meta = merge(out, len(shards), nbits, lang, [str(p) for p in shards], budget=budget)
    if not keep_parts:
        for f in pdir.glob("p*"):
            f.unlink()
        pdir.rmdir()
    log(f"xong {time.time() - t0:.0f}s: {meta['docs']} doc ({meta['unique_ids']} id duy nhất), "
        f"{meta['postings']} posting, {meta['ranges']} dải")
    return meta


class BigIndex:
    def __init__(self, path: Path):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.mask = (1 << self.meta["nbits"]) - 1
        self.n = self.meta["docs"]
        self.indptr = np.load(path / "indptr.npy", mmap_mode="r")
        self.docs = np.memmap(path / "docs.u32", dtype=np.uint32, mode="r")
        self.scores = np.memmap(path / "scores.f32", dtype=np.float32, mode="r")
        self.ids = np.load(path / "ids.npy")
        self.doc_part = np.load(path / "doc_part.npy")
        self.shards = self.meta["shards"]

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        """[(chỉ số doc, điểm)] giảm dần, chỉ điểm > 0. Term lặp trong truy vấn chỉ tính một lần."""
        hs = {zlib.crc32(t.encode("utf-8")) & self.mask for t in query_tokens}
        if not hs or not self.n:
            return []
        acc = np.zeros(self.n, dtype=np.float32)
        for h in hs:
            a, z = int(self.indptr[h]), int(self.indptr[h + 1])
            if z > a:
                acc[self.docs[a:z]] += self.scores[a:z]
        k = min(k, self.n)
        top = np.argpartition(acc, -k)[-k:]
        top = top[np.argsort(-acc[top], kind="stable")]
        return [(int(i), float(acc[i])) for i in top if acc[i] > 0]

    def doc_id(self, idx: int) -> int:
        return int(self.ids[idx])

    def shard_of(self, idx: int) -> str:
        return self.shards[int(self.doc_part[idx])]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--shards", nargs="+", required=True, help="một hoặc nhiều thư mục chứa *.jsonl.zst")
    b.add_argument("--out", required=True)
    b.add_argument("--lang", default="vi", choices=["vi", "zh"])
    b.add_argument("--nbits", type=int, default=NBITS_DEFAULT)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--budget", type=int, default=60_000_000)
    b.add_argument("--refit-boilerplate", action="store_true")
    b.add_argument("--keep-parts", action="store_true")
    q = sub.add_parser("query")
    q.add_argument("--index", required=True)
    q.add_argument("--q", required=True)
    q.add_argument("-k", type=int, default=10)
    a = ap.parse_args()

    if a.cmd == "build":
        paths = [p for d in a.shards for p in sorted(Path(d).glob("*.jsonl.zst"))]
        build(paths, Path(a.out), a.lang, a.nbits, a.workers, a.budget, reuse_bp=not a.refit_boilerplate,
              keep_parts=a.keep_parts, log=lambda m: print(m, file=sys.stderr, flush=True))
    else:
        idx = BigIndex(Path(a.index))
        for i, s in idx.search(tokenize(a.q, idx.meta["lang"]), k=a.k):
            print(idx.doc_id(i), round(s, 3))


if __name__ == "__main__":
    main()
