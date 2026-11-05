"""Incremental, sharded indexing: docs parquet -> chunks parquet -> BM25 (per language) + dense.

Layout (all under cfg.index.dir, default `indexes/`; chunks under data/chunks/):
  manifest.json                     ordered list of "entries" (one per source doc file):
                                    {name, doc_file, size, mtime, n_docs, n_chunks, doc_offset,
                                     chunk_offset, active}. Global ids are stable and append-only:
                                     chunk gid = chunk_offset + row, doc gid = doc_offset + doc_row.
  doctab/<name>.parquet             per doc row: doc_id, lang, source, title, first, n (chunk rows)
  data/chunks/<name>.parquet        Chunk rows (schema in docs/INTERFACES.md), row groups of 2048
  bm25/<lang>/<shard>/              compact postings (see BM25Shard): terms uint32, ptr int64,
                                    rows uint16 (<=65535 chunks/shard), vals uint8, gids int64,
                                    meta.json; IDF is computed globally over all shards at load.
  dense/<unit>_<lang>/              e<entry>.f16.npy + e<entry>.ids.npy (brute force, mmap) and,
                                    above ivf.min_vectors, ivf.faiss (IVF-PQ, ids = gids).

CLI:
  python -m src.index chunk   [--docs 'data/docs/*.parquet']
  python -m src.index bm25    [--lang vi zh en]
  python -m src.index dense   [--lang vi zh en] [--max-files N]
  python -m src.index ivf     [--lang zh]          # train/add IVF-PQ once enough vectors exist
  python -m src.index all
  python -m src.index stats
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
import zlib
from collections import Counter, OrderedDict
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.chunk import chunk_doc

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("vi", "zh", "en")
CHUNK_SCHEMA = pa.schema([("chunk_id", pa.string()), ("doc_id", pa.string()), ("lang", pa.string()),
                          ("source", pa.string()), ("start", pa.int32()), ("end", pa.int32()),
                          ("text", pa.string())])


# =============================================================================== config
def load_config(path: str | None = None) -> dict:
    import yaml
    p = Path(path) if path else ROOT / "configs/default.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def icfg(cfg: dict | None) -> dict:
    """index block with defaults."""
    cfg = cfg or {}
    ix = dict(cfg.get("index") or {})
    ix.setdefault("dir", (cfg.get("paths") or {}).get("indexes", "indexes"))
    ix.setdefault("chunks_dir", (cfg.get("paths") or {}).get("chunks", "data/chunks"))
    ix.setdefault("docs_glob", str(Path((cfg.get("paths") or {}).get("docs", "data/docs")) / "*.parquet"))
    ix.setdefault("workers", 8)
    b = dict(ix.get("bm25") or {})
    for k, v in {"k1": 1.2, "b": 0.75, "hash_bits": 24, "shard_max_chunks": 65000, "zh_hmm": False,
                 "zh_char_bigrams": False, "vi_bigrams": True, "vi_pyvi": False}.items():
        b.setdefault(k, v)
    ix["bm25"] = b
    d = dict(ix.get("dense") or {})
    d.setdefault("unit", {"vi": "doc", "zh": "doc", "en": "chunk"})
    d.setdefault("doc_lead_chars", 150)
    iv = dict(d.get("ivf") or {})
    for k, v in {"min_vectors": 300_000, "factory": "IVF4096,PQ64", "train_size": 200_000,
                 "nprobe": 64, "keep_fp16": False, "add_every": 20}.items():
        iv.setdefault(k, v)
    d["ivf"] = iv
    ix["dense"] = d
    return ix


def _unit(ix: dict, lang: str) -> str:
    u = ix["dense"]["unit"]
    return u.get(lang, "chunk") if isinstance(u, dict) else u


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


# ============================================================================ tokenizers
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CJK_RUN = re.compile(r"[㐀-鿿豈-﫿]+")
_HAS_W = re.compile(r"\w", re.UNICODE)

VI_STOP = set("""và của là có các những được cho với này đó thì mà như khi để trong một ra vào
lại cũng đã đang sẽ rằng nên nếu nhưng hay hoặc thế vậy nào gì ạ à ơi ạ nhé nhỉ xin hỏi em tôi mình
anh chị bác sĩ cháu con ở tại từ về theo đến bởi vì do""".split())
ZH_STOP = set("""的 了 是 在 和 与 及 或 也 就 都 而 等 这 那 我 你 他 她 它 们 吗 呢 吧 啊 呀 么 什么 怎么
一个 可以 如果 因为 所以 但是 还 会 要 被 把 对 从 到 为 以 之 其 着 过 请问 医生 您好 你好 一下""".split())
EN_STOP = set("""a an the and or of to in on for with by from as at is are was were be been being this
that these those it its which who whom what when where why how not no but if than then so such can
could may might will would should do does did has have had we our you your they their there here
also into about between after before during over under more most other some any all each both""".split())


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer
    return Stemmer.Stemmer("english")


_JIEBA_READY = False


def _jieba():
    global _JIEBA_READY
    import jieba
    if not _JIEBA_READY:
        jieba.setLogLevel(60)
        jieba.initialize()
        _JIEBA_READY = True
    return jieba


_HF_TOK = None


def _hf_tok():
    """Tokenizer HuggingFace (vd bge-m3/XLM-R) khi đặt env R2AI_HF_TOK=<đường dẫn>; dùng thử để mô phỏng BM25 của BTC."""
    global _HF_TOK
    if _HF_TOK is None:
        import os
        from tokenizers import Tokenizer
        _HF_TOK = Tokenizer.from_file(os.path.join(os.environ["R2AI_HF_TOK"], "tokenizer.json"))
    return _HF_TOK


def tokenize(text: str, lang: str, bcfg: dict | None = None) -> list[str]:
    import os
    if os.environ.get("R2AI_HF_TOK"):
        toks = _hf_tok().encode(unicodedata.normalize("NFKC", text).lower(), add_special_tokens=False).tokens
        return [t for t in toks if t.strip("\u2581")]
    bcfg = bcfg or {}
    text = unicodedata.normalize("NFC", text).lower()
    if lang == "zh":
        jb = _jieba()
        toks = [t for t in jb.lcut(text, HMM=bool(bcfg.get("zh_hmm", False)))
                if t.strip() and _HAS_W.search(t) and t not in ZH_STOP]
        if bcfg.get("zh_char_bigrams", False):
            for run in _CJK_RUN.findall(text):
                toks.extend(run[i:i + 2] for i in range(len(run) - 1))
        return toks
    if lang == "en":
        words = [w for w in _WORD.findall(text) if w not in EN_STOP]
        return _stemmer().stemWords(words)
    # vi (default): syllables + syllable bigrams (cheap stand-in for word segmentation)
    if bcfg.get("vi_pyvi", False):
        from pyvi import ViTokenizer
        words = ViTokenizer.tokenize(text).split()
        return [w for w in words if _HAS_W.search(w) and w not in VI_STOP]
    syl = _WORD.findall(text)
    toks = [s for s in syl if s not in VI_STOP]
    if bcfg.get("vi_bigrams", True):
        toks.extend(f"{a}_{b}" for a, b in zip(syl, syl[1:]))
    return toks


def hash_tokens(toks: list[str], bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    return np.fromiter((zlib.crc32(t.encode("utf-8")) & mask for t in toks), dtype=np.int64, count=len(toks))


def _tok_hash_batch(args):
    texts, lang, bcfg = args
    bits = int(bcfg["hash_bits"])
    lens = np.zeros(len(texts), dtype=np.int32)
    terms, tfs, nu = [], [], np.zeros(len(texts), dtype=np.int32)
    for i, t in enumerate(texts):
        h = hash_tokens(tokenize(t, lang, bcfg), bits)
        lens[i] = len(h)
        if len(h):
            u, c = np.unique(h, return_counts=True)
            terms.append(u.astype(np.int32))
            tfs.append(c.astype(np.int32))
            nu[i] = len(u)
    terms = np.concatenate(terms) if terms else np.zeros(0, np.int32)
    tfs = np.concatenate(tfs) if tfs else np.zeros(0, np.int32)
    return lens, nu, terms, tfs


# ================================================================================ BM25
class BM25Shard:
    """Compact postings of one shard: sparse hashed vocab + CSC-like slices.

    terms   uint32[T]  sorted unique hashed term ids present in the shard
    ptr     int64[T+1] posting slice of terms[i] = rows/vals[ptr[i]:ptr[i+1]]
    rows    uint16 (n < 65536) or int32 local chunk index
    vals    uint8  quantised BM25 tf-part tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)) / (k1+1) * 255
    gids    int64[n] global chunk ids
    => ~3 bytes per posting (uint16 rows) instead of ~8-12 for float32/int32 CSC.
    """

    FILES = ("terms", "ptr", "rows", "vals", "gids")

    def __init__(self, terms, ptr, rows, vals, gids, meta):
        self.terms, self.ptr, self.rows, self.vals, self.gids, self.meta = terms, ptr, rows, vals, gids, meta
        self.n = int(meta["n"])
        self.scale = (float(meta["k1"]) + 1.0) / 255.0

    @classmethod
    def build(cls, lens, nu, terms, tfs, gids, bcfg) -> "BM25Shard":
        k1, b = float(bcfg["k1"]), float(bcfg["b"])
        n = len(lens)
        avgdl = float(lens.mean()) if n else 1.0
        rows = np.repeat(np.arange(n, dtype=np.int32), nu)
        dl = lens[rows].astype(np.float32)
        tf = tfs.astype(np.float32)
        w = tf / (tf + k1 * (1 - b + b * dl / max(avgdl, 1e-6)))  # in (0, 1)
        vals = np.clip(np.rint(w * 255), 1, 255).astype(np.uint8)
        del dl, tf, w
        order = np.argsort(terms, kind="stable")
        terms_s = terms[order]
        rows = rows[order].astype(np.uint16 if n < 65536 else np.int32)
        vals = vals[order]
        del order
        uterms, starts = np.unique(terms_s, return_index=True)
        ptr = np.append(starts, len(terms_s)).astype(np.int64)
        meta = {"n": n, "avgdl": avgdl, "nnz": int(len(rows)), "k1": k1, "b": b,
                "hash_bits": int(bcfg["hash_bits"]), "format": 2}
        return cls(uterms.astype(np.uint32), ptr, rows, vals, np.asarray(gids, dtype=np.int64), meta)

    @property
    def df_terms(self):
        return np.asarray(self.terms), np.diff(np.asarray(self.ptr))

    def save(self, d: Path):
        tmp = d.with_name(d.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for k in self.FILES:
            np.save(tmp / f"{k}.npy", getattr(self, k))
        _write_json(tmp / "meta.json", self.meta)
        if d.exists():
            shutil.rmtree(d)
        os.replace(tmp, d)

    @classmethod
    def load(cls, d: Path, mmap: bool = True) -> "BM25Shard":
        mm = "r" if mmap else None
        arrs = {k: np.load(d / f"{k}.npy", mmap_mode=mm) for k in cls.FILES}
        arrs["terms"] = np.asarray(arrs["terms"])  # small; keep in RAM for searchsorted
        return cls(meta=_read_json(d / "meta.json", {}), **arrs)

    def search(self, qterms: np.ndarray, qweights: np.ndarray, k: int):
        """qweights already include global idf. Returns (local_rows, scores)."""
        pos = np.searchsorted(self.terms, qterms)
        acc = None
        for p, t, w in zip(pos, qterms, qweights):
            if p >= len(self.terms) or self.terms[p] != t:
                continue
            s, e = int(self.ptr[p]), int(self.ptr[p + 1])
            if acc is None:
                acc = np.zeros(self.n, dtype=np.float32)
            acc[np.asarray(self.rows[s:e])] += (w * self.scale) * np.asarray(self.vals[s:e], dtype=np.float32)
        if acc is None:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        nz = np.flatnonzero(acc)
        if len(nz) > k:
            top = nz[np.argpartition(-acc[nz], k - 1)[:k]]
        else:
            top = nz
        top = top[np.argsort(-acc[top], kind="stable")]
        return top, acc[top]


class BM25Index:
    """All shards of one language with global IDF (df summed over shards)."""

    def __init__(self, shards: list[BM25Shard], bcfg: dict):
        self.shards = shards
        self.bcfg = bcfg
        self.N = sum(s.n for s in shards)
        self.idf = None
        if shards:
            df = np.zeros(1 << int(shards[0].meta["hash_bits"]), dtype=np.int32)
            for s in shards:
                t, c = s.df_terms
                df[t] += c.astype(np.int32)
            self.idf = np.log(1.0 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)

    def query_terms(self, texts: list[str], lang: str, weights: list[float] | None = None):
        cnt: Counter = Counter()
        for i, t in enumerate(texts):
            w = 1.0 if weights is None else weights[i]
            for h in set(hash_tokens(tokenize(t, lang, self.bcfg), int(self.bcfg["hash_bits"])).tolist()):
                cnt[h] += w
        if not cnt:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        terms = np.array(sorted(cnt), dtype=np.int64)
        qw = np.array([1.0 + math.log(cnt[t]) if cnt[t] >= 1 else cnt[t] for t in terms], dtype=np.float32)
        return terms, qw

    def search(self, texts: list[str], lang: str, k: int = 100, weights: list[float] | None = None):
        """Returns list of (gid, score) sorted desc."""
        if not self.shards:
            return []
        terms, qw = self.query_terms(texts, lang, weights)
        if len(terms) == 0:
            return []
        w = qw * self.idf[terms]
        res_g, res_s = [], []
        for sh in self.shards:
            r, s = sh.search(terms, w, k)
            if len(r):
                res_g.append(np.asarray(sh.gids)[r])
                res_s.append(s)
        if not res_g:
            return []
        g = np.concatenate(res_g)
        s = np.concatenate(res_s)
        o = np.argsort(-s, kind="stable")[:k]
        return list(zip(g[o].tolist(), s[o].tolist()))


def build_bm25_memory(texts: list[str], lang: str, bcfg: dict) -> BM25Index:
    """Small in-memory index (e.g. PubMed candidates of one query); gids = 0..n-1."""
    bc = dict(bcfg)
    bc["hash_bits"] = min(int(bc.get("hash_bits", 22)), 20)
    lens, nu, terms, tfs = _tok_hash_batch((texts, lang, bc))
    sh = BM25Shard.build(lens, nu, terms, tfs, np.arange(len(texts)), bc)
    return BM25Index([sh], bc)


# ============================================================================ manifest
class Manifest:
    def __init__(self, ix: dict):
        self.ix = ix
        self.dir = _abs(ix["dir"])
        self.path = self.dir / "manifest.json"
        self.data = _read_json(self.path, {"entries": []})
        self.entries: list[dict] = self.data["entries"]

    def save(self):
        _write_json(self.path, self.data)

    @property
    def n_chunks(self):
        return sum(e["n_chunks"] for e in self.entries)

    @property
    def n_docs(self):
        return sum(e["n_docs"] for e in self.entries)

    def chunk_file(self, e) -> Path:
        return _abs(self.ix["chunks_dir"]) / f"{e['name']}.parquet"

    def doctab_file(self, e) -> Path:
        return self.dir / "doctab" / f"{e['name']}.parquet"


def _doc_key(doc_id: str, source: str) -> int:
    try:
        v = int(doc_id)
    except ValueError:
        v = zlib.crc32(doc_id.encode()) + (1 << 40)
    return v + (1 << 50 if source == "pubmed" else 0)


def _chunk_worker(args):
    rows, ccfg = args
    out_docs = []
    cols = {k: [] for k in CHUNK_SCHEMA.names}
    for d in rows:
        if not d.get("text"):
            out_docs.append(0)
            continue
        ch = chunk_doc(d, ccfg)
        out_docs.append(len(ch))
        for c in ch:
            for k in CHUNK_SCHEMA.names:
                cols[k].append(c[k])
    return out_docs, cols


def build_chunks(cfg: dict, docs_glob: str | None = None, workers: int | None = None, verbose=True) -> Manifest:
    """Chunk every new/changed doc parquet into data/chunks/<name>.parquet (append-only manifest)."""
    ix = icfg(cfg)
    ccfg = cfg.get("chunk") or {}
    man = Manifest(ix)
    files = sorted(p for p in glob.glob(str(_abs(docs_glob or ix["docs_glob"])))
                   if not Path(p).name.startswith("."))
    known = {}
    for e in man.entries:
        if e.get("active", True):
            known[e["doc_file"]] = e
    # seen doc keys (dedupe across shards: crawler may write a doc in several files)
    seen: set[int] = set()
    for e in man.entries:
        if e.get("active", True) and man.doctab_file(e).exists():
            t = pq.read_table(man.doctab_file(e), columns=["doc_id", "source"])
            for did, src in zip(t["doc_id"].to_pylist(), t["source"].to_pylist()):
                seen.add(_doc_key(did, src))
    workers = workers or int(ix["workers"])
    ctx = get_context("fork")
    pool = ctx.Pool(workers) if workers > 1 else None
    try:
        for f in files:
            rel = os.path.relpath(f, ROOT)
            st = os.stat(f)
            e = known.get(rel)
            if e and e["size"] == st.st_size and int(e["mtime"]) == int(st.st_mtime):
                continue
            if e:  # changed file -> tombstone old entry, re-add
                e["active"] = False
            t0 = time.time()
            idx = len(man.entries)
            name = f"{Path(f).stem}-e{idx:05d}"
            pf = pq.ParquetFile(f)
            have = set(pf.schema_arrow.names)
            cols = [c for c in ("doc_id", "source", "lang", "title", "text") if c in have]
            entry = {"name": name, "doc_file": rel, "size": st.st_size, "mtime": int(st.st_mtime),
                     "n_docs": 0, "n_chunks": 0, "doc_offset": man.n_docs, "chunk_offset": man.n_chunks,
                     "active": True}
            cpath = man.chunk_file(entry)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            schema = CHUNK_SCHEMA
            writer = pq.ParquetWriter(str(cpath) + ".tmp", schema, compression="zstd")
            dt = {"doc_id": [], "lang": [], "source": [], "title": [], "first": [], "n": []}
            nchunks = 0
            buf_cols = {k: [] for k in schema.names}

            def flush_buf(force=False):
                if buf_cols["chunk_id"] and (force or len(buf_cols["chunk_id"]) >= 8192):
                    writer.write_table(pa.table({k: buf_cols[k] for k in schema.names}, schema=schema),
                                       row_group_size=2048)
                    for k in buf_cols:
                        buf_cols[k].clear()

            batches = []
            for rb in pf.iter_batches(batch_size=512, columns=cols):
                rows = rb.to_pylist()
                for r in rows:
                    r["doc_id"] = str(r["doc_id"])
                    r.setdefault("source", "corpus")
                    r.setdefault("title", "")
                    k = _doc_key(r["doc_id"], r.get("source") or "corpus")
                    if k in seen:
                        r["_dup"] = True
                        r["text"] = ""
                    else:
                        seen.add(k)
                batches.append(rows)
            jobs = [(rows, ccfg) for rows in batches]
            it = pool.imap(_chunk_worker, jobs, chunksize=1) if pool else map(_chunk_worker, jobs)
            for rows, (nper, ccols) in zip(batches, it):
                for r, n in zip(rows, nper):
                    dt["doc_id"].append(r["doc_id"])
                    dt["lang"].append(r.get("lang") or "")
                    dt["source"].append("dup" if r.get("_dup") else (r.get("source") or "corpus"))
                    dt["title"].append(r.get("title") or "")
                    dt["first"].append(nchunks)
                    dt["n"].append(n)
                    nchunks += n
                for k in schema.names:
                    buf_cols[k].extend(ccols[k])
                flush_buf()
            flush_buf(force=True)
            writer.close()
            os.replace(str(cpath) + ".tmp", cpath)
            dpath = man.doctab_file(entry)
            dpath.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"doc_id": dt["doc_id"], "lang": dt["lang"], "source": dt["source"],
                                     "title": dt["title"], "first": pa.array(dt["first"], pa.int64()),
                                     "n": pa.array(dt["n"], pa.int32())}), dpath, compression="zstd")
            entry["n_docs"] = len(dt["doc_id"])
            entry["n_chunks"] = nchunks
            man.entries.append(entry)
            man.save()
            if verbose:
                print(f"[chunk] {rel}: {entry['n_docs']} docs -> {nchunks} chunks "
                      f"({time.time() - t0:.1f}s)", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return man


# ========================================================================== chunk store
class ChunkStore:
    """Random access to chunks / docs by global id (row-group LRU cache)."""

    def __init__(self, cfg: dict, man: Manifest | None = None, cache_groups: int = 512):
        self.ix = icfg(cfg)
        self.man = man or Manifest(self.ix)
        E = self.man.entries
        self.entries = E
        self.c_off = np.array([e["chunk_offset"] for e in E] + [self.man.n_chunks], dtype=np.int64)
        self.d_off = np.array([e["doc_offset"] for e in E] + [self.man.n_docs], dtype=np.int64)
        self._pf: dict[int, pq.ParquetFile] = {}
        self._rg_off: dict[int, np.ndarray] = {}
        self._rg = OrderedDict()
        self._cache_groups = cache_groups
        self._doctab = None

    # --- doc table (lazy, numpy, ~30 bytes/doc)
    def doctab(self):
        if self._doctab is None:
            keys, first, n, lang, ent = [], [], [], [], []
            for i, e in enumerate(self.entries):
                p = self.man.doctab_file(e)
                if not p.exists():
                    continue
                t = pq.read_table(p, columns=["doc_id", "source", "lang", "first", "n"])
                srcs = t["source"].to_pylist()
                keys.append(np.array([_doc_key(d, s if s != "dup" else "corpus") if e.get("active", True) and s != "dup"
                                      else -1 for d, s in zip(t["doc_id"].to_pylist(), srcs)], dtype=np.int64))
                first.append(t["first"].to_numpy().astype(np.int64) + e["chunk_offset"])
                n.append(t["n"].to_numpy().astype(np.int32))
                lang.append(np.array([{"vi": 0, "zh": 1, "en": 2}.get(x, 3) for x in t["lang"].to_pylist()], np.int8))
            if keys:
                K = np.concatenate(keys)
                self._doctab = {"key": K, "first": np.concatenate(first), "n": np.concatenate(n),
                                "lang": np.concatenate(lang)}
                o = np.argsort(K, kind="stable")
                self._doctab["order"] = o
                self._doctab["skey"] = K[o]
            else:
                self._doctab = {"key": np.zeros(0, np.int64), "first": np.zeros(0, np.int64),
                                "n": np.zeros(0, np.int32), "lang": np.zeros(0, np.int8),
                                "order": np.zeros(0, np.int64), "skey": np.zeros(0, np.int64)}
        return self._doctab

    def doc_chunk_range(self, doc_gid: int) -> tuple[int, int]:
        t = self.doctab()
        return int(t["first"][doc_gid]), int(t["n"][doc_gid])

    def doc_gid(self, doc_id: str, source: str = "corpus") -> int | None:
        t = self.doctab()
        k = _doc_key(str(doc_id), source)
        i = np.searchsorted(t["skey"], k)
        if i < len(t["skey"]) and t["skey"][i] == k:
            return int(t["order"][i])
        return None

    def is_active_chunk(self, gid: int) -> bool:
        ei = int(np.searchsorted(self.c_off, gid, side="right") - 1)
        return 0 <= ei < len(self.entries) and self.entries[ei].get("active", True)

    # --- parquet access
    def _file(self, ei: int) -> pq.ParquetFile:
        if ei not in self._pf:
            pf = pq.ParquetFile(self.man.chunk_file(self.entries[ei]))
            self._pf[ei] = pf
            nr = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
            self._rg_off[ei] = np.concatenate([[0], np.cumsum(nr)]).astype(np.int64)
        return self._pf[ei]

    def _group(self, ei: int, rg: int):
        key = (ei, rg)
        if key in self._rg:
            self._rg.move_to_end(key)
            return self._rg[key]
        tab = self._file(ei).read_row_group(rg).to_pydict()
        self._rg[key] = tab
        if len(self._rg) > self._cache_groups:
            self._rg.popitem(last=False)
        return tab

    def get(self, gids) -> list[dict]:
        out = []
        for g in gids:
            g = int(g)
            ei = int(np.searchsorted(self.c_off, g, side="right") - 1)
            row = g - int(self.c_off[ei])
            self._file(ei)
            off = self._rg_off[ei]
            rg = int(np.searchsorted(off, row, side="right") - 1)
            tab = self._group(ei, rg)
            j = row - int(off[rg])
            d = {k: tab[k][j] for k in tab}
            d["gid"] = g
            out.append(d)
        return out

    def iter_entry_chunks(self, ei: int, columns=("text", "lang"), batch_size=8192):
        pf = self._file(ei)
        off = int(self.c_off[ei])
        for rb in pf.iter_batches(batch_size=batch_size, columns=list(columns)):
            yield off, rb.to_pydict()
            off += rb.num_rows


# =============================================================================== builds
def build_bm25(cfg: dict, langs=LANGS, verbose=True):
    """Build missing BM25 shards per language; the trailing unsealed shard is rebuilt to absorb new entries."""
    ix = icfg(cfg)
    bcfg = ix["bm25"]
    man = Manifest(ix)
    store = ChunkStore(cfg, man)
    workers = int(ix["workers"])
    ctx = get_context("fork")
    for lang in langs:
        ldir = _abs(ix["dir"]) / "bm25" / lang
        state = _read_json(ldir / "state.json", {"shards": []})
        covered = {ei for s in state["shards"] for ei in s["entries"]}
        if state["shards"] and not state["shards"][-1]["sealed"]:
            last = state["shards"].pop()
            covered -= set(last["entries"])
            pending = list(last["entries"])
            pending_n = last["n_est"]
        else:
            pending, pending_n = [], 0
        todo = [i for i, e in enumerate(man.entries) if e.get("active", True) and i not in covered and e["n_chunks"]]
        if not todo:
            continue  # nothing new (an unsealed trailing shard stays as is)
        # group entries into shards of <= shard_max_chunks (a single bigger entry gets its own shard);
        # n_est counts all-language chunks, so per-language shards are usually smaller
        groups, cur, cur_n = [], pending, pending_n
        smax = int(bcfg["shard_max_chunks"])
        for ei in todo:
            ne = man.entries[ei]["n_chunks"]
            if cur and cur_n + ne > smax:
                groups.append((cur, cur_n, True))
                cur, cur_n = [], 0
            cur.append(ei)
            cur_n += ne
        if cur:
            groups.append((cur, cur_n, cur_n >= smax))
        with ctx.Pool(workers) as pool:
            for ents, n_est, sealed in groups:
                t0 = time.time()
                jobs, gid_parts = [], []
                for ei in ents:
                    for off, b in store.iter_entry_chunks(ei, columns=("text", "lang"), batch_size=2048):
                        sel = [i for i, l in enumerate(b["lang"]) if l == lang]
                        if not sel:
                            continue
                        jobs.append(([b["text"][i] for i in sel], lang, bcfg))
                        gid_parts.append(off + np.array(sel, dtype=np.int64))
                if not jobs:
                    state["shards"].append({"dir": None, "entries": ents, "n_est": n_est, "sealed": sealed, "n": 0})
                    continue
                res = pool.map(_tok_hash_batch, jobs, chunksize=1)
                lens = np.concatenate([r[0] for r in res])
                nu = np.concatenate([r[1] for r in res])
                terms = np.concatenate([r[2] for r in res])
                tfs = np.concatenate([r[3] for r in res])
                del res, jobs
                sh = BM25Shard.build(lens, nu, terms, tfs, np.concatenate(gid_parts), bcfg)
                del terms, tfs
                name = f"s{ents[0]:05d}-{ents[-1]:05d}"
                sh.save(ldir / name)
                # remove a stale dir of the rebuilt unsealed shard
                state["shards"].append({"dir": name, "entries": ents, "n_est": n_est, "sealed": sealed, "n": sh.n,
                                        "nnz": sh.meta["nnz"]})
                _write_json(ldir / "state.json", state)
                if verbose:
                    print(f"[bm25:{lang}] shard {name}: {sh.n} chunks nnz={sh.meta['nnz']} "
                          f"({time.time() - t0:.1f}s)", flush=True)
        # garbage-collect shard dirs not referenced
        live = {s["dir"] for s in state["shards"]}
        for p in ldir.glob("s*-*"):
            if p.is_dir() and p.name not in live and not p.name.endswith(".tmp"):
                shutil.rmtree(p)
        _write_json(ldir / "state.json", state)


def load_bm25(cfg: dict, lang: str) -> BM25Index | None:
    ix = icfg(cfg)
    ldir = _abs(ix["dir"]) / "bm25" / lang
    state = _read_json(ldir / "state.json", None)
    if not state:
        return None
    shards = [BM25Shard.load(ldir / s["dir"]) for s in state["shards"] if s.get("dir")]
    if not shards:
        return None
    return BM25Index(shards, ix["bm25"])


def _doc_dense_text(title: str, text: str, lead: int) -> str:
    title = (title or "").strip()
    body = (text or "")[:lead]
    if title and body.startswith(title):
        return body
    return (title + "\n" + body).strip()


def build_dense(cfg: dict, langs=LANGS, max_files: int | None = None, verbose=True):
    """Embed new entries per (unit, lang) into fp16 .npy files; train/extend IVF-PQ when large."""
    from src.embed import get_encoder
    ix = icfg(cfg)
    man = Manifest(ix)
    store = ChunkStore(cfg, man)
    enc = None
    lead = int(ix["dense"]["doc_lead_chars"])
    for lang in langs:
        unit = _unit(ix, lang)
        ddir = _abs(ix["dir"]) / "dense" / f"{unit}_{lang}"
        state = _read_json(ddir / "state.json", {"files": {}, "ivf_added": [], "model": None})
        done = 0
        for ei, e in enumerate(man.entries):
            if not e.get("active", True) or str(ei) in state["files"] or not e["n_chunks"]:
                continue
            if max_files is not None and done >= max_files:
                break
            t0 = time.time()
            texts, ids = [], []
            if unit == "chunk":
                for off, b in store.iter_entry_chunks(ei, columns=("text", "lang")):
                    for i, (t, l) in enumerate(zip(b["text"], b["lang"])):
                        if l == lang:
                            texts.append(t)
                            ids.append(off + i)
            else:
                dt = pq.read_table(man.doctab_file(e), columns=["lang", "source", "n"]).to_pydict()
                pf = pq.ParquetFile(_abs(e["doc_file"]))
                row = 0
                for rb in pf.iter_batches(batch_size=1024, columns=["title", "text"]):
                    b = rb.to_pydict()
                    for t_title, t_text in zip(b["title"], b["text"]):
                        if dt["lang"][row] == lang and dt["source"][row] != "dup" and dt["n"][row] > 0:
                            texts.append(_doc_dense_text(t_title, t_text, lead))
                            ids.append(e["doc_offset"] + row)
                        row += 1
            if texts:
                if enc is None:
                    enc = get_encoder(cfg)
                vec = enc.encode(texts, show_progress=verbose and len(texts) > 20000)
                ddir.mkdir(parents=True, exist_ok=True)
                np.save(ddir / f"e{ei:05d}.f16.npy", vec)
                np.save(ddir / f"e{ei:05d}.ids.npy", np.array(ids, dtype=np.int64))
            state["files"][str(ei)] = len(texts)
            state["model"] = enc.name if enc else state.get("model")
            _write_json(ddir / "state.json", state)
            done += 1
            if verbose:
                dt_s = time.time() - t0
                print(f"[dense:{unit}_{lang}] entry {ei}: {len(texts)} vecs in {dt_s:.1f}s "
                      f"({len(texts) / max(dt_s, 1e-6):.1f}/s)", flush=True)
            # bound fp16 disk usage: once large enough, move vectors into IVF-PQ periodically
            if done % int(ix["dense"]["ivf"]["add_every"]) == 0 and \
                    sum(state["files"].values()) >= ix["dense"]["ivf"]["min_vectors"]:
                build_ivf(cfg, lang, verbose=verbose)
                state = _read_json(ddir / "state.json", state)
        n_total = sum(state["files"].values())
        if n_total >= ix["dense"]["ivf"]["min_vectors"]:
            build_ivf(cfg, lang, verbose=verbose)


def build_ivf(cfg: dict, lang: str, verbose=True):
    import faiss
    ix = icfg(cfg)
    iv = ix["dense"]["ivf"]
    unit = _unit(ix, lang)
    ddir = _abs(ix["dir"]) / "dense" / f"{unit}_{lang}"
    state = _read_json(ddir / "state.json", None)
    if not state:
        return
    files = sorted(int(k) for k, v in state["files"].items() if v and (ddir / f"e{int(k):05d}.f16.npy").exists())
    ipath = ddir / "ivf.faiss"
    if ipath.exists():
        index = faiss.read_index(str(ipath))
    else:
        rng = np.random.default_rng(0)
        total = sum(state["files"][str(f)] for f in files)
        frac = min(1.0, iv["train_size"] / max(total, 1))
        sample = []
        for f in files:
            v = np.load(ddir / f"e{f:05d}.f16.npy", mmap_mode="r")
            m = rng.random(len(v)) < frac
            sample.append(np.asarray(v[m], dtype=np.float32))
        xt = np.concatenate(sample)
        d = xt.shape[1]
        if verbose:
            print(f"[ivf:{unit}_{lang}] training {iv['factory']} on {len(xt)} vectors", flush=True)
        t0 = time.time()
        index = faiss.index_factory(d, iv["factory"], faiss.METRIC_INNER_PRODUCT)
        index.train(xt)
        del xt, sample
        if verbose:
            print(f"[ivf] trained in {time.time() - t0:.0f}s", flush=True)
    added = set(state.get("ivf_added", []))
    for f in files:
        if f in added:
            continue
        v = np.asarray(np.load(ddir / f"e{f:05d}.f16.npy"), dtype=np.float32)
        ids = np.load(ddir / f"e{f:05d}.ids.npy")
        index.add_with_ids(v, ids)
        added.add(f)
        state["ivf_added"] = sorted(added)
    faiss.write_index(index, str(ipath) + ".tmp")
    os.replace(str(ipath) + ".tmp", ipath)
    _write_json(ddir / "state.json", state)
    if not iv["keep_fp16"]:
        for f in added:
            p = ddir / f"e{f:05d}.f16.npy"
            if p.exists():
                p.unlink()
    if verbose:
        print(f"[ivf:{unit}_{lang}] ntotal={index.ntotal}", flush=True)


class DenseIndex:
    """Brute-force over fp16 memmaps (+ IVF-PQ for large collections). ids are chunk or doc gids."""

    def __init__(self, ddir: Path, nprobe: int = 64):
        self.ddir = ddir
        self.state = _read_json(ddir / "state.json", {"files": {}, "ivf_added": []})
        added = set(self.state.get("ivf_added", []))
        self.parts = []
        for k, v in sorted(self.state["files"].items(), key=lambda x: int(x[0])):
            f = int(k)
            p = ddir / f"e{f:05d}.f16.npy"
            if v and f not in added and p.exists():
                self.parts.append((np.load(p, mmap_mode="r"), np.load(ddir / f"e{f:05d}.ids.npy")))
        self.ivf = None
        if (ddir / "ivf.faiss").exists():
            import faiss
            self.ivf = faiss.read_index(str(ddir / "ivf.faiss"))
            try:
                faiss.extract_index_ivf(self.ivf).nprobe = nprobe
            except Exception:
                pass

    def __len__(self):
        return sum(len(v) for v, _ in self.parts) + (self.ivf.ntotal if self.ivf is not None else 0)

    def search(self, Q: np.ndarray, k: int = 100, block: int = 262144):
        """Q: [nq, d] float. Score = max over query vectors. Returns [(id, score)] desc."""
        Q = np.asarray(Q, dtype=np.float32)
        best: dict[int, float] = {}
        for vecs, ids in self.parts:
            for s in range(0, len(vecs), block):
                V = np.asarray(vecs[s:s + block], dtype=np.float32)
                sc = (V @ Q.T).max(axis=1)
                kk = min(k, len(sc))
                top = np.argpartition(-sc, kk - 1)[:kk]
                for i in top:
                    g = int(ids[s + i])
                    v = float(sc[i])
                    if v > best.get(g, -1e9):
                        best[g] = v
        if self.ivf is not None and self.ivf.ntotal:
            D, I = self.ivf.search(Q, k)
            for drow, irow in zip(D, I):
                for d, g in zip(drow, irow):
                    if g >= 0 and d > best.get(int(g), -1e9):
                        best[int(g)] = float(d)
        return sorted(best.items(), key=lambda x: -x[1])[:k]


def load_dense(cfg: dict, lang: str) -> DenseIndex | None:
    ix = icfg(cfg)
    ddir = _abs(ix["dir"]) / "dense" / f"{_unit(ix, lang)}_{lang}"
    if not (ddir / "state.json").exists():
        return None
    di = DenseIndex(ddir, nprobe=int(ix["dense"]["ivf"]["nprobe"]))
    return di if len(di) else None


def stats(cfg: dict) -> dict:
    ix = icfg(cfg)
    man = Manifest(ix)
    out = {"entries": len(man.entries), "docs": man.n_docs, "chunks": man.n_chunks, "bm25": {}, "dense": {}}
    for lang in LANGS:
        st = _read_json(_abs(ix["dir"]) / "bm25" / lang / "state.json", None)
        if st:
            out["bm25"][lang] = {"shards": len(st["shards"]), "n": sum(s.get("n", 0) for s in st["shards"]),
                                 "nnz": sum(s.get("nnz", 0) for s in st["shards"])}
        dd = _abs(ix["dir"]) / "dense" / f"{_unit(ix, lang)}_{lang}" / "state.json"
        st = _read_json(dd, None)
        if st:
            out["dense"][f"{_unit(ix, lang)}_{lang}"] = {"files": len(st["files"]), "n": sum(st["files"].values()),
                                                         "ivf_added": len(st.get("ivf_added", []))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["chunk", "bm25", "dense", "ivf", "all", "stats"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--docs", default=None, help="glob of doc parquet files")
    ap.add_argument("--lang", nargs="*", default=list(LANGS))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--index-dir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.index_dir:
        cfg.setdefault("index", {})["dir"] = a.index_dir
    if a.workers:
        cfg.setdefault("index", {})["workers"] = a.workers
    if a.cmd in ("chunk", "all"):
        build_chunks(cfg, a.docs)
    if a.cmd in ("bm25", "all"):
        build_bm25(cfg, a.lang)
    if a.cmd in ("dense", "all"):
        build_dense(cfg, a.lang, max_files=a.max_files)
    if a.cmd == "ivf":
        for l in a.lang:
            build_ivf(cfg, l)
    print(json.dumps(stats(cfg), indent=1))


if __name__ == "__main__":
    sys.exit(main())
