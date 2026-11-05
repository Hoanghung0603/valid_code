"""BM25 over chunks with Tantivy: one on-disk index per language, terms produced here in Python.

Tantivy has no Vietnamese or Chinese analyzer, so each chunk is tokenised in Python and indexed as a
space-joined term string with the `whitespace` tokenizer (Tantivy only does BM25, k1 = 1.2, b = 0.75):
  vi: lower-cased syllables + syllable bigrams ("đái_tháo", "tháo_đường") for multi-syllable words;
  zh: character bigrams of CJK runs (single characters for one-character runs) + lower-cased latin / digit
      words (drug names, "HPV", doses). Bigrams avoid relying on a segmenter's dictionary for medical terms;
  en: lower-cased words (PubMed titles + abstracts, corpus v2); common words are pruned at query time by doc_freq.
Languages get separate indexes because BM25's IDF and average length are corpus statistics; results of the
two indexes are fused by rank (RRF) downstream, never by raw score.
Layout: <index_dir>/bm25/<lang>/ (Tantivy files) + medret_bm25.json (n_docs, tokenizer version).
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TOKENIZER_VERSION = "vi-syl+bi/zh-bi+latin-v1"
EN_TOKENIZER_VERSION = "en-word-v1"
META_FILE = "medret_bm25.json"  # tantivy owns meta.json in the same directory
_CJK = "㐀-䶿一-鿿豈-﫿"
_VI_WORD = re.compile(r"[^\W_]+")
_ZH_PIECE = re.compile(rf"[{_CJK}]+|[^\W_{_CJK}]+")
_CJK_RUN = re.compile(rf"[{_CJK}]+")


def vi_terms(text: str) -> list[str]:
    syl = [s.lower() for s in _VI_WORD.findall(unicodedata.normalize("NFC", text))]
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:], strict=False)]


def zh_terms(text: str) -> list[str]:
    out: list[str] = []
    for piece in _ZH_PIECE.findall(unicodedata.normalize("NFKC", text)):
        if _CJK_RUN.fullmatch(piece):
            out += [piece] if len(piece) == 1 else [piece[i : i + 2] for i in range(len(piece) - 1)]
        else:
            out.append(piece.lower())
    return out


_EN_WORD = re.compile(r"[^\W_]+")


def en_terms(text: str) -> list[str]:
    return [w.lower() for w in _EN_WORD.findall(unicodedata.normalize("NFKC", text))]


TERMS = {"vi": vi_terms, "zh": zh_terms, "en": en_terms}


def tokenizer_version(lang: str) -> str:
    return EN_TOKENIZER_VERSION if lang == "en" else TOKENIZER_VERSION


def _schema():
    import tantivy

    sb = tantivy.SchemaBuilder()
    sb.add_integer_field("chunk_id", stored=True, indexed=False, fast=True)  # python ints arrive as i64
    sb.add_text_field("body", stored=False, tokenizer_name="ws", index_option="freq")
    return sb.build()


def _open(path: Path, create: bool):
    import tantivy

    path.mkdir(parents=True, exist_ok=True)
    index = tantivy.Index(_schema(), path=str(path), reuse=not create)
    index.register_tokenizer("ws", tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.whitespace()).build())
    return index


PART_ROWS = 20_000  # chunks per worker task: bounded IPC payloads (Windows pipes fail on very large results)


_SCOPE: dict[str, frozenset[int]] = {}  # per worker process: scope file -> doc ids


def _in_scope(scope_file: str | None) -> frozenset[int] | None:
    if not scope_file:
        return None
    if scope_file not in _SCOPE:
        import numpy as np

        _SCOPE[scope_file] = frozenset(np.load(scope_file).tolist())
    return _SCOPE[scope_file]


def _file_terms(chunk_file: str, doc_file: str, lang: str, part: int = 0,
                scope_file: str | None = None) -> list[tuple[int, str]]:
    import pyarrow.parquet as pq

    chunks = pq.read_table(chunk_file, columns=["chunk_id", "doc_id", "lang", "start", "end"]).to_pylist()
    chunks = [c for c in chunks[part * PART_ROWS : (part + 1) * PART_ROWS] if c["lang"] == lang]
    scope = _in_scope(scope_file)
    if scope is not None:
        chunks = [c for c in chunks if c["doc_id"] in scope]
    if not chunks:
        return []
    need = {c["doc_id"] for c in chunks}
    docs = {d["doc_id"]: d["text"] for d in pq.read_table(doc_file, columns=["doc_id", "text"]).to_pylist()
            if d["doc_id"] in need}
    fn = TERMS[lang]
    return [(c["chunk_id"], " ".join(fn(docs[c["doc_id"]][c["start"] : c["end"]]))) for c in chunks]


def build_bm25(docs_dir: Path, chunks_dir: Path, out_dir: Path, lang: str, workers: int = 6,
               heap_bytes: int = 1_500_000_000, scope_file: Path | None = None, log=print) -> int:
    """(Re)build the BM25 index of one language from all chunk files (only docs in scope_file, if given)."""
    import shutil

    if out_dir.exists():
        shutil.rmtree(out_dir)
    index = _open(out_dir, create=True)
    writer = index.writer(heap_size=heap_bytes, num_threads=max(1, min(4, heap_bytes // 20_000_000)))  # >=15 MB/thread
    import pyarrow.parquet as pq
    import tantivy

    files = sorted(chunks_dir.glob("*/*.parquet"))
    tasks = [(f, k) for f in files for k in range(-(-pq.ParquetFile(f).metadata.num_rows // PART_ROWS) or 1)]
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = pool.map(_file_terms, [str(f) for f, _ in tasks],
                        [str(docs_dir / f.relative_to(chunks_dir)) for f, _ in tasks], [lang] * len(tasks),
                        [k for _, k in tasks], [str(scope_file) if scope_file else None] * len(tasks), chunksize=1)
        for i, rows in enumerate(futs, 1):
            for cid, body in rows:
                writer.add_document(tantivy.Document(chunk_id=cid, body=body))
            n += len(rows)
            if i % 500 == 0:
                log(f"[bm25 {lang}] {i}/{len(tasks)} tasks, {n:,} chunks, {n / (time.time() - t0):,.0f}/s")
    writer.commit()
    writer.wait_merging_threads()
    (out_dir / META_FILE).write_text(json.dumps({"lang": lang, "n_docs": n, "tokenizer": tokenizer_version(lang),
                                                 "scope": str(scope_file) if scope_file else None}))
    log(f"[bm25 {lang}] {n:,} chunks indexed in {time.time() - t0:,.0f}s")
    return n


class BM25Index:
    def __init__(self, path: Path):
        self.path = path
        meta = json.loads((path / META_FILE).read_text())
        if meta["tokenizer"] != tokenizer_version(meta["lang"]):
            raise ValueError(f"{path} was built with {meta['tokenizer']}, code is {tokenizer_version(meta['lang'])}")
        self.lang = meta["lang"]
        self.index = _open(path, create=False)
        self.searcher = self.index.searcher()

    def search_terms(self, terms: Iterable[str], k: int = 100) -> list[tuple[int, float]]:
        """(chunk_id, bm25 score) of the top-k chunks for an OR query over distinct terms."""
        import tantivy

        uniq = list(dict.fromkeys(t for t in terms if t))
        if not uniq:
            return []
        schema = self.index.schema
        query = tantivy.Query.boolean_query(
            [(tantivy.Occur.Should, tantivy.Query.term_query(schema, "body", t)) for t in uniq]
        )
        res = self.searcher.search(query, k)
        return [(int(self.searcher.doc(addr)["chunk_id"][0]), float(score)) for score, addr in res.hits]

    def prune(self, terms: Iterable[str], max_df: float = 0.1, stop: frozenset[str] = frozenset()) -> list[str]:
        """Distinct query terms minus stopwords and terms found in more than `max_df` of the chunks.

        Patient questions are long and chatty; without pruning, "có" (83% of vi chunks), "của", "bác sĩ", 需要, 检查
        outweigh the few informative terms ("scc": 355 chunks) by sheer number.
        """
        limit = max_df * self.searcher.num_docs
        out = []
        for t in dict.fromkeys(terms):
            if t in stop or all(p in stop for p in t.split("_")):  # a bigram goes only if both syllables are stop
                continue
            if self.searcher.doc_freq("body", t) <= limit:
                out.append(t)
        return out

    def search(self, text: str, k: int = 100, max_df: float | None = None,
               stop: frozenset[str] = frozenset()) -> list[tuple[int, float]]:
        terms = TERMS[self.lang](text)
        if max_df is not None:
            terms = self.prune(terms, max_df, stop) or terms  # all terms common: fall back to the full query
        return self.search_terms(terms, k)
