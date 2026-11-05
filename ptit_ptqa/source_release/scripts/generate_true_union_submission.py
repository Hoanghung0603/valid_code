"""Generate an independent-candidate, multilingual RRF submission.

Unlike the legacy hybrid pipeline, every retrieval branch searches the full
index independently before fusion:

* BGE-M3 dense Vietnamese
* BGE-M3 dense English translation
* BGE-M3 dense Chinese translation
* BGE-M3 multilingual sparse lexical

An old SQLite FTS5 document index can optionally be added as a low-weight BM25
branch.  It is deliberately opt-in because it was built from a stale corpus
and performed poorly in Exp20.

All expensive stages are cached under ``index/.cache/exp41_true_union`` so the
script is safe to resume after interruption.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.query_encoder import BGEM3QueryEncoder  # noqa: E402


LOGGER = logging.getLogger("exp41_true_union")
BRANCHES = ("dense_vi", "dense_en", "dense_zh", "sparse")

# Conservative weights: translated queries may rescue foreign-language pages,
# but should not displace strong Vietnamese results on their own.
DEFAULT_WEIGHTS = {
    "dense_vi": 1.00,
    "sparse": 0.90,
    "dense_zh": 0.45,
    "dense_en": 0.35,
    "bm25": 0.20,
}

VI_STOPWORDS = {
    "ai", "bị", "bởi", "các", "cái", "cho", "chúng", "có", "của", "do",
    "đã", "đang", "đâu", "đây", "đến", "để", "được", "gì", "hay", "hơn",
    "khi", "không", "là", "lại", "làm", "mà", "mình", "một", "nào", "này",
    "nên", "những", "như", "nhưng", "ở", "phải", "ra", "rằng", "rất", "sao",
    "sẽ", "sau", "thì", "trên", "trong", "tôi", "và", "vào", "với", "vừa",
    "em", "xin", "hỏi", "bác", "sĩ", "ạ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "data/indices/new_dataset/index",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=ROOT / "data/raw/btc_provided/query_translated.parquet",
    )
    parser.add_argument(
        "--official-corpus",
        type=Path,
        default=ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet",
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=ROOT / "data/indices/new_dataset/.cache/query_encodings_1200.pkl",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=ROOT / "submission_exp41_true_union_rrf",
    )
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--top-docs", type=int, default=100)
    parser.add_argument("--top-chunks", type=int, default=98)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--faiss-threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--zh-sparse-weight", type=float, default=1.0)
    parser.add_argument("--en-sparse-weight", type=float, default=0.5)
    parser.add_argument("--dense-vi-weight", type=float, default=DEFAULT_WEIGHTS["dense_vi"])
    parser.add_argument("--dense-en-weight", type=float, default=DEFAULT_WEIGHTS["dense_en"])
    parser.add_argument("--dense-zh-weight", type=float, default=DEFAULT_WEIGHTS["dense_zh"])
    parser.add_argument("--sparse-weight", type=float, default=DEFAULT_WEIGHTS["sparse"])
    parser.add_argument("--include-legacy-bm25", action="store_true")
    parser.add_argument(
        "--fts-db",
        type=Path,
        default=ROOT / "data/raw/final_corpus_text/corpus_fts.db",
    )
    parser.add_argument("--bm25-weight", type=float, default=DEFAULT_WEIGHTS["bm25"])
    parser.add_argument("--bm25-max-terms", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Ignore retrieval caches and recompute.")
    return parser.parse_args()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_queries(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"id", "query", "query_en", "query_zh"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing query columns: {sorted(missing)}")
    if frame["id"].duplicated().any():
        raise ValueError("Query IDs are not unique")
    for col in ("query", "query_en", "query_zh"):
        frame[col] = frame[col].fillna("").astype(str)
    return frame.reset_index(drop=True)


def load_encoding_cache(path: Path, query_ids: np.ndarray) -> Mapping[str, object]:
    with path.open("rb") as handle:
        cached = pickle.load(handle)
    cached_ids = np.asarray(cached["query_ids"], dtype=np.int64)
    if not np.array_equal(cached_ids, query_ids):
        raise ValueError("Query encoding cache IDs/order do not match query file")
    if np.asarray(cached["dense_q"]).shape[0] != len(query_ids):
        raise ValueError("Vietnamese dense cache has the wrong number of rows")
    return cached


def load_or_encode_dense_queries(
    queries: pd.DataFrame,
    base_cache: Mapping[str, object],
    cache_path: Path,
    batch_size: int,
    force: bool,
) -> Dict[str, np.ndarray]:
    query_ids = queries["id"].to_numpy(dtype=np.int64)
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if np.array_equal(cached["query_ids"], query_ids):
            LOGGER.info("Loaded multilingual dense query cache: %s", cache_path)
            return {name: cached[name].astype(np.float32) for name in ("dense_vi", "dense_en", "dense_zh")}
        LOGGER.warning("Ignoring multilingual dense cache with mismatched query IDs")

    encoder = BGEM3QueryEncoder(use_fp16=True)
    dense_vi = np.asarray(base_cache["dense_q"], dtype=np.float32)
    LOGGER.info("Encoding %d English translated queries", len(queries))
    dense_en = encoder.encode_dense(queries["query_en"].tolist(), batch_size=batch_size, max_length=1024)
    LOGGER.info("Encoding %d Chinese translated queries", len(queries))
    dense_zh = encoder.encode_dense(queries["query_zh"].tolist(), batch_size=batch_size, max_length=1024)
    encoder.unload()
    atomic_save_npz(
        cache_path,
        query_ids=query_ids,
        dense_vi=dense_vi,
        dense_en=dense_en,
        dense_zh=dense_zh,
    )
    LOGGER.info("Saved multilingual dense query cache: %s", cache_path)
    return {"dense_vi": dense_vi, "dense_en": dense_en, "dense_zh": dense_zh}


def load_or_search_dense(
    index_path: Path,
    dense_queries: Mapping[str, np.ndarray],
    query_ids: np.ndarray,
    candidate_k: int,
    cache_path: Path,
    threads: int,
    force: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["query_ids"], query_ids)
            and int(cached["candidate_k"]) == candidate_k
        ):
            LOGGER.info("Loaded dense retrieval cache: %s", cache_path)
            return {
                name: (cached[f"{name}_rows"], cached[f"{name}_scores"])
                for name in ("dense_vi", "dense_en", "dense_zh")
            }
        LOGGER.warning("Ignoring incompatible dense retrieval cache")

    faiss.omp_set_num_threads(threads)
    LOGGER.info("Memory-mapping FAISS index: %s", index_path)
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
    k = min(candidate_k, index.ntotal)
    saved: Dict[str, np.ndarray] = {
        "query_ids": query_ids,
        "candidate_k": np.asarray(candidate_k, dtype=np.int64),
    }
    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in ("dense_vi", "dense_en", "dense_zh"):
        vectors = np.array(dense_queries[name], dtype=np.float32, copy=True, order="C")
        faiss.normalize_L2(vectors)
        started = time.time()
        scores, rows = index.search(vectors, k)
        LOGGER.info("Independent %s full-index search finished in %.1fs", name, time.time() - started)
        rows = rows.astype(np.int32)
        scores = scores.astype(np.float32)
        results[name] = (rows, scores)
        saved[f"{name}_rows"] = rows
        saved[f"{name}_scores"] = scores
        del vectors
        gc.collect()

    atomic_save_npz(cache_path, **saved)
    LOGGER.info("Saved dense retrieval cache: %s", cache_path)
    del index
    gc.collect()
    return results


def fused_sparse_weights(
    cached: Mapping[str, object],
    zh_weight: float,
    en_weight: float,
) -> List[Dict[int, float]]:
    output: List[Dict[int, float]] = []
    for vi, zh, en in zip(cached["lex_vi"], cached["lex_zh"], cached["lex_en"]):
        merged = {int(token): float(weight) for token, weight in vi.items()}
        for token, weight in zh.items():
            token_id = int(token)
            merged[token_id] = max(merged.get(token_id, 0.0), float(weight) * zh_weight)
        for token, weight in en.items():
            token_id = int(token)
            merged[token_id] = max(merged.get(token_id, 0.0), float(weight) * en_weight)
        output.append(merged)
    return output


def load_or_search_sparse(
    sparse_path: Path,
    query_weights: Sequence[Mapping[int, float]],
    query_ids: np.ndarray,
    candidate_k: int,
    cache_path: Path,
    force: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["query_ids"], query_ids)
            and int(cached["candidate_k"]) == candidate_k
        ):
            LOGGER.info("Loaded sparse retrieval cache: %s", cache_path)
            return cached["rows"], cached["scores"]
        LOGGER.warning("Ignoring incompatible sparse retrieval cache")

    started = time.time()
    LOGGER.info("Loading BGE-M3 sparse matrix: %s", sparse_path)
    csr = sp.load_npz(sparse_path)
    LOGGER.info("Converting sparse matrix to CSC for full-corpus posting-list retrieval")
    csc = csr.tocsc()
    del csr
    gc.collect()
    LOGGER.info("CSC ready in %.1fs", time.time() - started)

    n_queries = len(query_weights)
    n_rows, n_vocab = csc.shape
    k = min(candidate_k, n_rows)
    top_rows = np.full((n_queries, k), -1, dtype=np.int32)
    top_scores = np.full((n_queries, k), -np.inf, dtype=np.float32)

    for q_idx, weights in enumerate(query_weights):
        scores = np.zeros(n_rows, dtype=np.float32)
        for token_id, query_weight in weights.items():
            if token_id < 0 or token_id >= n_vocab or query_weight == 0:
                continue
            left = csc.indptr[token_id]
            right = csc.indptr[token_id + 1]
            posting_rows = csc.indices[left:right]
            scores[posting_rows] += csc.data[left:right] * np.float32(query_weight)

        if k == n_rows:
            chosen = np.argsort(scores)[::-1]
        else:
            chosen = np.argpartition(scores, -k)[-k:]
            chosen = chosen[np.argsort(scores[chosen])[::-1]]
        positive = chosen[scores[chosen] > 0]
        take = min(k, len(positive))
        top_rows[q_idx, :take] = positive[:take].astype(np.int32)
        top_scores[q_idx, :take] = scores[positive[:take]]

        if (q_idx + 1) % 50 == 0 or q_idx + 1 == n_queries:
            elapsed = time.time() - started
            LOGGER.info("Sparse full-index retrieval: %d/%d queries (%.1f q/s)", q_idx + 1, n_queries, (q_idx + 1) / elapsed)

    atomic_save_npz(
        cache_path,
        query_ids=query_ids,
        candidate_k=np.asarray(candidate_k, dtype=np.int64),
        rows=top_rows,
        scores=top_scores,
    )
    LOGGER.info("Saved sparse retrieval cache: %s", cache_path)
    del csc
    gc.collect()
    return top_rows, top_scores


def bm25_terms(text: str, max_terms: int) -> List[str]:
    terms: List[str] = []
    seen = set()
    for term in re.findall(r"(?u)\w+", text.lower()):
        if len(term) <= 1 or term.isdigit() or term in VI_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def load_or_search_legacy_bm25(
    db_path: Path,
    queries: pd.DataFrame,
    official_doc_ids: set[int],
    candidate_k: int,
    max_terms: int,
    cache_path: Path,
    force: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    query_ids = queries["id"].to_numpy(dtype=np.int64)
    if cache_path.exists() and not force:
        cached = np.load(cache_path)
        if (
            np.array_equal(cached["query_ids"], query_ids)
            and int(cached["candidate_k"]) == candidate_k
            and int(cached["max_terms"]) == max_terms
        ):
            LOGGER.info("Loaded legacy BM25 retrieval cache: %s", cache_path)
            return cached["doc_ids"], cached["scores"]

    n_queries = len(queries)
    doc_ids = np.full((n_queries, candidate_k), -1, dtype=np.int64)
    scores = np.full((n_queries, candidate_k), -np.inf, dtype=np.float32)
    uri = db_path.resolve().as_uri().replace("file:///", "file:/") + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    started = time.time()
    try:
        for q_idx, query in enumerate(queries["query"]):
            terms = bm25_terms(query, max_terms)
            if not terms:
                continue
            expression = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
            # Oversample because the stale FTS database also contains out-of-corpus rows.
            rows = connection.execute(
                "SELECT id, bm25(corpus) AS score FROM corpus "
                "WHERE corpus MATCH ? ORDER BY score LIMIT ?",
                (expression, candidate_k * 2),
            ).fetchall()
            kept = [(int(doc_id), -float(score)) for doc_id, score in rows if int(doc_id) in official_doc_ids]
            kept = kept[:candidate_k]
            if kept:
                doc_ids[q_idx, : len(kept)] = [item[0] for item in kept]
                scores[q_idx, : len(kept)] = [item[1] for item in kept]
            if (q_idx + 1) % 25 == 0 or q_idx + 1 == n_queries:
                LOGGER.info("Legacy BM25: %d/%d queries (%.2f q/s)", q_idx + 1, n_queries, (q_idx + 1) / (time.time() - started))
    finally:
        connection.close()

    atomic_save_npz(
        cache_path,
        query_ids=query_ids,
        candidate_k=np.asarray(candidate_k, dtype=np.int64),
        max_terms=np.asarray(max_terms, dtype=np.int64),
        doc_ids=doc_ids,
        scores=scores,
    )
    return doc_ids, scores


def add_branch_rrf(
    chunk_scores: Dict[int, float],
    doc_scores: Dict[int, float],
    rows: np.ndarray,
    row_doc_ids: np.ndarray,
    weight: float,
    rrf_k: int,
) -> None:
    seen_rows = set()
    seen_docs = set()
    doc_rank = 0
    for rank, row in enumerate(rows, start=1):
        row = int(row)
        if row < 0 or row in seen_rows:
            continue
        seen_rows.add(row)
        doc_id = int(row_doc_ids[row])
        chunk_scores[row] += weight / (rrf_k + rank)
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            doc_rank += 1
            doc_scores[doc_id] += weight / (rrf_k + doc_rank)


def fuse_rankings(
    branch_rows: Mapping[str, np.ndarray],
    row_doc_ids: np.ndarray,
    weights: Mapping[str, float],
    rrf_k: int,
    top_docs: int,
    chunk_buffer: int,
    bm25_docs: np.ndarray | None,
) -> Tuple[List[List[int]], List[List[int]]]:
    n_queries = next(iter(branch_rows.values())).shape[0]
    fused_docs: List[List[int]] = []
    fused_chunk_rows: List[List[int]] = []
    for q_idx in range(n_queries):
        chunk_scores: Dict[int, float] = defaultdict(float)
        doc_scores: Dict[int, float] = defaultdict(float)
        for name in BRANCHES:
            add_branch_rrf(
                chunk_scores,
                doc_scores,
                branch_rows[name][q_idx],
                row_doc_ids,
                weights[name],
                rrf_k,
            )
        if bm25_docs is not None and weights.get("bm25", 0.0) > 0:
            seen = set()
            rank = 0
            for doc_id in bm25_docs[q_idx]:
                doc_id = int(doc_id)
                if doc_id < 0 or doc_id in seen:
                    continue
                seen.add(doc_id)
                rank += 1
                doc_scores[doc_id] += weights["bm25"] / (rrf_k + rank)

        ranked_docs = sorted(doc_scores, key=lambda doc_id: (-doc_scores[doc_id], doc_id))
        ranked_rows = sorted(chunk_scores, key=lambda row: (-chunk_scores[row], row))
        fused_docs.append(ranked_docs[:top_docs])
        fused_chunk_rows.append(ranked_rows[:chunk_buffer])
    return fused_docs, fused_chunk_rows


def load_candidate_texts(text_dir: Path, needed_doc_ids: set[int]) -> Dict[int, str]:
    texts: Dict[int, str] = {}
    shards = sorted(text_dir.glob("part_*.parquet"))
    started = time.time()
    for shard_idx, shard in enumerate(shards, start=1):
        frame = pd.read_parquet(shard, columns=["doc_id", "text"])
        subset = frame[frame["doc_id"].isin(needed_doc_ids)]
        for row in subset.itertuples(index=False):
            texts[int(row.doc_id)] = str(row.text)
        if shard_idx % 10 == 0 or shard_idx == len(shards):
            LOGGER.info("Text loading: %d/%d shards, %d/%d docs", shard_idx, len(shards), len(texts), len(needed_doc_ids))
        if len(texts) == len(needed_doc_ids):
            break
    LOGGER.info("Loaded candidate text in %.1fs", time.time() - started)
    return texts


def materialize_submission(
    query_ids: np.ndarray,
    ranked_docs: Sequence[Sequence[int]],
    ranked_rows: Sequence[Sequence[int]],
    row_doc_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    texts: Mapping[int, str],
    top_docs: int,
    top_chunks: int,
) -> List[dict]:
    submission: List[dict] = []
    for q_idx, query_id in enumerate(query_ids):
        chunks = []
        seen_spans = set()
        seen_chunk_texts = set()
        for row in ranked_rows[q_idx]:
            row = int(row)
            doc_id = int(row_doc_ids[row])
            start = int(starts[row])
            end = int(ends[row])
            span = (doc_id, start, end)
            if span in seen_spans:
                continue
            full_text = texts.get(doc_id, "")
            chunk_text = full_text[start:end] if full_text else ""
            if not chunk_text.strip():
                continue
            # Two legacy rows can occasionally point at different spans that
            # materialize to exactly the same text. Keep only one copy per doc.
            text_key = (doc_id, chunk_text)
            if text_key in seen_chunk_texts:
                continue
            seen_spans.add(span)
            seen_chunk_texts.add(text_key)
            chunks.append({"doc_id": doc_id, "chunk_text": chunk_text})
            if len(chunks) >= top_chunks:
                break
        submission.append(
            {
                "id": int(query_id),
                "relevant_docs": [int(doc_id) for doc_id in ranked_docs[q_idx][:top_docs]],
                "relevant_chunks": chunks,
            }
        )
    return submission


def validate_submission(
    submission: Sequence[dict],
    expected_query_ids: np.ndarray,
    official_doc_ids: set[int],
    source_texts: Mapping[int, str],
    top_docs: int,
    top_chunks: int,
) -> dict:
    errors: List[str] = []
    actual_ids = np.asarray([item.get("id") for item in submission], dtype=np.int64)
    if not np.array_equal(actual_ids, expected_query_ids):
        errors.append("Query IDs/order differ from official query file")
    for item in submission:
        query_id = item["id"]
        docs = item["relevant_docs"]
        chunks = item["relevant_chunks"]
        if len(docs) > top_docs or len(set(docs)) != len(docs):
            errors.append(f"Query {query_id}: invalid/duplicate relevant_docs")
        if any(not isinstance(doc_id, int) or doc_id not in official_doc_ids for doc_id in docs):
            errors.append(f"Query {query_id}: relevant_docs contains invalid corpus ID")
        seen_chunks = set()
        if len(chunks) > top_chunks:
            errors.append(f"Query {query_id}: too many chunks")
        for chunk in chunks:
            doc_id = chunk.get("doc_id")
            text = chunk.get("chunk_text")
            if not isinstance(doc_id, int) or doc_id not in official_doc_ids:
                errors.append(f"Query {query_id}: chunk has invalid corpus ID")
                break
            if not isinstance(text, str) or not text.strip() or text not in source_texts.get(doc_id, ""):
                errors.append(f"Query {query_id}: chunk is empty or not verbatim source text")
                break
            key = (doc_id, text)
            if key in seen_chunks:
                errors.append(f"Query {query_id}: duplicate chunk")
                break
            seen_chunks.add(key)
    if errors:
        raise ValueError("Submission validation failed:\n" + "\n".join(errors[:20]))
    return {
        "queries": len(submission),
        "docs_min": min(len(item["relevant_docs"]) for item in submission),
        "docs_max": max(len(item["relevant_docs"]) for item in submission),
        "chunks_min": min(len(item["relevant_chunks"]) for item in submission),
        "chunks_max": max(len(item["relevant_chunks"]) for item in submission),
    }


def export_submission(submission: Sequence[dict], out_prefix: Path) -> Tuple[Path, Path, float]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    zip_path = out_prefix.with_suffix(".zip")
    max_chunks = max(len(item["relevant_chunks"]) for item in submission)
    # Use decimal bytes here. A 96.7 MiB archive is already >100,000,000 bytes
    # and may be rejected by dashboards that interpret the rule as SI MB.
    target_bytes = 99_500_000
    while True:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(submission, handle, ensure_ascii=False, separators=(",", ":"))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(json_path, arcname="submission.json")
        size_bytes = zip_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        if size_bytes < target_bytes:
            break
        max_chunks -= 2
        if max_chunks < 10:
            raise ValueError(f"Could not reduce submission below size limit; last ZIP was {size_mb:.2f} MB")
        LOGGER.warning(
            "ZIP is %.2f MiB (%d bytes); trimming all queries to at most %d chunks",
            size_mb,
            size_bytes,
            max_chunks,
        )
        for item in submission:
            item["relevant_chunks"] = item["relevant_chunks"][:max_chunks]
    return json_path, zip_path, size_mb


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.time()
    cache_dir = args.index_dir / ".cache" / "exp41_true_union"
    cache_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.query_path)
    query_ids = queries["id"].to_numpy(dtype=np.int64)
    base_cache = load_encoding_cache(args.query_cache, query_ids)
    official = pd.read_parquet(args.official_corpus, columns=["id"])
    official_doc_ids = set(official["id"].astype(np.int64).tolist())
    LOGGER.info("Loaded %d queries and %d official corpus IDs", len(queries), len(official_doc_ids))

    dense_queries = load_or_encode_dense_queries(
        queries,
        base_cache,
        cache_dir / "dense_queries_vi_en_zh.npz",
        args.encode_batch_size,
        args.force,
    )
    dense = load_or_search_dense(
        args.index_dir / "faiss_index_flat_ip.bin",
        dense_queries,
        query_ids,
        args.candidate_k,
        cache_dir / f"dense_top{args.candidate_k}.npz",
        args.faiss_threads,
        args.force,
    )
    del dense_queries
    gc.collect()

    sparse_weights = fused_sparse_weights(base_cache, args.zh_sparse_weight, args.en_sparse_weight)
    sparse_rows, _ = load_or_search_sparse(
        args.index_dir / "sparse_csr.npz",
        sparse_weights,
        query_ids,
        args.candidate_k,
        cache_dir / f"sparse_top{args.candidate_k}_zh{args.zh_sparse_weight:g}_en{args.en_sparse_weight:g}.npz",
        args.force,
    )
    del sparse_weights
    gc.collect()

    bm25_docs = None
    if args.include_legacy_bm25:
        LOGGER.warning("Including stale Exp20 SQLite FTS5 BM25 branch at low weight %.2f", args.bm25_weight)
        bm25_docs, _ = load_or_search_legacy_bm25(
            args.fts_db,
            queries,
            official_doc_ids,
            args.candidate_k,
            args.bm25_max_terms,
            cache_dir / f"legacy_bm25_top{args.candidate_k}_terms{args.bm25_max_terms}.npz",
            args.force,
        )

    metadata = pd.read_parquet(args.index_dir / "chunks_metadata.parquet")
    row_doc_ids = metadata["doc_id"].to_numpy(dtype=np.int64)
    starts = metadata["start"].to_numpy(dtype=np.int64)
    ends = metadata["end"].to_numpy(dtype=np.int64)
    branch_rows = {
        "dense_vi": dense["dense_vi"][0],
        "dense_en": dense["dense_en"][0],
        "dense_zh": dense["dense_zh"][0],
        "sparse": sparse_rows,
    }
    weights = {
        "dense_vi": args.dense_vi_weight,
        "dense_en": args.dense_en_weight,
        "dense_zh": args.dense_zh_weight,
        "sparse": args.sparse_weight,
        "bm25": args.bm25_weight,
    }
    ranked_docs, ranked_rows = fuse_rankings(
        branch_rows,
        row_doc_ids,
        weights,
        args.rrf_k,
        args.top_docs,
        max(args.top_chunks * 2, args.top_chunks + 20),
        bm25_docs,
    )
    needed_doc_ids = {
        int(row_doc_ids[row])
        for query_rows in ranked_rows
        for row in query_rows
        if int(row) >= 0
    }
    texts = load_candidate_texts(args.index_dir / "text", needed_doc_ids)
    submission = materialize_submission(
        query_ids,
        ranked_docs,
        ranked_rows,
        row_doc_ids,
        starts,
        ends,
        texts,
        args.top_docs,
        args.top_chunks,
    )
    validate_submission(
        submission,
        query_ids,
        official_doc_ids,
        texts,
        args.top_docs,
        args.top_chunks,
    )
    json_path, zip_path, size_mb = export_submission(submission, args.out_prefix)
    # Export may trim chunk counts to satisfy the byte-size limit.
    stats = validate_submission(
        submission,
        query_ids,
        official_doc_ids,
        texts,
        args.top_docs,
        args.top_chunks,
    )
    manifest = {
        "experiment": "exp41_true_union_rrf",
        "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "index_dir": str(args.index_dir),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "weights": weights,
        "legacy_bm25_included": bool(args.include_legacy_bm25),
        "stats": stats,
        "zip_size_mb": size_mb,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Validation passed: %s", stats)
    LOGGER.info("Submission JSON: %s", json_path)
    LOGGER.info("Submission ZIP: %s (%.2f MB)", zip_path, size_mb)
    LOGGER.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
