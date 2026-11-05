"""Retrieve and audit document rankings from full BGE and Qwen3 indexes."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
import statistics
import sys

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.submission_io import iter_json_array, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "submission_exp106_addonly_bge4.json")
    parser.add_argument("--active-index", type=Path, default=ROOT / "data/indices/new_dataset/index")
    parser.add_argument("--qwen-index", type=Path, default=ROOT / "data/indices/new_dataset/qwen3-0.6b/index")
    parser.add_argument("--bge-queries", type=Path, default=ROOT / "data/indices/new_dataset/.cache/query_encodings_1200.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/qwen3_0p6b_retrieval")
    parser.add_argument("--chunk-k", type=int, default=5_000)
    parser.add_argument("--doc-k", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    return parser.parse_args()


def collapse(
    rows: np.ndarray,
    scores: np.ndarray,
    row_doc_ids: np.ndarray,
    doc_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    result_docs = np.full((len(rows), doc_k), -1, dtype=np.int64)
    result_scores = np.full((len(rows), doc_k), -np.inf, dtype=np.float32)
    for query_index, (query_rows, query_scores) in enumerate(zip(rows, scores)):
        seen: set[int] = set()
        count = 0
        for row, score in zip(query_rows, query_scores):
            if int(row) < 0:
                continue
            doc_id = int(row_doc_ids[int(row)])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            result_docs[query_index, count] = doc_id
            result_scores[query_index, count] = float(score)
            count += 1
            if count == doc_k:
                break
        if count != doc_k:
            raise RuntimeError(f"Query row {query_index} produced only {count} documents")
    return result_docs, result_scores


def retrieve(
    label: str,
    index_path: Path,
    queries: np.ndarray,
    row_doc_ids: np.ndarray,
    chunk_k: int,
    doc_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.ascontiguousarray(queries, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise RuntimeError(f"Invalid query vectors for {label}")
    faiss.normalize_L2(vectors)
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
    if index.ntotal != len(row_doc_ids) or index.d != vectors.shape[1]:
        raise RuntimeError(f"Index/query mismatch for {label}: {index.ntotal}/{index.d}")
    print(f"search {label}: {len(vectors)} queries x top-{chunk_k}", flush=True)
    chunk_scores, chunk_rows = index.search(vectors, chunk_k)
    del index, vectors
    docs, doc_scores = collapse(chunk_rows, chunk_scores, row_doc_ids, doc_k)
    del chunk_rows, chunk_scores
    print(f"collapsed {label}: {docs.shape}", flush=True)
    return docs, doc_scores


def values(values: list[int | float]) -> dict:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def overlap(left: np.ndarray, right: np.ndarray, depth: int) -> dict:
    counts = [len(set(a[:depth].tolist()) & set(b[:depth].tolist())) for a, b in zip(left, right)]
    return {"depth": depth, "intersection": values(counts), "mean_fraction": statistics.fmean(counts) / depth}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(args.threads)

    baseline_ids: list[int] = []
    baseline_docs: list[list[int]] = []
    for item in iter_json_array(args.baseline):
        baseline_ids.append(int(item["id"]))
        baseline_docs.append(list(map(int, item["relevant_docs"])))
    if len(baseline_ids) != 1_200 or len(set(baseline_ids)) != 1_200:
        raise RuntimeError("Expected 1,200 unique baseline queries")
    baseline_docs_array = np.asarray(baseline_docs, dtype=np.int64)
    baseline_ids_array = np.asarray(baseline_ids, dtype=np.int64)

    metadata = pd.read_parquet(args.active_index / "chunks_metadata.parquet", columns=["chunk_id", "doc_id"])
    if not np.array_equal(metadata.chunk_id.to_numpy(), np.arange(len(metadata), dtype=np.int64)):
        raise RuntimeError("Active chunk_id is not row-aligned")
    row_doc_ids = metadata.doc_id.to_numpy(dtype=np.int64, copy=False)

    qwen_path = args.qwen_index / "query_embeddings.npz"
    with np.load(qwen_path, allow_pickle=False) as qwen:
        if not np.array_equal(qwen["query_id"], baseline_ids_array):
            raise RuntimeError("Qwen query IDs/order differ from baseline")
        qwen_queries = {language: qwen[language].astype(np.float32) for language in ("vi", "en", "zh")}

    with args.bge_queries.open("rb") as handle:
        bge = pickle.load(handle)
    if not np.array_equal(np.asarray(bge["query_ids"], dtype=np.int64), baseline_ids_array):
        raise RuntimeError("BGE query IDs/order differ from baseline")

    rankings: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for language in ("vi", "en", "zh"):
        rankings[f"qwen_{language}"], scores[f"qwen_{language}"] = retrieve(
            f"qwen_{language}",
            args.qwen_index / "faiss_index_flat_ip.bin",
            qwen_queries[language],
            row_doc_ids,
            args.chunk_k,
            args.doc_k,
        )
    del qwen_queries
    rankings["bge_vi"], scores["bge_vi"] = retrieve(
        "bge_vi",
        args.active_index / "faiss_index_flat_ip.bin",
        np.asarray(bge["dense_q"], dtype=np.float32),
        row_doc_ids,
        args.chunk_k,
        args.doc_k,
    )

    cache_path = args.output_dir / "doc_rankings_top1000.npz"
    np.savez(
        cache_path,
        query_ids=baseline_ids_array,
        baseline_docs=baseline_docs_array,
        **{f"{label}_docs": ranking for label, ranking in rankings.items()},
        **{f"{label}_scores": score for label, score in scores.items()},
    )

    audit = {
        "baseline": args.baseline.name,
        "queries": len(baseline_ids),
        "baseline_docs_per_query": int(baseline_docs_array.shape[1]),
        "chunk_k": args.chunk_k,
        "doc_k": args.doc_k,
        "threads": args.threads,
        "overlap_with_baseline": {
            label: {
                str(depth): overlap(baseline_docs_array, ranking, depth)
                for depth in (50, 150, baseline_docs_array.shape[1])
            }
            for label, ranking in rankings.items()
        },
        "qwen_language_overlap": {
            "vi_en": overlap(rankings["qwen_vi"], rankings["qwen_en"], baseline_docs_array.shape[1]),
            "vi_zh": overlap(rankings["qwen_vi"], rankings["qwen_zh"], baseline_docs_array.shape[1]),
            "en_zh": overlap(rankings["qwen_en"], rankings["qwen_zh"], baseline_docs_array.shape[1]),
        },
        "cache": str(cache_path),
    }
    write_json(args.output_dir / "retrieval_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
