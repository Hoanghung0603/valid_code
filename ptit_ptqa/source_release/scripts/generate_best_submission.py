"""Apply the Exp50-winning output policies to a freshly retrieved baseline.

The document axis keeps baseline ranking and fills to 125 unique documents
from the same index's Vietnamese dense top-500. The chunk axis merges nearby
same-document spans when their gap is at most 2,000 characters and the final
verbatim envelope is at most 7,000 characters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "submission_exp51_champion_full1205k_base.json",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "data/indices/new_dataset/index",
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=ROOT / "data/indices/new_dataset/.cache/query_encodings_1200.pkl",
    )
    parser.add_argument(
        "--official-corpus",
        type=Path,
        default=ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=ROOT / "submission_exp51_full1205k_docs125_merged_envelopes",
    )
    parser.add_argument("--dense-k", type=int, default=500)
    parser.add_argument("--top-docs", type=int, default=125)
    parser.add_argument("--max-gap", type=int, default=2_000)
    parser.add_argument("--max-span", type=int, default=7_000)
    return parser.parse_args()


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dense_queries(path: Path, query_ids: np.ndarray) -> np.ndarray:
    with path.open("rb") as handle:
        cached = pickle.load(handle)
    cached_ids = np.asarray(cached["query_ids"], dtype=np.int64)
    if not np.array_equal(cached_ids, query_ids):
        raise RuntimeError("Query cache IDs/order do not match baseline")
    vectors = np.array(cached["dense_q"], dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(vectors)
    return vectors


def extend_documents(
    baseline: list[dict], row_doc_ids: np.ndarray, dense_rows: np.ndarray, top_docs: int
) -> list[list[int]]:
    output: list[list[int]] = []
    for item, rows in zip(baseline, dense_rows):
        docs = [int(doc_id) for doc_id in item["relevant_docs"]]
        seen = set(docs)
        for row in rows:
            doc_id = int(row_doc_ids[int(row)])
            if doc_id not in seen:
                docs.append(doc_id)
                seen.add(doc_id)
            if len(docs) >= top_docs:
                break
        if len(docs) < top_docs:
            raise RuntimeError(f"Query {item['id']} has only {len(docs)} unique docs")
        output.append(docs[:top_docs])
    return output


def load_needed_texts(index_dir: Path, baseline: list[dict]) -> dict[int, str]:
    needed = {
        int(chunk["doc_id"])
        for item in baseline
        for chunk in item["relevant_chunks"]
    }
    texts: dict[int, str] = {}
    for path in sorted((index_dir / "text").glob("part_*.parquet")):
        frame = pd.read_parquet(path, columns=["doc_id", "text"])
        subset = frame[frame["doc_id"].isin(needed)]
        texts.update(
            (int(row.doc_id), str(row.text)) for row in subset.itertuples(index=False)
        )
        if len(texts) == len(needed):
            break
    missing = needed - set(texts)
    if missing:
        raise RuntimeError(f"Missing source text for {len(missing):,} documents")
    return texts


def locate(text: str, chunk: str) -> tuple[int, int]:
    start = text.find(chunk)
    if start < 0:
        raise RuntimeError("Retrieved chunk is not a verbatim source substring")
    return start, start + len(chunk)


def merge_chunks(
    item: dict, texts: dict[int, str], max_gap: int, max_span: int
) -> list[dict]:
    by_doc: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rank, chunk in enumerate(item["relevant_chunks"]):
        doc_id = int(chunk["doc_id"])
        start, end = locate(texts[doc_id], str(chunk["chunk_text"]))
        by_doc[doc_id].append((rank, start, end))

    envelopes: list[tuple[int, int, str]] = []
    for doc_id, spans in by_doc.items():
        spans.sort(key=lambda value: value[1])
        clusters: list[list[tuple[int, int, int]]] = []
        for span in spans:
            if not clusters:
                clusters.append([span])
                continue
            cluster = clusters[-1]
            cluster_start = min(value[1] for value in cluster)
            cluster_end = max(value[2] for value in cluster)
            new_start = min(cluster_start, span[1])
            new_end = max(cluster_end, span[2])
            gap = max(0, span[1] - cluster_end)
            if gap <= max_gap and new_end - new_start <= max_span:
                cluster.append(span)
            else:
                clusters.append([span])
        for cluster in clusters:
            rank = min(value[0] for value in cluster)
            start = min(value[1] for value in cluster)
            end = max(value[2] for value in cluster)
            envelopes.append((rank, doc_id, texts[doc_id][start:end]))
    envelopes.sort(key=lambda value: value[0])
    return [
        {"doc_id": doc_id, "chunk_text": chunk_text}
        for _, doc_id, chunk_text in envelopes
    ]


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    if len(baseline) != 1_200:
        raise RuntimeError("Baseline must contain exactly 1,200 queries")
    query_ids = np.asarray([int(item["id"]) for item in baseline], dtype=np.int64)
    if len(set(query_ids.tolist())) != len(query_ids):
        raise RuntimeError("Baseline query IDs are not unique")

    dense_queries = load_dense_queries(args.query_cache, query_ids)
    index = faiss.read_index(
        str(args.index_dir / "faiss_index_flat_ip.bin"), faiss.IO_FLAG_MMAP
    )
    _, dense_rows = index.search(dense_queries, args.dense_k)
    metadata = pd.read_parquet(args.index_dir / "chunks_metadata.parquet", columns=["doc_id"])
    row_doc_ids = metadata["doc_id"].to_numpy(dtype=np.int64, copy=False)
    extended_docs = extend_documents(baseline, row_doc_ids, dense_rows, args.top_docs)

    texts = load_needed_texts(args.index_dir, baseline)
    output: list[dict] = []
    for item, docs in zip(baseline, extended_docs):
        output.append(
            {
                "id": int(item["id"]),
                "relevant_docs": docs,
                "relevant_chunks": merge_chunks(
                    item, texts, args.max_gap, args.max_span
                ),
            }
        )

    official_ids = set(
        pd.read_parquet(args.official_corpus, columns=["id"])["id"].astype(int)
    )
    chunk_lengths: list[int] = []
    for item in output:
        docs = item["relevant_docs"]
        if len(docs) != args.top_docs or len(set(docs)) != args.top_docs:
            raise RuntimeError(f"Invalid document count for query {item['id']}")
        if not set(docs).issubset(official_ids):
            raise RuntimeError(f"Out-of-corpus document for query {item['id']}")
        seen: set[tuple[int, str]] = set()
        for chunk in item["relevant_chunks"]:
            doc_id = int(chunk["doc_id"])
            text = str(chunk["chunk_text"])
            key = (doc_id, text)
            if doc_id not in official_ids or not text or key in seen:
                raise RuntimeError(f"Invalid/duplicate chunk for query {item['id']}")
            if text not in texts.get(doc_id, ""):
                raise RuntimeError(f"Non-verbatim chunk for query {item['id']}")
            seen.add(key)
            chunk_lengths.append(len(text))

    json_path = args.out_prefix.with_suffix(".json")
    zip_path = args.out_prefix.with_suffix(".zip")
    manifest_path = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        archive.write(json_path, arcname="submission.json")
    if zip_path.stat().st_size >= 100_000_000:
        raise RuntimeError(f"Submission exceeds decimal 100 MB: {zip_path.stat().st_size:,}")

    with zip_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    chunk_counts = [len(item["relevant_chunks"]) for item in output]
    stats = json.loads((args.index_dir / "stats.json").read_text(encoding="utf-8"))
    manifest = {
        "experiment": args.out_prefix.name.removeprefix("submission_"),
        "baseline": args.baseline.name,
        "index_docs": stats["total_docs"],
        "index_chunks": stats["total_chunks"],
        "policy": {
            "documents": f"baseline ranking plus same-index dense-VI tail to {args.top_docs}",
            "chunks": f"merge same-document spans at gap <= {args.max_gap} and span <= {args.max_span}",
        },
        "queries": len(output),
        "docs_per_query": args.top_docs,
        "chunks_per_query": {
            "min": min(chunk_counts),
            "mean": statistics.fmean(chunk_counts),
            "max": max(chunk_counts),
        },
        "chunk_chars": {
            "min": min(chunk_lengths),
            "mean": statistics.fmean(chunk_lengths),
            "median": statistics.median(chunk_lengths),
            "max": max(chunk_lengths),
        },
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": digest,
        "validation": {
            "official_doc_ids_only": True,
            "unique_docs_per_query": True,
            "unique_doc_chunk_pairs_per_query": True,
            "verbatim_source_substrings": True,
            "single_json_at_zip_root": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
