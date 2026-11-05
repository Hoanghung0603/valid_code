"""Create an Exp122-preserving, translation-driven chunk retrieval treatment.

Document membership is fixed exactly to Exp122. Audited EN/ZH query views can
only change which source passages are selected from those documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_translation_submission import (
    BASELINE_HASH_FROM_LOG,
    EXPECTED_QUERY_COUNT,
    RRF_K,
    ROOT,
    VIEW_WEIGHTS,
    encode_views,
    iter_json_array,
    load_translations,
    sha256_file,
)


BASELINE_ZIP = ROOT / "submission_exp122_exp121_add19_miss.zip"
INDEX_DIR = ROOT / "data/indices/new_dataset/index"
OFFICIAL_PATH = ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"
OUTPUT_PREFIX = ROOT / "submission_exp137_translation_chunks_fixed_docs"
TRANSLATION_ZIP = ROOT / "data/query_translations/translation_production_v1.zip"
QUERY_PATH = ROOT / "data/raw/btc_provided/query.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, default=BASELINE_ZIP)
    parser.add_argument("--translations-zip", type=Path, default=TRANSLATION_ZIP)
    parser.add_argument("--queries", type=Path, default=QUERY_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--official-corpus", type=Path, default=OFFICIAL_PATH)
    parser.add_argument("--out-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--model-id", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-baseline-hash-mismatch", action="store_true")
    return parser.parse_args()


def read_baseline(path: Path) -> tuple[list[dict], int, str]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Exp122 ZIP layout/CRC validation failed")
        info = archive.getinfo("submission.json")
        with archive.open("submission.json") as binary:
            import io

            with io.TextIOWrapper(binary, encoding="utf-8") as text:
                rows = list(iter_json_array(text))
    if len(rows) != EXPECTED_QUERY_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_QUERY_COUNT} Exp122 queries, got {len(rows)}")
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(1, EXPECTED_QUERY_COUNT + 1)):
        raise RuntimeError("Exp122 query IDs/order are invalid")
    for row in rows:
        docs = row["relevant_docs"]
        if len(docs) != 216 or len(set(docs)) != 216:
            raise RuntimeError(f"Invalid fixed docs for query {row['id']}")
    return rows, info.file_size, sha256_file(path)


def make_candidate_ids(rows: list[dict], metadata: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    metadata = metadata.sort_values("chunk_id").reset_index(drop=True)
    chunk_ids = metadata.chunk_id.to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(chunk_ids, np.arange(len(metadata), dtype=np.int64)):
        raise RuntimeError("Chunk metadata IDs are not contiguous/aligned")
    doc_to_chunks = metadata.groupby("doc_id", sort=False).chunk_id.apply(
        lambda values: values.to_numpy(dtype=np.int64, copy=False)
    ).to_dict()
    candidates: list[np.ndarray] = []
    for row in rows:
        missing = [doc for doc in row["relevant_docs"] if doc not in doc_to_chunks]
        if missing:
            raise RuntimeError(f"Exp122 docs lack indexed chunks for query {row['id']}: {missing[:3]}")
        ids = np.concatenate([doc_to_chunks[doc] for doc in row["relevant_docs"]])
        candidates.append(np.sort(ids))
    return candidates, metadata.doc_id.to_numpy(dtype=np.int64, copy=False)


def load_vector_shards(index_dir: Path, expected_rows: int, dimension: int):
    id_paths = sorted((index_dir / "emb_dense").glob("ids_*.npy"))
    vec_paths = sorted((index_dir / "emb_dense").glob("vec_*.npy"))
    if not id_paths or len(id_paths) != len(vec_paths):
        raise RuntimeError("Dense vector/id shard set is incomplete")
    shards = []
    start = 0
    for id_path, vec_path in zip(id_paths, vec_paths, strict=True):
        ids = np.load(id_path, mmap_mode="r", allow_pickle=False)
        vectors = np.load(vec_path, mmap_mode="r", allow_pickle=False)
        if vectors.shape != (len(ids), dimension) or not np.array_equal(
            ids, np.arange(start, start + len(ids), dtype=np.int64)
        ):
            raise RuntimeError(f"Invalid/alignment mismatch in {vec_path.name}")
        shards.append((start, start + len(ids), vectors))
        start += len(ids)
        if len(shards) % 5 == 0 or len(shards) == len(id_paths):
            print(f"Validated dense shards {len(shards)}/{len(id_paths)}", flush=True)
    if start != expected_rows:
        raise RuntimeError(f"Dense shards cover {start:,} rows, metadata has {expected_rows:,}")
    return shards


def fetch_vectors(chunk_ids: np.ndarray, shards) -> np.ndarray:
    output = np.empty((len(chunk_ids), shards[0][2].shape[1]), dtype=np.float16)
    shard_starts = np.fromiter((item[0] for item in shards), dtype=np.int64)
    shard_indices = np.searchsorted(shard_starts, chunk_ids, side="right") - 1
    for shard_index in np.unique(shard_indices):
        start, stop, vectors = shards[int(shard_index)]
        mask = shard_indices == shard_index
        selected = chunk_ids[mask]
        if np.any(selected < start) or np.any(selected >= stop):
            raise RuntimeError("Candidate chunk ID falls outside its vector shard")
        output[mask] = vectors[selected - start]
    return output


def rank_candidates(
    candidate_ids: list[np.ndarray],
    query_views: dict[str, np.ndarray],
    translations: pd.DataFrame,
    shards,
    output_counts: list[int],
) -> list[np.ndarray]:
    ranked: list[np.ndarray] = []
    for qi, ids in enumerate(candidate_ids):
        candidate_vectors = fetch_vectors(ids, shards)
        matrix = torch.as_tensor(candidate_vectors, dtype=torch.float16, device="cuda:0")
        scores_by_view: dict[str, np.ndarray] = {}
        for view, vectors in query_views.items():
            enabled = view == "vi" or bool(translations.iloc[qi][f"use_{view}"])
            if not enabled:
                continue
            q = torch.as_tensor(vectors[qi], dtype=torch.float16, device="cuda:0")
            scores_by_view[view] = (matrix @ q).float().cpu().numpy()
        del matrix

        rrf = np.zeros(len(ids), dtype=np.float64)
        for view, scores in scores_by_view.items():
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(order), dtype=np.int32)
            ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
            rrf += VIEW_WEIGHTS[view] / (RRF_K + ranks)
        order = np.lexsort((ids, -rrf))
        count = output_counts[qi]
        ranked.append(ids[order[:count]])
        if qi == 0:
            print(f"First fixed-doc query ranked ({len(ids)} candidate chunks)", flush=True)
        if (qi + 1) % 100 == 0 or qi + 1 == len(candidate_ids):
            print(f"Ranked fixed-doc chunks {qi + 1}/{len(candidate_ids)}", flush=True)
    return ranked


def load_source_texts(index_dir: Path, needed_docs: set[int]) -> dict[int, str]:
    texts: dict[int, str] = {}
    text_paths = sorted((index_dir / "text").glob("part_*.parquet"))
    for i, path in enumerate(text_paths, 1):
        frame = pd.read_parquet(path, columns=["doc_id", "text"])
        subset = frame[frame.doc_id.isin(needed_docs.difference(texts))]
        texts.update((int(row.doc_id), str(row.text)) for row in subset.itertuples(index=False))
        if len(texts) == len(needed_docs):
            break
        if i % 20 == 0 or i == len(text_paths):
            print(f"Loaded source texts {len(texts):,}/{len(needed_docs):,}", flush=True)
    missing = needed_docs.difference(texts)
    if missing:
        raise RuntimeError(f"Missing source text for {len(missing):,} selected docs")
    return texts


def main() -> None:
    args = parse_args()
    out_json = args.out_prefix.with_suffix(".json")
    out_zip = args.out_prefix.with_suffix(".zip")
    out_manifest = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    existing = [path for path in (out_json, out_zip, out_manifest) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}; use --force")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact dense chunk scoring")

    baseline_rows, baseline_json_bytes, baseline_hash = read_baseline(args.baseline_zip)
    if baseline_hash != BASELINE_HASH_FROM_LOG and not args.allow_baseline_hash_mismatch:
        raise RuntimeError(f"Unexpected Exp122 baseline hash: {baseline_hash}")
    translations, translation_manifest, translation_usage = load_translations(
        args.translations_zip, args.queries
    )
    metadata = pd.read_parquet(
        args.index_dir / "chunks_metadata.parquet",
        columns=["chunk_id", "doc_id", "start", "end"],
    ).sort_values("chunk_id").reset_index(drop=True)
    stats = json.loads((args.index_dir / "stats.json").read_text(encoding="utf-8"))
    if len(metadata) != int(stats["total_chunks"]):
        raise RuntimeError("Chunk metadata count does not match active index stats")
    candidate_ids, row_doc_ids = make_candidate_ids(baseline_rows, metadata)
    print(
        "Fixed-doc candidate chunk counts: "
        f"min={min(map(len, candidate_ids))}, "
        f"mean={statistics.fmean(map(len, candidate_ids)):.1f}, "
        f"max={max(map(len, candidate_ids))}",
        flush=True,
    )
    needed_docs = {int(doc) for row in baseline_rows for doc in row["relevant_docs"]}
    official_ids = set(pd.read_parquet(args.official_corpus, columns=["id"]).id.astype(int))
    if not needed_docs.issubset(official_ids):
        raise RuntimeError("Exp122 contains document IDs outside BTC corpus")

    print("Encoding original VI and audited EN/ZH translations with BGE-M3...", flush=True)
    query_views, model_revision = encode_views(translations, args.model_id, args.batch_size)
    dimension = next(iter(query_views.values())).shape[1]
    shards = load_vector_shards(args.index_dir, len(metadata), dimension)
    counts = [len(row["relevant_chunks"]) for row in baseline_rows]
    print("Scoring only chunks owned by each query's fixed Exp122 docs...", flush=True)
    selected_ids = rank_candidates(candidate_ids, query_views, translations, shards, counts)

    selected_doc_ids: list[np.ndarray] = []
    selected_spans: list[np.ndarray] = []
    for ids in selected_ids:
        selected_doc_ids.append(row_doc_ids[ids])
        selected_spans.append(metadata.loc[ids, ["start", "end"]].to_numpy(dtype=np.int64))
    texts = load_source_texts(args.index_dir, needed_docs)

    output: list[dict] = []
    for qi, base in enumerate(baseline_rows):
        chunks = []
        for doc_id, (start, end) in zip(selected_doc_ids[qi], selected_spans[qi], strict=True):
            doc_id = int(doc_id)
            chunk_text = texts[doc_id][int(start):int(end)]
            if not chunk_text.strip():
                raise RuntimeError(f"Empty source span in query {base['id']}")
            chunks.append({"doc_id": doc_id, "chunk_text": chunk_text})
        if any(chunk["doc_id"] not in base["relevant_docs"] for chunk in chunks):
            raise RuntimeError(f"Chunk escaped fixed Exp122 docs for query {base['id']}")
        if len(chunks) != counts[qi]:
            raise RuntimeError(f"Chunk count changed for query {base['id']}")
        output.append({
            "id": int(base["id"]),
            "relevant_docs": base["relevant_docs"],
            "relevant_chunks": chunks,
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_LZMA, compresslevel=9) as archive:
        archive.write(out_json, arcname="submission.json")
    if out_zip.stat().st_size >= 100_000_000:
        raise RuntimeError(f"Submission exceeds decimal 100 MB limit: {out_zip.stat().st_size:,}")
    with zipfile.ZipFile(out_zip) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Output ZIP layout/CRC validation failed")

    chunk_counts = [len(item["relevant_chunks"]) for item in output]
    manifest = {
        "experiment": "Exp137 production translations; chunks-only, fixed Exp122 docs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_zip": str(args.baseline_zip),
        "baseline_zip_sha256": baseline_hash,
        "baseline_json_bytes": baseline_json_bytes,
        "translation_zip": str(args.translations_zip),
        "translation_zip_sha256": sha256_file(args.translations_zip),
        "translation_manifest": translation_manifest,
        "translation_usage": translation_usage,
        "model_id": args.model_id,
        "model_revision": model_revision,
        "active_index_model": "BAAI/bge-m3",
        "retrieval": "weighted RRF over exact dense rankings restricted to Exp122 docs/query",
        "rrf_k": RRF_K,
        "view_weights": VIEW_WEIGHTS,
        "docs_unchanged_from_exp122": True,
        "chunks_selected_per_query": "same count as Exp122",
        "queries": len(output),
        "chunks_per_query": {
            "min": min(chunk_counts), "mean": statistics.fmean(chunk_counts), "max": max(chunk_counts)
        },
        "zip_bytes": out_zip.stat().st_size,
        "zip_sha256": sha256_file(out_zip),
        "json_sha256": sha256_file(out_json),
        "score": None,
        "validation": {
            "all_docs_exactly_equal_exp122": True,
            "all_chunks_owned_by_returned_docs": True,
            "source_substrings_only": True,
            "same_chunk_count_as_exp122_per_query": True,
            "zip_single_root_submission_json_crc_ok": True,
            "zip_under_decimal_100mb": True,
        },
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "zip": str(out_zip), "json": str(out_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
