"""Create a docs-only multilingual-translation treatment over Exp122.

The experiment keeps Exp122's first 150 documents and every chunk unchanged.
It fills the remaining 66 document slots from weighted RRF over BGE-M3 dense
retrieval for the original Vietnamese query and audited EN/ZH translations.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ZIP = ROOT / "submission_exp122_exp121_add19_miss.zip"
TRANSLATION_ZIP = ROOT / "data/query_translations/translation_production_v1.zip"
QUERY_PATH = ROOT / "data/raw/btc_provided/query.parquet"
INDEX_DIR = ROOT / "data/indices/new_dataset/index"
OFFICIAL_PATH = ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"
OUTPUT_PREFIX = ROOT / "submission_exp136_translation_production_v1"

BASELINE_HASH_FROM_LOG = "8c7f221cfafa401dfb5435df7da06464999c4b60594e69bc7726bb4ffb379886"
TRANSLATION_MEMBER = "query_translated_production_v1.parquet"
AUDIT_MEMBER = "translation_production_audit_v1.parquet"
MANIFEST_MEMBER = "translation_production_manifest_v1.json"
EXPECTED_QUERY_COUNT = 1200
TOP_DOCS = 216
LOCKED_BASE_DOCS = 150
VIEW_WEIGHTS = {"vi": 1.0, "en": 0.35, "zh": 0.35}
RRF_K = 60
DENSE_K = 4096
MODEL_ID = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, default=BASELINE_ZIP)
    parser.add_argument("--translations-zip", type=Path, default=TRANSLATION_ZIP)
    parser.add_argument("--queries", type=Path, default=QUERY_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--official-corpus", type=Path, default=OFFICIAL_PATH)
    parser.add_argument("--out-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--dense-k", type=int, default=DENSE_K)
    parser.add_argument("--top-docs", type=int, default=TOP_DOCS)
    parser.add_argument("--locked-base-docs", type=int, default=LOCKED_BASE_DOCS)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Replace existing output JSON/ZIP/manifest")
    parser.add_argument("--allow-baseline-hash-mismatch", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def iter_json_array(stream: TextIO, block_size: int = 256 * 1024) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False
    while True:
        if not buffer and not eof:
            block = stream.read(block_size)
            eof = not block
            buffer += block
        buffer = buffer.lstrip()
        if not started:
            if not buffer or buffer[0] != "[":
                raise ValueError("Baseline submission must be a top-level JSON array")
            buffer = buffer[1:]
            started = True
            continue
        buffer = buffer.lstrip()
        if buffer.startswith(","):
            buffer = buffer[1:]
            continue
        if buffer.startswith("]"):
            return
        try:
            item, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof:
                raise
            block = stream.read(block_size)
            eof = not block
            buffer += block
            continue
        if not isinstance(item, dict):
            raise ValueError("Every submission row must be an object")
        yield item
        buffer = buffer[end:]


def load_translations(path: Path, query_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Translation ZIP CRC failed")
        names = archive.namelist()
        if names.count(TRANSLATION_MEMBER) != 1 or names.count(AUDIT_MEMBER) != 1:
            raise RuntimeError("Translation ZIP is missing required parquet members")
        manifest = json.loads(archive.read(MANIFEST_MEMBER))
        translation_bytes = archive.read(TRANSLATION_MEMBER)
        audit_bytes = archive.read(AUDIT_MEMBER)

    expected_hash = manifest["artifacts"][TRANSLATION_MEMBER]["sha256"]
    if hashlib.sha256(translation_bytes).hexdigest() != expected_hash:
        raise RuntimeError("Translation parquet hash does not match its manifest")
    translations = pd.read_parquet(io.BytesIO(translation_bytes)).sort_values("id").reset_index(drop=True)
    audit = pd.read_parquet(io.BytesIO(audit_bytes)).sort_values("id").reset_index(drop=True)
    queries = pd.read_parquet(query_path).sort_values("id").reset_index(drop=True)
    if len(translations) != EXPECTED_QUERY_COUNT or translations.id.duplicated().any():
        raise RuntimeError("Expected 1,200 unique production translations")
    if translations.id.astype(int).tolist() != queries.id.astype(int).tolist():
        raise RuntimeError("Translation query IDs do not match official query IDs/order")
    if translations["query"].astype(str).tolist() != queries["query"].astype(str).tolist():
        raise RuntimeError("Translation artifact's original queries differ from BTC input")
    if audit.id.astype(int).tolist() != queries.id.astype(int).tolist():
        raise RuntimeError("Translation audit IDs do not match official queries")

    translations = translations.merge(
        audit[["id", "status_en_effective", "status_zh"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    for language, status_col in (("en", "status_en_effective"), ("zh", "status_zh")):
        valid = translations[status_col].eq("ok") & translations[f"query_{language}"].fillna("").str.strip().ne("")
        translations[f"use_{language}"] = valid
    return translations, manifest, {
        "en_ok": int(translations.use_en.sum()),
        "en_fallback_original": int((~translations.use_en).sum()),
        "zh_ok": int(translations.use_zh.sum()),
        "zh_fallback_original": int((~translations.use_zh).sum()),
    }


def read_baseline_archive(path: Path) -> tuple[zipfile.ZipFile, str, int]:
    archive = zipfile.ZipFile(path)
    if archive.namelist() != ["submission.json"]:
        archive.close()
        raise RuntimeError("Exp122 ZIP must contain exactly one root submission.json")
    info = archive.getinfo("submission.json")
    return archive, "submission.json", info.file_size


def extract_locked_docs(archive: zipfile.ZipFile, member: str, lock_count: int) -> dict[int, list[int]]:
    output: dict[int, list[int]] = {}
    with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
        for item in iter_json_array(text):
            query_id = int(item["id"])
            docs = [int(doc_id) for doc_id in item["relevant_docs"]]
            if query_id in output or len(docs) != TOP_DOCS or len(set(docs)) != TOP_DOCS:
                raise RuntimeError(f"Invalid/duplicate Exp122 document list for query {query_id}")
            output[query_id] = docs[:lock_count]
    if len(output) != EXPECTED_QUERY_COUNT or set(output) != set(range(1, EXPECTED_QUERY_COUNT + 1)):
        raise RuntimeError("Exp122 baseline does not contain the expected 1,200 query IDs")
    return output


def encode_views(frame: pd.DataFrame, model_id: str, batch_size: int) -> tuple[dict[str, np.ndarray], str]:
    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=model_id)
    encoder = BGEM3FlagModel(model_path, use_fp16=True, devices="cuda:0")
    vectors: dict[str, np.ndarray] = {}
    for view, column in (("vi", "query"), ("en", "query_en"), ("zh", "query_zh")):
        valid = np.ones(len(frame), dtype=bool) if view == "vi" else frame[f"use_{view}"].to_numpy(dtype=bool)
        selected = frame.loc[valid, column].astype(str).tolist()
        if not selected:
            continue
        encoded = encoder.encode(
            selected,
            batch_size=batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        matrix = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        full = np.zeros((len(frame), matrix.shape[1]), dtype=np.float32)
        full[valid] = matrix
        vectors[view] = full
        del encoded, matrix, full
    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return vectors, Path(model_path).name


def exact_streaming_chunk_search(
    index_dir: Path,
    query_vectors: np.ndarray,
    k: int,
    query_batch: int = 256,
    chunk_batch: int = 32768,
) -> np.ndarray:
    """Exact inner-product top-k over aligned FP16 shards without loading the 7-GB FAISS file."""
    metadata = pd.read_parquet(index_dir / "chunks_metadata.parquet", columns=["chunk_id", "doc_id"])
    metadata = metadata.sort_values("chunk_id").reset_index(drop=True)
    chunk_ids = metadata.chunk_id.to_numpy(dtype=np.int64, copy=False)
    row_doc_ids = metadata.doc_id.to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(chunk_ids, np.arange(len(chunk_ids), dtype=np.int64)):
        raise RuntimeError("Chunk metadata IDs are not contiguous/aligned")

    id_paths = sorted((index_dir / "emb_dense").glob("ids_*.npy"))
    vec_paths = sorted((index_dir / "emb_dense").glob("vec_*.npy"))
    if not id_paths or len(id_paths) != len(vec_paths):
        raise RuntimeError("Dense vector/id shard set is incomplete")

    all_rows = np.full((len(query_vectors), k), -1, dtype=np.int64)
    query_tensor = torch.as_tensor(query_vectors, dtype=torch.float16, device="cuda:0")
    shard_start = 0
    for shard_number, (id_path, vec_path) in enumerate(zip(id_paths, vec_paths), 1):
        ids = np.load(id_path, mmap_mode="r", allow_pickle=False)
        vectors = np.load(vec_path, mmap_mode="r", allow_pickle=False)
        if len(ids) != len(vectors) or vectors.ndim != 2 or vectors.shape[1] != query_vectors.shape[1]:
            raise RuntimeError(f"Invalid dense shard pair: {vec_path.name}")
        if not np.array_equal(ids, chunk_ids[shard_start : shard_start + len(ids)]):
            raise RuntimeError(f"Dense shard IDs do not align with chunk metadata: {id_path.name}")

        vec_gpu = torch.as_tensor(np.array(vectors, copy=True), dtype=torch.float16, device="cuda:0")
        for q_start in range(0, len(query_vectors), query_batch):
            q_stop = min(q_start + query_batch, len(query_vectors))
            q = query_tensor[q_start:q_stop]
            best_scores = torch.full((len(q), k), -torch.inf, dtype=torch.float32, device="cuda:0")
            best_rows = torch.full((len(q), k), -1, dtype=torch.int64, device="cuda:0")
            for c_start in range(0, len(vectors), chunk_batch):
                c_stop = min(c_start + chunk_batch, len(vectors))
                scores = torch.matmul(q, vec_gpu[c_start:c_stop].T).float()
                local_k = min(k, c_stop - c_start)
                block_scores, block_rows = torch.topk(scores, local_k, dim=1, largest=True, sorted=False)
                block_rows = block_rows.to(torch.int64) + shard_start + c_start
                merged_scores = torch.cat((best_scores, block_scores), dim=1)
                merged_rows = torch.cat((best_rows, block_rows), dim=1)
                best_scores, positions = torch.topk(merged_scores, k, dim=1, largest=True, sorted=False)
                best_rows = torch.gather(merged_rows, 1, positions)
                del scores, block_scores, block_rows, merged_scores, merged_rows, positions
            all_rows[q_start:q_stop] = best_rows.cpu().numpy()
            del best_scores, best_rows
        del vec_gpu, vectors, ids
        torch.cuda.empty_cache()
        shard_start += len(np.load(id_path, mmap_mode="r", allow_pickle=False))
        print(f"Dense index shards: {shard_number}/{len(vec_paths)}", flush=True)
    if shard_start != len(row_doc_ids):
        raise RuntimeError("Dense shard rows do not cover all chunk metadata")
    torch.cuda.empty_cache()
    return all_rows


def docs_from_chunk_ranks(rows: np.ndarray, row_doc_ids: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = []
    for row_list in rows:
        seen: set[int] = set()
        docs: list[int] = []
        for row in row_list:
            if row >= 0:
                doc_id = int(row_doc_ids[int(row)])
                if doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(doc_id)
        result.append(docs)
    return result


def main() -> None:
    args = parse_args()
    if args.top_docs != TOP_DOCS or not 1 <= args.locked_base_docs < args.top_docs:
        raise ValueError("This experiment expects 216 output docs and a lock count in 1..215")
    if args.dense_k < 1 or args.rrf_k < 1:
        raise ValueError("dense-k and rrf-k must be positive")
    output_path = args.out_prefix.with_suffix(".json")
    zip_path = args.out_prefix.with_suffix(".zip")
    manifest_path = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    existing = [path for path in (output_path, zip_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}; use --force deliberately")

    baseline_hash = sha256_file(args.baseline_zip)
    if baseline_hash != BASELINE_HASH_FROM_LOG and not args.allow_baseline_hash_mismatch:
        raise RuntimeError(
            f"Baseline ZIP hash {baseline_hash} differs from the verified Exp122 artifact; "
            "pass --allow-baseline-hash-mismatch only after reviewing its provenance"
        )
    translations, translation_manifest, translation_counts = load_translations(args.translations_zip, args.queries)
    baseline_archive, baseline_member, baseline_json_bytes = read_baseline_archive(args.baseline_zip)
    try:
        print("Reading and validating Exp122 control rows...", flush=True)
        locked_docs = extract_locked_docs(baseline_archive, baseline_member, args.locked_base_docs)
        print("Encoding original and audited translated query views with BGE-M3...", flush=True)
        views, model_revision = encode_views(translations, args.model_id, args.batch_size)

        print("Loading aligned dense index and retrieving query views...", flush=True)
        meta = pd.read_parquet(args.index_dir / "chunks_metadata.parquet", columns=["chunk_id", "doc_id"])
        meta = meta.sort_values("chunk_id").reset_index(drop=True)
        row_doc_ids = meta.doc_id.to_numpy(dtype=np.int64, copy=False)
        if len(row_doc_ids) != int(json.loads((args.index_dir / "stats.json").read_text(encoding="utf-8"))["total_chunks"]):
            raise RuntimeError("Chunk metadata count does not match active index stats")

        official_ids = set(pd.read_parquet(args.official_corpus, columns=["id"]).id.astype(int))
        index_docs = set(map(int, np.unique(row_doc_ids)))
        if not index_docs.issubset(official_ids):
            raise RuntimeError("Active dense index contains IDs outside the BTC corpus")

        packed_vectors: list[np.ndarray] = []
        packed_map: list[tuple[str, int]] = []
        for view, vectors in views.items():
            if view == "en":
                mask = translations.use_en.to_numpy(dtype=bool)
            elif view == "zh":
                mask = translations.use_zh.to_numpy(dtype=bool)
            else:
                mask = np.ones(len(translations), dtype=bool)
            if mask.any():
                positions = np.flatnonzero(mask)
                packed_vectors.append(vectors[mask])
                packed_map.extend((view, int(position)) for position in positions)

        query_vectors = np.concatenate(packed_vectors, axis=0)
        print(f"Exact GPU search for {len(query_vectors):,} valid query views...", flush=True)
        chunk_ranks = exact_streaming_chunk_search(args.index_dir, query_vectors, args.dense_k)
        doc_ranks = docs_from_chunk_ranks(chunk_ranks, row_doc_ids)
        per_view: dict[str, list[list[int]]] = {
            view: [[] for _ in range(len(translations))] for view in views
        }
        for (view, position), ranked_docs in zip(packed_map, doc_ranks):
            per_view[view][position] = ranked_docs
        del query_vectors, chunk_ranks, doc_ranks, packed_vectors, packed_map

        print("Applying weighted RRF and building docs-only output...", flush=True)
        rrf_lists: dict[int, list[int]] = {}
        for qi, query_id in enumerate(translations.id.astype(int)):
            scores: dict[int, float] = {}
            for view in ("vi", "en", "zh"):
                for rank, doc_id in enumerate(per_view[view][qi], 1):
                    scores[doc_id] = scores.get(doc_id, 0.0) + VIEW_WEIGHTS[view] / (args.rrf_k + rank)
            rrf_lists[query_id] = [doc for doc, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        changed_queries = 0
        added_docs: list[int] = []
        docs_per_query: list[int] = []
        chunk_counts: list[int] = []
        seen_queries: set[int] = set()
        with baseline_archive.open(baseline_member) as binary, io.TextIOWrapper(binary, encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="") as dest:
            dest.write("[")
            first = True
            for item in iter_json_array(source):
                query_id = int(item["id"])
                if query_id in seen_queries:
                    raise RuntimeError(f"Duplicate baseline query {query_id}")
                seen_queries.add(query_id)
                original_docs = [int(doc_id) for doc_id in item["relevant_docs"]]
                keep = original_docs[:args.locked_base_docs]
                keep_set = set(keep)
                tail: list[int] = []
                for doc_id in rrf_lists[query_id]:
                    if doc_id not in keep_set and doc_id not in tail:
                        tail.append(doc_id)
                    if len(keep) + len(tail) == args.top_docs:
                        break
                if len(keep) + len(tail) != args.top_docs:
                    raise RuntimeError(f"Not enough distinct candidate documents for query {query_id}")
                docs = keep + tail
                if any(doc not in official_ids or doc not in index_docs for doc in docs):
                    raise RuntimeError(f"Out-of-corpus/unindexed document in query {query_id}")
                if docs != original_docs:
                    changed_queries += 1
                added_docs.extend(tail)
                item["relevant_docs"] = docs
                if not first:
                    dest.write(",")
                json.dump(item, dest, ensure_ascii=False, separators=(",", ":"))
                first = False
                docs_per_query.append(len(docs))
                chunk_counts.append(len(item["relevant_chunks"]))
            dest.write("]")
        if seen_queries != set(range(1, EXPECTED_QUERY_COUNT + 1)):
            raise RuntimeError("Generated submission is missing query IDs")
        baseline_archive.close()

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_LZMA, compresslevel=9) as output_zip:
            output_zip.write(output_path, arcname="submission.json")
        if zip_path.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Submission ZIP exceeds decimal 100 MB: {zip_path.stat().st_size:,}")
        with zipfile.ZipFile(zip_path) as result_zip:
            if result_zip.namelist() != ["submission.json"] or result_zip.testzip() is not None:
                raise RuntimeError("Output ZIP structure/CRC validation failed")

        manifest = {
            "experiment": "Exp136 translation production dense-RRF docs-only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_zip": str(args.baseline_zip),
            "baseline_zip_sha256": baseline_hash,
            "baseline_json_bytes": baseline_json_bytes,
            "translation_zip": str(args.translations_zip),
            "translation_zip_sha256": sha256_file(args.translations_zip),
            "translation_manifest": translation_manifest,
            "translation_usage": translation_counts,
            "model_id": args.model_id,
            "model_revision": model_revision,
            "active_index_model": "BAAI/bge-m3",
            "dense_k_chunks_per_view": args.dense_k,
            "rrf_k": args.rrf_k,
            "view_weights": VIEW_WEIGHTS,
            "locked_exp122_docs": args.locked_base_docs,
            "output_docs": args.top_docs,
            "changed_queries": changed_queries,
            "tail_doc_additions": len(added_docs),
            "unique_tail_docs": len(set(added_docs)),
            "docs_per_query": {"min": min(docs_per_query), "max": max(docs_per_query)},
            "chunks_per_query": {
                "min": min(chunk_counts), "mean": statistics.fmean(chunk_counts), "max": max(chunk_counts)
            },
            "chunks_preserved_from_baseline": True,
            "queries": len(seen_queries),
            "zip_bytes": zip_path.stat().st_size,
            "zip_sha256": sha256_file(zip_path),
            "json_sha256": sha256_file(output_path),
            "score": "pending BTC evaluation; no score inferred from retrieval overlap",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    finally:
        if not baseline_archive.fp is None:
            baseline_archive.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
