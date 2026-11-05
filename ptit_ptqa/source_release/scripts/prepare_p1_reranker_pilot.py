"""Prepare a compact Exp122 document-reranking pilot for Kaggle.

The pack contains only the 120 stratified pilot queries, Exp122's exact 216
documents per query, and at most two active-index semantic passages per
query/document.  Kaggle performs reranker inference only; no corpus index or
submission payload is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_translation_submission import iter_json_array, sha256_file
from generate_translation_chunks_fixed_docs import fetch_vectors, load_vector_shards


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_exp122_exp121_add19_miss.zip"
CONTROL_SHA256 = "8c7f221cfafa401dfb5435df7da06464999c4b60594e69bc7726bb4ffb379886"
PILOT = ROOT / "data/audits/queryspec_pilot_v6_final/pilot_manifest.parquet"
QUERIES = ROOT / "data/raw/btc_provided/query.parquet"
CORPUS = ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"
INDEX = ROOT / "data/indices/new_dataset/index"
OUT = ROOT / "data/kaggle_inputs/p1_reranker_pilot_v1"
OUT_FULL = ROOT / "data/kaggle_inputs/p1_reranker_full_v1"
MODEL_ID = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-passages", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-control-hash-mismatch", action="store_true")
    return parser.parse_args()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def detect_script_language(text: str) -> str:
    sample = text[:6000]
    non_space = sum(not ch.isspace() for ch in sample)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in sample)
    if non_space and cjk / non_space >= 0.05:
        return "zh"
    if any(ch in "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ" for ch in sample):
        return "vi"
    return "en_or_plain_latin"


def read_control(path: Path, wanted_ids: set[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Exp122 ZIP layout/CRC validation failed")
        with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
            for row in iter_json_array(text):
                query_id = int(row["id"])
                if query_id not in wanted_ids:
                    continue
                docs = [int(value) for value in row["relevant_docs"]]
                if len(docs) != 216 or len(set(docs)) != 216:
                    raise RuntimeError(f"Query {query_id} does not have 216 unique control docs")
                all_owners = {int(chunk["doc_id"]) for chunk in row["relevant_chunks"]}
                # BTC scores document and chunk axes independently.  Exp122 can
                # therefore contain chunk owners outside its relevant_docs list;
                # only owners already in the 216-doc set can be protected by a
                # docs-only pruning treatment.
                rows[query_id] = {
                    "docs": docs,
                    "owner_docs": all_owners.intersection(docs),
                    "external_owner_docs": all_owners.difference(docs),
                }
    if set(rows) != wanted_ids:
        raise RuntimeError(f"Control is missing pilot IDs: {sorted(wanted_ids - set(rows))[:10]}")
    return rows


def select_top_passages(
    chunk_ids: np.ndarray,
    doc_ids: np.ndarray,
    scores: np.ndarray,
    max_passages: int,
) -> dict[int, list[int]]:
    by_doc: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for chunk_id, doc_id, score in zip(chunk_ids, doc_ids, scores, strict=True):
        by_doc[int(doc_id)].append((float(score), int(chunk_id)))
    result: dict[int, list[int]] = {}
    for doc_id, values in by_doc.items():
        values.sort(key=lambda item: (-item[0], item[1]))
        result[doc_id] = [chunk_id for _, chunk_id in values[:max_passages]]
    return result


def select_nonempty_passages(
    ranked_chunk_ids: list[int],
    source: str,
    starts: np.ndarray,
    ends: np.ndarray,
    max_passages: int,
) -> tuple[list[int], int]:
    """Keep the highest-ranked exact index spans that contain visible text."""
    selected: list[int] = []
    skipped = 0
    for chunk_id in ranked_chunk_ids:
        passage = source[int(starts[chunk_id]):int(ends[chunk_id])]
        if not passage.strip():
            skipped += 1
            continue
        selected.append(int(chunk_id))
        if len(selected) == max_passages:
            break
    return selected, skipped


def load_needed_texts(index_dir: Path, needed_docs: set[int]) -> dict[int, str]:
    texts: dict[int, str] = {}
    paths = sorted((index_dir / "text").glob("part_*.parquet"))
    for number, path in enumerate(paths, 1):
        frame = pd.read_parquet(path, columns=["doc_id", "text"])
        subset = frame[frame.doc_id.isin(needed_docs.difference(texts))]
        texts.update((int(row.doc_id), str(row.text)) for row in subset.itertuples(index=False))
        if len(texts) == len(needed_docs):
            break
        if number % 20 == 0:
            print(f"Loaded source text {len(texts):,}/{len(needed_docs):,}", flush=True)
    missing = needed_docs.difference(texts)
    if missing:
        raise RuntimeError(f"Active text store misses {len(missing):,} control docs")
    return texts


def encode_queries(frame: pd.DataFrame, model_id: str, batch_size: int) -> tuple[np.ndarray, str]:
    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=model_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = BGEM3FlagModel(model_path, use_fp16=device.startswith("cuda"), devices=device)
    encoded = encoder.encode(
        frame["query"].astype(str).tolist(),
        batch_size=batch_size,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = np.asarray(encoded["dense_vecs"], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors, Path(model_path).name


def validate_pack(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    control: dict[int, dict],
    texts: dict[int, str],
    max_passages: int,
) -> dict:
    if queries.query_id.duplicated().any() or len(queries) != len(control):
        raise RuntimeError("Pilot query table has duplicate/missing IDs")
    if pairs.pair_id.duplicated().any():
        raise RuntimeError("Duplicate pair_id in P1 pack")
    pair_counts = pairs.groupby(["query_id", "doc_id"]).size()
    if pair_counts.min() < 1 or pair_counts.max() > max_passages:
        raise RuntimeError("Passages/query-doc outside configured bounds")
    expected = {(qid, doc) for qid, item in control.items() for doc in item["docs"]}
    actual = set(zip(pairs.query_id.astype(int), pairs.doc_id.astype(int)))
    if actual != expected:
        raise RuntimeError(f"Query-doc coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    source_failures = 0
    for row in pairs.itertuples(index=False):
        source = texts[int(row.doc_id)]
        passage = source[int(row.start):int(row.end)]
        if passage != row.passage_text or hash_text(passage) != row.passage_sha256:
            source_failures += 1
    if source_failures:
        raise RuntimeError(f"{source_failures} passages failed source-offset/hash validation")
    owner_expected = {(qid, doc) for qid, item in control.items() for doc in item["owner_docs"]}
    owner_actual = set(
        zip(pairs.loc[pairs.owner_doc, "query_id"].astype(int), pairs.loc[pairs.owner_doc, "doc_id"].astype(int))
    )
    if owner_actual != owner_expected:
        raise RuntimeError("owner_doc flags differ from Exp122 chunk owners")
    return {
        "queries": len(control),
        "query_docs": len(expected),
        "pairs": len(pairs),
        "owner_query_docs": len(owner_expected),
        "non_owner_query_docs": len(expected) - len(owner_expected),
        "external_chunk_owner_query_docs": sum(len(item["external_owner_docs"]) for item in control.values()),
        "passages_per_query_doc": {
            "min": int(pair_counts.min()),
            "mean": float(pair_counts.mean()),
            "max": int(pair_counts.max()),
        },
        "language_pairs": {str(key): int(value) for key, value in pairs.language.value_counts().items()},
        "source_offset_hash_failures": 0,
    }


def main() -> None:
    args = parse_args()
    output_dir = OUT_FULL if args.scope == "full" and args.output_dir == OUT else args.output_dir
    output = output_dir.resolve()
    zip_path = output.with_suffix(".zip")
    outputs = [output / "p1_pairs.parquet", output / "p1_queries.parquet", output / "p1_manifest.json", zip_path]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("P1 pilot outputs already exist; pass --force only after reviewing provenance")

    control_hash = sha256_file(args.control)
    if control_hash != CONTROL_SHA256 and not args.allow_control_hash_mismatch:
        raise RuntimeError(f"Unexpected Exp122 control hash: {control_hash}")
    official_frame = pd.read_parquet(args.queries).sort_values("id").reset_index(drop=True)
    official_queries = official_frame.set_index("id")
    if args.scope == "pilot":
        pilot = pd.read_parquet(args.pilot).sort_values("id").reset_index(drop=True)
        if len(pilot) != 120 or pilot.id.duplicated().any():
            raise RuntimeError("Expected exactly 120 unique stratified pilot IDs")
        original = pilot["query"].astype(str).tolist() if "query" in pilot else None
        pilot["query"] = [str(official_queries.loc[int(qid), "query"]) for qid in pilot.id]
        if original is not None and pilot["query"].tolist() != original:
            raise RuntimeError("Pilot query text differs from official query.parquet")
    else:
        if len(official_frame) != 1200 or official_frame.id.duplicated().any():
            raise RuntimeError("Expected 1,200 unique official queries for full scope")
        pilot = official_frame.copy()
        pilot["stratum"] = "full"
        pilot["length_bin"] = "unknown"
        pilot["intent_bin"] = "unknown"
        pilot["multi_clause"] = False
    wanted_ids = set(pilot.id.astype(int))
    control = read_control(args.control, wanted_ids)
    needed_docs = {doc for item in control.values() for doc in item["docs"]}
    official_ids = set(pd.read_parquet(args.corpus, columns=["id"]).id.astype(int))
    if not needed_docs.issubset(official_ids):
        raise RuntimeError("Control contains IDs outside official corpus")

    metadata = pd.read_parquet(
        args.index / "chunks_metadata.parquet", columns=["chunk_id", "doc_id", "start", "end"]
    ).sort_values("chunk_id").reset_index(drop=True)
    chunk_ids = metadata.chunk_id.to_numpy(dtype=np.int64, copy=False)
    if not np.array_equal(chunk_ids, np.arange(len(metadata), dtype=np.int64)):
        raise RuntimeError("Active chunk IDs are not contiguous/aligned")
    selected_meta = metadata[metadata.doc_id.isin(needed_docs)].copy()
    missing_docs = needed_docs.difference(set(selected_meta.doc_id.astype(int)))
    if missing_docs:
        raise RuntimeError(f"{len(missing_docs):,} control docs have no active indexed chunk")
    doc_to_chunks = selected_meta.groupby("doc_id", sort=False).chunk_id.apply(
        lambda values: values.to_numpy(dtype=np.int64, copy=False)
    ).to_dict()
    texts = load_needed_texts(args.index, needed_docs)
    query_vectors, model_revision = encode_queries(
        pilot.rename(columns={"id": "query_id"}), args.model_id, args.batch_size
    )
    stats = json.loads((args.index / "stats.json").read_text(encoding="utf-8"))
    shards = load_vector_shards(args.index, len(metadata), query_vectors.shape[1])

    meta_doc = metadata.doc_id.to_numpy(dtype=np.int64, copy=False)
    meta_start = metadata.start.to_numpy(dtype=np.int64, copy=False)
    meta_end = metadata.end.to_numpy(dtype=np.int64, copy=False)
    records: list[dict] = []
    skipped_empty_spans = 0
    for query_number, pilot_row in enumerate(pilot.itertuples(index=False)):
        query_id = int(pilot_row.id)
        docs = control[query_id]["docs"]
        candidates = np.concatenate([doc_to_chunks[doc] for doc in docs])
        vectors = fetch_vectors(candidates, shards).astype(np.float32)
        scores = vectors @ query_vectors[query_number]
        score_map = {int(chunk_id): float(score) for chunk_id, score in zip(candidates, scores, strict=True)}
        owner_docs = control[query_id]["owner_docs"]
        for doc_rank, doc_id in enumerate(docs, 1):
            source = texts[doc_id]
            language = detect_script_language(source)
            ranked_chunks = sorted(
                (int(chunk_id) for chunk_id in doc_to_chunks[doc_id]),
                key=lambda chunk_id: (-score_map[chunk_id], chunk_id),
            )
            chosen, skipped = select_nonempty_passages(
                ranked_chunks, source, meta_start, meta_end, args.max_passages
            )
            skipped_empty_spans += skipped
            if not chosen:
                raise RuntimeError(f"No non-empty indexed source span: query={query_id}, doc={doc_id}")
            for passage_rank, chunk_id in enumerate(chosen, 1):
                start, end = int(meta_start[chunk_id]), int(meta_end[chunk_id])
                passage = source[start:end]
                records.append({
                    "pair_id": f"{query_id}:{doc_id}:{passage_rank}",
                    "query_id": query_id,
                    "query_text": str(pilot_row.query),
                    "doc_id": doc_id,
                    "doc_rank": doc_rank,
                    "owner_doc": doc_id in owner_docs,
                    "passage_rank": passage_rank,
                    "chunk_id": int(chunk_id),
                    "start": start,
                    "end": end,
                    "language": language,
                    "dense_score": score_map[int(chunk_id)],
                    "passage_text": passage,
                    "source_sha256": hash_text(source),
                    "passage_sha256": hash_text(passage),
                })
        if (query_number + 1) % 20 == 0 or query_number + 1 == len(pilot):
            print(f"Prepared {args.scope} query {query_number + 1}/{len(pilot)}", flush=True)

    pairs = pd.DataFrame.from_records(records).sort_values(
        ["query_id", "doc_rank", "passage_rank"]
    ).reset_index(drop=True)
    queries_out = pilot.rename(columns={"id": "query_id"})[
        ["query_id", "query", "stratum", "length_bin", "intent_bin", "multi_clause"]
    ].copy()
    validation = validate_pack(pairs, queries_out, control, texts, args.max_passages)

    output.mkdir(parents=True, exist_ok=True)
    pairs_path = output / "p1_pairs.parquet"
    queries_path = output / "p1_queries.parquet"
    manifest_path = output / "p1_manifest.json"
    pairs.to_parquet(pairs_path, index=False, compression="zstd")
    queries_out.to_parquet(queries_path, index=False, compression="zstd")
    manifest = {
        "schema_version": f"p1_reranker_{args.scope}_v1",
        "scope": args.scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_zip_sha256": control_hash,
        "pilot_manifest_sha256": sha256_file(args.pilot) if args.scope == "pilot" else None,
        "official_queries_sha256": sha256_file(args.queries),
        "official_corpus_sha256": sha256_file(args.corpus),
        "active_index_stats_sha256": sha256_file(args.index / "stats.json"),
        "active_index_payload_sha256": stats.get("payload_sha256"),
        "passage_preselector": {"model": args.model_id, "revision": model_revision, "max_passages": args.max_passages},
        "artifacts": {
            pairs_path.name: {"bytes": pairs_path.stat().st_size, "sha256": sha256_file(pairs_path)},
            queries_path.name: {"bytes": queries_path.stat().st_size, "sha256": sha256_file(queries_path)},
        },
        "validation": {**validation, "skipped_empty_index_spans": skipped_empty_spans},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in (pairs_path, queries_path, manifest_path):
            archive.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path) as archive:
        if set(archive.namelist()) != {"p1_pairs.parquet", "p1_queries.parquet", "p1_manifest.json"} or archive.testzip() is not None:
            raise RuntimeError("P1 ZIP layout/CRC validation failed")
    print(json.dumps({**validation, "zip": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": sha256_file(zip_path)}, indent=2))


if __name__ == "__main__":
    main()
