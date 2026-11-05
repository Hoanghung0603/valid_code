"""Build a fixed-Exp122-doc chunk treatment from ranked sentence passages.

The document list and per-query chunk count are locked to Exp122. Candidate
passages are sentence-aligned blocks cut at natural boundaries from selected
source documents, then ranked by dense relevance of their containing indexed
chunk plus lexical relevance of the passage itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_translation_submission import (
    BASELINE_HASH_FROM_LOG,
    EXPECTED_QUERY_COUNT,
    ROOT,
    iter_json_array,
    sha256_file,
)
from generate_translation_chunks_fixed_docs import (
    BASELINE_ZIP,
    INDEX_DIR,
    OFFICIAL_PATH,
    QUERY_PATH,
    fetch_vectors,
    load_source_texts,
    load_vector_shards,
    make_candidate_ids,
    read_baseline,
)


OUTPUT_PREFIX = ROOT / "submission_exp138_sentence_passages_fixed_docs"
MODEL_ID = "BAAI/bge-m3"
MAX_CHARS = 1500
MIN_CHARS = 120
MAX_SENTENCES = 6
RRF_K = 60
DENSE_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35
MAX_PER_DOC_FIRST_PASS = 1
MAX_PER_DOC_SECOND_PASS = 2

SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
STOP_WORDS = {
    "và", "là", "của", "có", "cho", "với", "tôi", "mình", "em", "anh", "chị",
    "bị", "được", "như", "thì", "khi", "ở", "làm", "nên", "nào", "sao", "bao",
    "nhiêu", "the", "a", "an", "is", "are", "of", "to", "for", "in", "on", "and",
    "or", "what", "how", "when", "where", "which", "can", "does", "do", "my", "i",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, default=BASELINE_ZIP)
    parser.add_argument("--queries", type=Path, default=QUERY_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--official-corpus", type=Path, default=OFFICIAL_PATH)
    parser.add_argument("--out-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-baseline-hash-mismatch", action="store_true")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS and len(token) > 1]


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Return exact source offsets for non-empty sentence-like spans."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in SENTENCE_SPLIT.finditer(text):
        start, end = cursor, match.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            ranges.append((start, end))
        cursor = match.end()
    start, end = cursor, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        ranges.append((start, end))
    return ranges


def make_passages(text: str) -> list[tuple[int, int, str]]:
    """Pack adjacent sentences at natural boundaries into medium passages."""
    sentences = sentence_ranges(text)
    passages: list[tuple[int, int, str]] = []
    if not sentences:
        return passages
    for start, end in _greedy_passage_ranges(sentences):
        passages.append((start, end, text[start:end]))
    return passages


def _greedy_passage_ranges(sentences: list[tuple[int, int]]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    if not sentences:
        return groups
    start: int | None = None
    end = count = 0

    def flush() -> None:
        if start is not None and end - start >= MIN_CHARS:
            groups.append((start, end))

    for s_start, s_end in sentences:
        sentence_len = s_end - s_start
        if sentence_len > MAX_CHARS:
            flush()
            start, end, count = None, 0, 0
            continue
        if start is None:
            start, end, count = s_start, s_end, 1
            continue
        if count >= MAX_SENTENCES or s_end - start > MAX_CHARS:
            flush()
            start, end, count = s_start, s_end, 1
        else:
            end, count = s_end, count + 1
    flush()
    return groups


def build_passage_ranges(texts: dict[int, str]) -> dict[int, list[tuple[int, int]]]:
    """Cache offsets only; keep one copy of source text in memory."""
    output: dict[int, list[tuple[int, int]]] = {}
    for i, (doc_id, text) in enumerate(texts.items(), 1):
        output[doc_id] = [(start, end) for start, end, _ in make_passages(text)]
        if i % 20_000 == 0 or i == len(texts):
            print(f"Segmented source docs {i:,}/{len(texts):,}", flush=True)
    return output


def score_lexical(query: str, passages: list[tuple[int, int, int, str]]) -> np.ndarray:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return np.zeros(len(passages), dtype=np.float64)
    token_lists = [tokenize(item[3]) for item in passages]
    document_frequency: Counter[str] = Counter()
    for tokens in token_lists:
        document_frequency.update(set(tokens))
    total = max(len(passages), 1)
    scores = np.zeros(len(passages), dtype=np.float64)
    for i, tokens in enumerate(token_lists):
        tf = Counter(tokens)
        if not tf:
            continue
        matched_weight = 0.0
        weighted_tf = 0.0
        for term, qtf in query_terms.items():
            count = tf.get(term, 0)
            if not count:
                continue
            idf = math.log1p((total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            matched_weight += idf * min(qtf, 2)
            weighted_tf += idf * min(count, 3)
        coverage = matched_weight / max(sum(math.log1p(total / (document_frequency[t] + 1)) for t in query_terms), 1e-9)
        length_norm = math.sqrt(max(len(tokens), 1) / 30.0)
        scores[i] = weighted_tf / max(length_norm, 0.7) + 0.5 * coverage
    return scores


def rank_passages(
    query: str,
    candidate_passages: list[tuple[int, int, int, str, int]],
    chunk_ranks: dict[int, int],
    output_count: int,
) -> list[tuple[int, int, int, str]]:
    """Fuse passage lexical rank with rank of its containing dense chunk."""
    if not candidate_passages:
        return []
    lexical_input = [(doc, start, end, text) for doc, start, end, text, _ in candidate_passages]
    lexical = score_lexical(query, lexical_input)
    lexical_order = np.argsort(-lexical, kind="stable")
    lexical_rank = np.empty(len(lexical_order), dtype=np.int32)
    lexical_rank[lexical_order] = np.arange(1, len(lexical_order) + 1, dtype=np.int32)

    fused = np.empty(len(candidate_passages), dtype=np.float64)
    for i, (_, _, _, _, chunk_id) in enumerate(candidate_passages):
        dense_rank = chunk_ranks.get(chunk_id, 2**20)
        fused[i] = DENSE_WEIGHT / (RRF_K + dense_rank) + LEXICAL_WEIGHT / (RRF_K + lexical_rank[i])
    order = np.argsort(-fused, kind="stable")

    # Greedy diversity: avoid redundant spans from the same document and
    # overlapping passages before permitting a second passage from a document.
    selected: list[tuple[int, int, int, str]] = []
    per_doc: Counter[int] = Counter()
    used_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for max_per_doc in (MAX_PER_DOC_FIRST_PASS, MAX_PER_DOC_SECOND_PASS):
        for idx in order:
            i = int(idx)
            doc_id, start, end, text, _ = candidate_passages[i]
            if per_doc[doc_id] >= max_per_doc:
                continue
            if any(start < old_end and end > old_start for old_start, old_end in used_ranges[doc_id]):
                continue
            if max_per_doc == MAX_PER_DOC_FIRST_PASS and per_doc[doc_id] >= 1:
                continue
            selected.append((doc_id, start, end, text))
            per_doc[doc_id] += 1
            used_ranges[doc_id].append((start, end))
            if len(selected) >= output_count:
                return selected
    return selected


def encode_queries(queries: pd.DataFrame, model_id: str, batch_size: int) -> tuple[np.ndarray, str]:
    from FlagEmbedding import BGEM3FlagModel
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=model_id)
    encoder = BGEM3FlagModel(model_path, use_fp16=True, devices="cuda:0")
    vectors: list[np.ndarray] = []
    values = queries["query"].astype(str).tolist()
    for start in range(0, len(values), batch_size):
        encoded = encoder.encode(
            values[start : start + batch_size],
            batch_size=batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        matrix = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        vectors.append(matrix)
    result = np.concatenate(vectors, axis=0)
    del encoder
    torch.cuda.empty_cache()
    return result, Path(model_path).name


def dense_chunk_ranks(ids: np.ndarray, vectors: np.ndarray, query_vector: np.ndarray) -> dict[int, int]:
    candidate_vectors = fetch_vectors(ids, vectors)
    matrix = torch.as_tensor(candidate_vectors, dtype=torch.float16, device="cuda:0")
    query = torch.as_tensor(query_vector, dtype=torch.float16, device="cuda:0")
    with torch.inference_mode():
        scores = (matrix @ query).float().cpu().numpy()
    del matrix, query
    torch.cuda.empty_cache()
    order = np.argsort(-scores, kind="stable")
    return {int(ids[position]): rank for rank, position in enumerate(order, 1)}


def map_passage_to_chunk(
    doc_id: int,
    passages: list[tuple[int, int, str]],
    doc_chunk_spans: np.ndarray,
    chunk_ranks: dict[int, int],
) -> list[tuple[int, int, int, str, int]]:
    result: list[tuple[int, int, int, str, int]] = []
    starts = doc_chunk_spans[:, 1]
    ends = doc_chunk_spans[:, 2]
    for start, end, text in passages:
        midpoint = (start + end) // 2
        right = int(np.searchsorted(starts, midpoint, side="right"))
        left = int(np.searchsorted(ends, midpoint, side="left"))
        contained_ids = doc_chunk_spans[left:right, 0]
        if len(contained_ids):
            best = min((int(chunk_id) for chunk_id in contained_ids), key=lambda chunk_id: chunk_ranks.get(chunk_id, 2**20))
        else:
            overlaps_left = int(np.searchsorted(ends, end, side="left"))
            overlaps_right = int(np.searchsorted(starts, start, side="right"))
            overlap_ids = doc_chunk_spans[overlaps_left:overlaps_right, 0]
            if len(overlap_ids):
                best = min((int(chunk_id) for chunk_id in overlap_ids), key=lambda chunk_id: chunk_ranks.get(chunk_id, 2**20))
            elif right:
                best = int(doc_chunk_spans[right - 1, 0])
            elif len(doc_chunk_spans):
                best = int(doc_chunk_spans[0, 0])
            else:
                continue
        result.append((doc_id, start, end, text, best))
    return result


def main() -> None:
    args = parse_args()
    out_json = args.out_prefix.with_suffix(".json")
    out_zip = args.out_prefix.with_suffix(".zip")
    out_manifest = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    existing = [path for path in (out_json, out_zip, out_manifest) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}; use --force")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense candidate passage scoring")

    baseline_rows, baseline_json_bytes, baseline_hash = read_baseline(args.baseline_zip)
    if baseline_hash != BASELINE_HASH_FROM_LOG and not args.allow_baseline_hash_mismatch:
        raise RuntimeError(f"Unexpected Exp122 baseline hash: {baseline_hash}")
    queries = pd.read_parquet(args.queries).sort_values("id").reset_index(drop=True)
    if len(queries) != EXPECTED_QUERY_COUNT or queries.id.astype(int).tolist() != list(range(1, EXPECTED_QUERY_COUNT + 1)):
        raise RuntimeError("BTC query file must contain ordered IDs 1..1200")
    metadata = pd.read_parquet(args.index_dir / "chunks_metadata.parquet", columns=["chunk_id", "doc_id", "start", "end"])
    metadata = metadata.sort_values("chunk_id").reset_index(drop=True)
    stats = json.loads((args.index_dir / "stats.json").read_text(encoding="utf-8"))
    if len(metadata) != int(stats["total_chunks"]) or not np.array_equal(metadata.chunk_id.to_numpy(), np.arange(len(metadata))):
        raise RuntimeError("Chunk metadata is not contiguous/aligned with active index")
    candidate_ids, _ = make_candidate_ids(baseline_rows, metadata)
    needed_docs = {int(doc) for row in baseline_rows for doc in row["relevant_docs"]}
    official_ids = set(pd.read_parquet(args.official_corpus, columns=["id"]).id.astype(int))
    if not needed_docs.issubset(official_ids):
        raise RuntimeError("Exp122 contains IDs outside the official BTC corpus")

    print(f"Loading source text for {len(needed_docs):,} unique Exp122 docs...", flush=True)
    texts = load_source_texts(args.index_dir, needed_docs)
    print("Packing source into sentence-aligned passages...", flush=True)
    passages_by_doc = build_passage_ranges(texts)
    print("Encoding 1,200 original queries with BGE-M3...", flush=True)
    query_vectors, model_revision = encode_queries(queries, args.model_id, args.batch_size)
    shards = load_vector_shards(args.index_dir, len(metadata), query_vectors.shape[1])

    needed_docs = {int(doc) for row in baseline_rows for doc in row["relevant_docs"]}
    selected_metadata = metadata[metadata.doc_id.isin(needed_docs)]
    docs_to_chunks: dict[int, np.ndarray] = {}
    for group_number, (doc_id, frame) in enumerate(selected_metadata.groupby("doc_id", sort=False), 1):
        docs_to_chunks[int(doc_id)] = frame[["chunk_id", "start", "end"]].sort_values("start").to_numpy(dtype=np.int64)
        if group_number % 40_000 == 0 or group_number == len(needed_docs):
            print(f"Prepared selected-doc chunk maps {group_number:,}/{len(needed_docs):,}", flush=True)
    output: list[dict] = []
    selected_lengths: list[int] = []
    fallbacks = 0
    for qi, base in enumerate(baseline_rows):
        query_started = time.perf_counter()
        qid = int(base["id"])
        ids = candidate_ids[qi]
        if qi == 0:
            print(f"Query 1 candidate indexed chunks: {len(ids):,}", flush=True)
        ranks = dense_chunk_ranks(ids, shards, query_vectors[qi])
        if qi == 0:
            print(f"Query 1 dense rank time: {time.perf_counter() - query_started:.1f}s", flush=True)
        pool: list[tuple[int, int, int, str, int]] = []
        for doc in base["relevant_docs"]:
            doc_id = int(doc)
            doc_chunk_spans = docs_to_chunks[doc_id]
            local_ranges = passages_by_doc[doc_id]
            if not local_ranges:
                continue
            source_text = texts[doc_id]
            local_passages = [(start, end, source_text[start:end]) for start, end in local_ranges]
            mapped = map_passage_to_chunk(doc_id, local_passages, doc_chunk_spans, ranks)
            pool.extend(mapped)
        if qi == 0:
            print(f"Query 1 passage mapping time: {time.perf_counter() - query_started:.1f}s; pool={len(pool):,}", flush=True)
        wanted = len(base["relevant_chunks"])
        selected = rank_passages(str(queries.iloc[qi]["query"]), pool, ranks, wanted)
        if qi == 0:
            print(f"Query 1 lexical/rank time: {time.perf_counter() - query_started:.1f}s", flush=True)
        if len(selected) < wanted:
            fallbacks += wanted - len(selected)
            existing_keys = {(doc_id, start, end) for doc_id, start, end, _ in selected}
            for chunk in base["relevant_chunks"]:
                if len(selected) >= wanted:
                    break
                doc_id = int(chunk["doc_id"])
                text = str(chunk["chunk_text"])
                source = texts.get(doc_id, "")
                start = source.find(text)
                end = start + len(text) if start >= 0 else -1
                key = (doc_id, start, end)
                if key not in existing_keys:
                    selected.append((doc_id, start, end, text))
                    existing_keys.add(key)
        chunks = [{"doc_id": doc_id, "chunk_text": text} for doc_id, _, _, text in selected[:wanted]]
        if len(chunks) != wanted or any(item["doc_id"] not in base["relevant_docs"] for item in chunks):
            raise RuntimeError(f"Invalid chunk output for query {qid}")
        if list(base["relevant_docs"]) != list(baseline_rows[qi]["relevant_docs"]):
            raise RuntimeError(f"Document list changed for query {qid}")
        selected_lengths.extend(len(item["chunk_text"]) for item in chunks)
        output.append({"id": qid, "relevant_docs": base["relevant_docs"], "relevant_chunks": chunks})
        if qi == 0 or (qi + 1) % 50 == 0 or qi + 1 == len(baseline_rows):
            print(f"Built query {qi + 1}/{len(baseline_rows)}; passage_pool={len(pool)}; elapsed={time.perf_counter()-query_started:.1f}s", flush=True)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_LZMA, compresslevel=9) as archive:
        archive.write(out_json, arcname="submission.json")
    if out_zip.stat().st_size >= 100_000_000:
        raise RuntimeError(f"Submission exceeds decimal 100 MB limit: {out_zip.stat().st_size:,}")
    with zipfile.ZipFile(out_zip) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Output ZIP layout/CRC validation failed")

    manifest = {
        "experiment": "Exp138 sentence-aligned passage selection; fixed Exp122 docs and chunk counts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_zip_sha256": baseline_hash,
        "baseline_json_bytes": baseline_json_bytes,
        "active_index_model": "BAAI/bge-m3",
        "query_model": args.model_id,
        "query_model_revision": model_revision,
        "passage_builder": {"max_chars": MAX_CHARS, "min_chars": MIN_CHARS, "max_sentences": MAX_SENTENCES},
        "ranking": {"dense_parent_weight": DENSE_WEIGHT, "lexical_passage_weight": LEXICAL_WEIGHT, "rrf_k": RRF_K, "max_passages_per_doc": MAX_PER_DOC_SECOND_PASS},
        "docs_unchanged_from_exp122": True,
        "chunk_count_unchanged_per_query": True,
        "fallback_chunk_count": fallbacks,
        "queries": len(output),
        "selected_chunk_char_length": {"min": min(selected_lengths), "median": statistics.median(selected_lengths), "p90": float(np.percentile(selected_lengths, 90)), "max": max(selected_lengths)},
        "zip_bytes": out_zip.stat().st_size,
        "zip_sha256": sha256_file(out_zip),
        "json_sha256": sha256_file(out_json),
        "score": None,
        "validation": {"all_docs_exactly_equal_exp122": True, "chunks_owned_by_returned_docs": True, "source_text_only_except_exp122_fallback": True, "same_chunk_count_as_exp122": True, "zip_crc_ok": True, "under_100mb": True},
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "zip": str(out_zip), "json": str(out_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
