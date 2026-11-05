"""Validate an Exp122-preserving multilingual translation submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from generate_translation_submission import (
    BASELINE_HASH_FROM_LOG,
    EXPECTED_QUERY_COUNT,
    iter_json_array,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, default=ROOT / "submission_exp122_exp121_add19_miss.zip")
    parser.add_argument("--submission-zip", type=Path, default=ROOT / "submission_exp136_translation_production_v1.zip")
    parser.add_argument("--submission-json", type=Path, default=ROOT / "submission_exp136_translation_production_v1.json")
    parser.add_argument("--translations-zip", type=Path, default=ROOT / "data/query_translations/translation_production_v1.zip")
    parser.add_argument("--official-corpus", type=Path, default=ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet")
    parser.add_argument("--report", type=Path, default=ROOT / "submission_exp136_translation_production_v1_manifest.json")
    return parser.parse_args()


def zipped_rows(archive: zipfile.ZipFile):
    with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
        yield from iter_json_array(text)


def main() -> None:
    args = args_parser()
    baseline_hash = sha256_file(args.baseline_zip)
    if baseline_hash != BASELINE_HASH_FROM_LOG:
        raise RuntimeError(f"Unexpected Exp122 baseline hash: {baseline_hash}")
    if args.submission_zip.stat().st_size >= 100_000_000:
        raise RuntimeError("Submission ZIP exceeds the decimal 100 MB limit")

    with zipfile.ZipFile(args.baseline_zip) as baseline_zip, zipfile.ZipFile(args.submission_zip) as submission_zip:
        if baseline_zip.namelist() != ["submission.json"] or submission_zip.namelist() != ["submission.json"]:
            raise RuntimeError("Baseline and submission ZIPs must each contain one root submission.json")
        if baseline_zip.testzip() is not None or submission_zip.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        payload_hash = hashlib.sha256()
        with submission_zip.open("submission.json") as payload:
            while block := payload.read(4 * 1024 * 1024):
                payload_hash.update(block)
        if payload_hash.hexdigest() != sha256_file(args.submission_json):
            raise RuntimeError("ZIP submission.json differs from the standalone JSON")
        official = set(pd.read_parquet(args.official_corpus, columns=["id"]).id.astype(int))
        baseline_iter = iter(zipped_rows(baseline_zip))
        submission_iter = iter(zipped_rows(submission_zip))
        query_count = changed_queries = total_chunks = 0
        total_tail_additions = 0
        per_query_docs: list[int] = []
        for base, treatment in zip(baseline_iter, submission_iter, strict=True):
            query_count += 1
            qid = int(base["id"])
            if treatment["id"] != qid or set(treatment) != {"id", "relevant_docs", "relevant_chunks"}:
                raise RuntimeError(f"Query/schema mismatch at row {query_count}")
            base_docs = base["relevant_docs"]
            docs = treatment["relevant_docs"]
            if len(docs) != 216 or len(set(docs)) != 216 or not set(docs).issubset(official):
                raise RuntimeError(f"Invalid docs for query {qid}")
            if docs[:150] != base_docs[:150]:
                raise RuntimeError(f"Locked Exp122 top-150 changed for query {qid}")
            if treatment["relevant_chunks"] != base["relevant_chunks"]:
                raise RuntimeError(f"Chunks differ from Exp122 for query {qid}")
            if any(type(doc_id) is not int for doc_id in docs):
                raise RuntimeError(f"Non-integer doc ID for query {qid}")
            total_tail_additions += sum(doc not in set(base_docs[:150]) for doc in docs[150:])
            changed_queries += docs != base_docs
            total_chunks += len(treatment["relevant_chunks"])
            per_query_docs.append(len(docs))
        if query_count != EXPECTED_QUERY_COUNT:
            raise RuntimeError(f"Expected {EXPECTED_QUERY_COUNT} queries, found {query_count}")
        try:
            next(baseline_iter)
            raise RuntimeError("Exp122 baseline has extra rows")
        except StopIteration:
            pass
        try:
            next(submission_iter)
            raise RuntimeError("Submission has extra rows")
        except StopIteration:
            pass

    with zipfile.ZipFile(args.translations_zip) as archive:
        translated = pd.read_parquet(io.BytesIO(archive.read("query_translated_production_v1.parquet")))
        audit = pd.read_parquet(io.BytesIO(archive.read("translation_production_audit_v1.parquet")))
    statuses = translated[["id", "query_en", "query_zh"]].merge(
        audit[["id", "status_en_effective", "status_zh"]], on="id", validate="one_to_one"
    )
    usage = {
        "en_ok": int((statuses.status_en_effective.eq("ok") & statuses.query_en.fillna("").str.strip().ne("")).sum()),
        "zh_ok": int((statuses.status_zh.eq("ok") & statuses.query_zh.fillna("").str.strip().ne("")).sum()),
        "en_fallback_to_original": int(statuses.status_en_effective.ne("ok").sum()),
        "zh_fallback_to_original": int(statuses.status_zh.ne("ok").sum()),
    }
    manifest = {
        "experiment": "Exp136 translation production dense-RRF docs-only",
        "status": "passed_validation_score_pending",
        "baseline_zip_sha256": baseline_hash,
        "submission_zip_sha256": sha256_file(args.submission_zip),
        "submission_json_sha256": sha256_file(args.submission_json),
        "zip_bytes": args.submission_zip.stat().st_size,
        "retrieval": {
            "model": "BAAI/bge-m3",
            "dense_k_chunks_per_view": 4096,
            "rrf_k": 60,
            "view_weights": {"original_vi": 1.0, "audited_en": 0.35, "audited_zh": 0.35},
            "output_docs": 216,
            "locked_exp122_docs": 150,
        },
        "queries": query_count,
        "docs_per_query": {"min": min(per_query_docs), "max": max(per_query_docs)},
        "locked_exp122_docs_per_query": 150,
        "tail_additions": total_tail_additions,
        "changed_queries": changed_queries,
        "chunks_preserved_from_exp122": True,
        "total_chunks": total_chunks,
        "translation_views": usage,
        "translation_zip_sha256": sha256_file(args.translations_zip),
        "validation": {
            "one_submission_json_member": True,
            "crc_ok": True,
            "zip_payload_matches_json": True,
            "official_doc_ids_only": True,
            "unique_docs_216_per_query": True,
            "exp122_top150_unchanged": True,
            "chunks_equal_exp122": True,
        },
        "score": None,
    }
    args.report.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
