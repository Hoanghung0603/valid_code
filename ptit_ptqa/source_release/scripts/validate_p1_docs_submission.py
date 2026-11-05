"""Independently validate a P1 docs-only ZIP against the accepted Exp122 control."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from itertools import zip_longest
from pathlib import Path

from build_p1_verified_docs_submission import CONTROL
from generate_translation_submission import iter_json_array, sha256_file


def validate(control_path: Path, candidate_path: Path) -> dict:
    if candidate_path.stat().st_size >= 100_000_000:
        raise ValueError("Candidate ZIP exceeds 100 MB")
    with zipfile.ZipFile(control_path) as control_zip, zipfile.ZipFile(candidate_path) as candidate_zip:
        if control_zip.namelist() != ["submission.json"] or candidate_zip.namelist() != ["submission.json"]:
            raise ValueError("Both ZIPs must contain only root submission.json")
        if candidate_zip.testzip() is not None:
            raise ValueError("Candidate ZIP CRC failed")
        with control_zip.open("submission.json") as control_binary, candidate_zip.open("submission.json") as candidate_binary:
            with io.TextIOWrapper(control_binary, encoding="utf-8") as control_text, io.TextIOWrapper(candidate_binary, encoding="utf-8") as candidate_text:
                control_rows = iter_json_array(control_text)
                candidate_rows = iter_json_array(candidate_text)
                query_ids = set()
                docs_before = docs_after = changed_queries = 0
                for base, treatment in zip_longest(control_rows, candidate_rows):
                    if base is None or treatment is None:
                        raise ValueError("Query count differs")
                    if set(treatment) != {"id", "relevant_docs", "relevant_chunks"}:
                        raise ValueError("Candidate row has unexpected keys")
                    qid = int(base["id"])
                    if qid in query_ids or qid != int(treatment["id"]):
                        raise ValueError("Missing, duplicate, or reordered query ID")
                    query_ids.add(qid)
                    if treatment["relevant_chunks"] != base["relevant_chunks"]:
                        raise ValueError(f"Chunks changed for query {qid}")
                    base_docs = [int(value) for value in base["relevant_docs"]]
                    docs = [int(value) for value in treatment["relevant_docs"]]
                    if len(base_docs) != 216 or len(set(base_docs)) != 216 or len(docs) != len(set(docs)):
                        raise ValueError(f"Invalid or duplicate documents for query {qid}")
                    base_set = set(base_docs)
                    kept = set(docs)
                    if not kept.issubset(base_set) or docs != [doc for doc in base_docs if doc in kept]:
                        raise ValueError(f"Docs were added or reordered for query {qid}")
                    owners = {int(chunk["doc_id"]) for chunk in base["relevant_chunks"]} & base_set
                    if not owners.issubset(kept):
                        raise ValueError(f"Chunk-owning docs removed for query {qid}")
                    docs_before += len(base_docs)
                    docs_after += len(docs)
                    changed_queries += docs != base_docs
    if len(query_ids) != 1200 or not changed_queries:
        raise ValueError("Expected 1,200 queries and a non-no-op treatment")
    return {
        "schema_version": "p1_docs_submission_validation_v1",
        "queries": len(query_ids), "docs_before": docs_before, "docs_after": docs_after,
        "removed_query_docs": docs_before - docs_after, "changed_queries": changed_queries,
        "chunks_exactly_equal_control": True, "owner_docs_preserved": True,
        "candidate_zip_bytes": candidate_path.stat().st_size,
        "candidate_zip_sha256": sha256_file(candidate_path),
        "control_zip_sha256": sha256_file(control_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write the independent validation report")
    args = parser.parse_args()
    report = validate(args.control, args.candidate)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
