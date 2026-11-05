"""Independently validate an Exp138 sentence-passage ZIP against Exp122."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import zipfile
from pathlib import Path

from generate_translation_submission import EXPECTED_QUERY_COUNT, iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "submission_exp122_exp121_add19_miss.zip"
DEFAULT_SUBMISSION = ROOT / "submission_exp138_sentence_passages_fixed_docs.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--submission-zip", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--max-passage-chars", type=int, default=1500)
    return parser.parse_args()


def rows(archive: zipfile.ZipFile):
    with archive.open("submission.json") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8") as text:
            yield from iter_json_array(text)


def main() -> None:
    args = parse_args()
    with zipfile.ZipFile(args.baseline_zip) as baseline, zipfile.ZipFile(args.submission_zip) as treatment:
        for name, archive in (("baseline", baseline), ("submission", treatment)):
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError(f"{name}: ZIP layout or CRC validation failed")
        baseline_rows, treatment_rows = rows(baseline), rows(treatment)
        seen = 0
        chunk_count = 0
        lengths: list[int] = []
        for expected_id in range(1, EXPECTED_QUERY_COUNT + 1):
            base = next(baseline_rows, None)
            row = next(treatment_rows, None)
            if base is None or row is None:
                raise RuntimeError(f"Missing query row at expected ID {expected_id}")
            if int(base["id"]) != expected_id or int(row["id"]) != expected_id:
                raise RuntimeError(f"Unexpected query ID/order at {expected_id}")
            docs = [int(doc) for doc in row["relevant_docs"]]
            if docs != [int(doc) for doc in base["relevant_docs"]]:
                raise RuntimeError(f"Document membership/order changed for query {expected_id}")
            baseline_chunks = base["relevant_chunks"]
            chunks = row["relevant_chunks"]
            if len(chunks) != len(baseline_chunks):
                raise RuntimeError(f"Chunk count changed for query {expected_id}")
            if any(int(chunk["doc_id"]) not in set(docs) for chunk in chunks):
                raise RuntimeError(f"Chunk points to a document outside returned docs for query {expected_id}")
            for chunk in chunks:
                text = str(chunk["chunk_text"])
                if not text.strip() or len(text) > args.max_passage_chars:
                    raise RuntimeError(f"Empty/overlong passage in query {expected_id}: {len(text)} chars")
                lengths.append(len(text))
            seen += 1
            chunk_count += len(chunks)
        if next(baseline_rows, None) is not None or next(treatment_rows, None) is not None:
            raise RuntimeError("Unexpected extra query rows")
    print(json.dumps({
        "queries": seen,
        "chunks": chunk_count,
        "docs_and_order_exactly_exp122": True,
        "chunk_count_per_query_exactly_exp122": True,
        "all_chunks_owned_by_returned_docs": True,
        "all_passages_nonempty_and_within_limit": True,
        "passage_chars": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "p90": __import__("numpy").percentile(lengths, 90).item(),
            "max": max(lengths),
        },
        "zip_sha256": sha256_file(args.submission_zip),
        "zip_bytes": args.submission_zip.stat().st_size,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
