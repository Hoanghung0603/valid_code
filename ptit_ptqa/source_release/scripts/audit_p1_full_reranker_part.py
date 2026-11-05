"""Independently audit one Kaggle P1 full-reranker score partition."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_REVISIONS = {
    "bge": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "qwen4b": ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5"),
}
MEMBERS = {
    "p1_pair_scores.parquet",
    "p1_doc_scores.parquet",
    "validation.json",
    "run_manifest.json",
}
LINEAGE_COLUMNS = [
    "pair_id", "query_id", "doc_id", "doc_rank", "owner_doc",
    "passage_rank", "language", "dense_score",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_partition(pairs: pd.DataFrame, part_index: int, part_count: int) -> pd.DataFrame:
    if part_count < 1 or not 0 <= part_index < part_count:
        raise ValueError("Require 0 <= part_index < part_count")
    query_ids = sorted(pairs.query_id.astype(int).unique())
    owned = {query_id for i, query_id in enumerate(query_ids) if i % part_count == part_index}
    return pairs[pairs.query_id.isin(owned)].copy()


def audit_frames(expected: pd.DataFrame, scored: pd.DataFrame, docs: pd.DataFrame, model: str) -> dict:
    if expected.pair_id.duplicated().any() or scored.pair_id.duplicated().any():
        raise RuntimeError("Duplicate pair_id in input or scores")
    if len(scored) != len(expected):
        raise RuntimeError(f"Pair coverage count mismatch: {len(scored)} != {len(expected)}")
    for frame, name in ((expected, "input"), (scored, "scores")):
        absent = set(LINEAGE_COLUMNS) - set(frame.columns)
        if absent:
            raise RuntimeError(f"{name} misses lineage columns: {sorted(absent)}")
    left = expected[LINEAGE_COLUMNS].sort_values("pair_id").reset_index(drop=True)
    right = scored[LINEAGE_COLUMNS].sort_values("pair_id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise RuntimeError("Pair IDs or source lineage differ from full input") from exc

    raw_col, score_col, rank_col = f"{model}_raw", f"{model}_score", f"{model}_rank"
    if not {raw_col, score_col, "input_tokens", "model_revision"}.issubset(scored.columns):
        raise RuntimeError("Scores miss model columns")
    if not {raw_col, score_col, rank_col, "passage_count"}.issubset(docs.columns):
        raise RuntimeError("Document scores miss model columns")
    raw = scored[raw_col].to_numpy(dtype=float)
    normalized = scored[score_col].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise RuntimeError("Nonfinite pair scores")
    if ((normalized < 0) | (normalized > 1)).any() or (scored.input_tokens.to_numpy(dtype=int) <= 0).any():
        raise RuntimeError("Invalid normalized score or input token count")
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(raw, -100, 100)))
    if not np.allclose(normalized, sigmoid, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Normalized pair scores do not match raw-score sigmoid")
    if scored.model_revision.nunique() != 1 or scored.model_revision.iloc[0] != MODEL_REVISIONS[model][1]:
        raise RuntimeError("Unexpected model revision in pair scores")

    grouped = scored.groupby(["query_id", "doc_id"], as_index=False).agg(
        expected_score=(score_col, "max"), expected_raw=(raw_col, "max"),
        expected_count=("pair_id", "size"), expected_owner=("owner_doc", "first"),
        expected_doc_rank=("doc_rank", "first"),
    )
    if docs.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Duplicate query-doc scores")
    joined = docs.merge(grouped, on=["query_id", "doc_id"], how="outer", indicator=True, validate="one_to_one")
    if len(joined) != len(grouped) or not joined._merge.eq("both").all():
        raise RuntimeError("Document coverage differs from scored pairs")
    if not np.allclose(joined[score_col], joined.expected_score, rtol=0, atol=1e-7):
        raise RuntimeError("Document score differs from maximum passage score")
    if not np.allclose(joined[raw_col], joined.expected_raw, rtol=0, atol=1e-6):
        raise RuntimeError("Document raw score differs from maximum passage raw score")
    if not joined.passage_count.eq(joined.expected_count).all():
        raise RuntimeError("Document passage_count mismatch")
    if not joined.owner_doc.eq(joined.expected_owner).all() or not joined.doc_rank.eq(joined.expected_doc_rank).all():
        raise RuntimeError("Document owner or original rank mismatch")
    counts = docs.groupby("query_id").size()
    if counts.empty or counts.min() != 216 or counts.max() != 216:
        raise RuntimeError("Expected exactly 216 documents per query")
    expected_ranks = set(range(1, 217))
    if any(set(group[rank_col].astype(int)) != expected_ranks for _, group in docs.groupby("query_id")):
        raise RuntimeError("Reranked documents do not contain ranks 1..216")
    return {
        "queries": int(docs.query_id.nunique()),
        "pairs": len(scored),
        "query_docs": len(docs),
        "lineage_exact": True,
        "doc_aggregation_exact": True,
        "nonfinite_scores": 0,
        "model_revision": str(scored.model_revision.iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pairs", type=Path, required=True)
    parser.add_argument("--part", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--part-index", type=int, required=True)
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    input_hash = sha256_file(args.input_pairs)
    expected = expected_partition(pd.read_parquet(args.input_pairs, columns=LINEAGE_COLUMNS), args.part_index, args.part_count)
    with zipfile.ZipFile(args.part) as archive:
        if set(archive.namelist()) != MEMBERS or archive.testzip() is not None:
            raise RuntimeError("Output ZIP members or CRC invalid")
        scored = pd.read_parquet(io.BytesIO(archive.read("p1_pair_scores.parquet")))
        docs = pd.read_parquet(io.BytesIO(archive.read("p1_doc_scores.parquet")))
        validation = json.loads(archive.read("validation.json"))
        manifest = json.loads(archive.read("run_manifest.json"))
    if manifest.get("schema_version") != "p1_full_reranker_part_v1" or manifest.get("validation") != validation:
        raise RuntimeError("Run manifest and validation disagree")
    if validation.get("model_key") != args.model or validation.get("model_id") != MODEL_REVISIONS[args.model][0]:
        raise RuntimeError("Unexpected model identity")
    if validation.get("model_revision") != MODEL_REVISIONS[args.model][1]:
        raise RuntimeError("Unexpected model revision")
    if int(validation.get("part_index", -1)) != args.part_index or int(validation.get("part_count", -1)) != args.part_count:
        raise RuntimeError("Wrong query partition")
    if validation.get("full_input_pairs_sha256") != input_hash:
        raise RuntimeError("Input pair payload hash mismatch")
    report = audit_frames(expected, scored, docs, args.model)
    if report["pairs"] != validation.get("pairs") or report["query_docs"] != validation.get("query_docs"):
        raise RuntimeError("Validation counts differ from audited payload")
    report.update({
        "schema_version": "p1_full_reranker_part_audit_v1",
        "model": args.model,
        "part_index": args.part_index,
        "part_count": args.part_count,
        "input_pairs_sha256": input_hash,
        "part_zip_sha256": sha256_file(args.part),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
