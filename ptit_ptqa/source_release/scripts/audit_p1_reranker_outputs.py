"""Validate and summarize a downloaded P1 Kaggle reranker pilot output ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PACK = ROOT / "data/kaggle_inputs/p1_reranker_pilot_v1.zip"
OUTPUT_ZIP = ROOT / "data/kaggle_outputs/p1_reranker_pilot_outputs.zip"
AUDIT_DIR = ROOT / "data/audits/p1_reranker_pilot_v1"
REQUIRED = {
    "bge_scores.parquet",
    "qwen4b_scores.parquet",
    "p1_pair_scores.parquet",
    "p1_doc_scores.parquet",
    "p1_annotation_queue.parquet",
    "validation.json",
    "run_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pack", type=Path, default=INPUT_PACK)
    parser.add_argument("--outputs", type=Path, default=OUTPUT_ZIP)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pack(path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Input pack CRC failed")
        pairs = pd.read_parquet(io.BytesIO(archive.read("p1_pairs.parquet")))
        manifest = json.loads(archive.read("p1_manifest.json"))
    return pairs, manifest


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def validate_score_frames(
    input_pairs: pd.DataFrame,
    bge_scores: pd.DataFrame,
    qwen_scores: pd.DataFrame,
    pair_scores: pd.DataFrame,
    doc_scores: pd.DataFrame,
    docs_per_query: int = 216,
) -> None:
    """Verify score lineage from model outputs through pair/doc aggregation."""
    expected_ids = set(input_pairs.pair_id.astype(str))
    if input_pairs.pair_id.astype(str).duplicated().any():
        raise RuntimeError("Input pair IDs are not unique")
    for model_name, scores in (("bge", bge_scores), ("qwen4b", qwen_scores)):
        if scores.pair_id.astype(str).duplicated().any() or set(scores.pair_id.astype(str)) != expected_ids:
            raise RuntimeError(f"{model_name} score coverage differs from input pairs")
        if not np.isfinite(scores[["raw_score", "normalized_score"]].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite {model_name} scores")
        if not scores.normalized_score.between(0, 1).all():
            raise RuntimeError(f"{model_name} normalized score is outside [0, 1]")

    if pair_scores.pair_id.astype(str).duplicated().any() or set(pair_scores.pair_id.astype(str)) != expected_ids:
        raise RuntimeError("Pair-score coverage differs from input pairs")
    joined = input_pairs[["pair_id", "query_id", "doc_id", "passage_rank"]].copy()
    joined["pair_id"] = joined.pair_id.astype(str)
    for model_name, scores, short_name in (
        ("bge", bge_scores, "bge"),
        ("qwen4b", qwen_scores, "qwen"),
    ):
        model_cols = scores[["pair_id", "query_id", "doc_id", "passage_rank", "raw_score", "normalized_score"]].copy()
        model_cols["pair_id"] = model_cols.pair_id.astype(str)
        joined = joined.merge(
            model_cols.rename(columns={
                "query_id": f"{short_name}_query_id",
                "doc_id": f"{short_name}_doc_id",
                "passage_rank": f"{short_name}_passage_rank",
                "raw_score": f"{short_name}_raw_expected",
                "normalized_score": f"{short_name}_score_expected",
            }),
            on="pair_id", validate="one_to_one",
        )
        pair_cols = pair_scores[["pair_id", f"{short_name}_raw", f"{short_name}_score"]].copy()
        pair_cols["pair_id"] = pair_cols.pair_id.astype(str)
        joined = joined.merge(pair_cols, on="pair_id", validate="one_to_one")
        if not (
            joined.query_id.eq(joined[f"{short_name}_query_id"]).all()
            and joined.doc_id.eq(joined[f"{short_name}_doc_id"]).all()
            and joined.passage_rank.eq(joined[f"{short_name}_passage_rank"]).all()
        ):
            raise RuntimeError(f"{model_name} pair identifiers disagree with source pair IDs")
        if not np.allclose(joined[f"{short_name}_raw"], joined[f"{short_name}_raw_expected"], rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"{model_name} raw pair scores differ from model output")
        if not np.allclose(joined[f"{short_name}_score"], joined[f"{short_name}_score_expected"], rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"{model_name} normalized pair scores differ from model output")

    expected_owners = input_pairs.groupby(["query_id", "doc_id"], as_index=False).agg(
        doc_rank=("doc_rank", "first"), owner_doc=("owner_doc", "first")
    )
    if input_pairs.groupby(["query_id", "doc_id"]).owner_doc.nunique().max() != 1:
        raise RuntimeError("Owner flag differs among passages of the same query-document")
    expected_owners["doc_id"] = expected_owners.doc_id.astype(int)
    docs = doc_scores.merge(expected_owners, on=["query_id", "doc_id"], suffixes=("", "_expected"), validate="one_to_one")
    if len(docs) != len(doc_scores) or set(zip(doc_scores.query_id.astype(int), doc_scores.doc_id.astype(int))) != set(
        zip(expected_owners.query_id.astype(int), expected_owners.doc_id.astype(int))
    ):
        raise RuntimeError("Document score membership differs from input pairs")
    if not docs.doc_rank.eq(docs.doc_rank_expected).all() or not docs.owner_doc.eq(docs.owner_doc_expected).all():
        raise RuntimeError("Document control rank/owner flags differ from input pairs")
    counts = doc_scores.groupby("query_id").size()
    if len(counts) != input_pairs.query_id.nunique() or not counts.eq(docs_per_query).all():
        raise RuntimeError(f"Document score coverage is not exactly {docs_per_query} docs/query")
    for rank_column in ("bge_rank", "qwen_rank"):
        if rank_column not in doc_scores:
            raise RuntimeError(f"Missing {rank_column}")
        valid = doc_scores.groupby("query_id")[rank_column].apply(
            lambda ranks: set(ranks.astype(int)) == set(range(1, docs_per_query + 1))
        )
        if not valid.all():
            raise RuntimeError(f"{rank_column} is not a per-query rank permutation")

    aggregates = joined.groupby(["query_id", "doc_id"], as_index=False).agg(
        bge_score_expected=("bge_score", "max"), qwen_score_expected=("qwen_score", "max")
    )
    check = doc_scores.merge(aggregates, on=["query_id", "doc_id"], validate="one_to_one")
    for column in ("bge_score", "qwen_score"):
        expected = f"{column}_expected"
        if not np.allclose(check[column], check[expected], rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"Document {column} does not equal max passage score")


def model_provenance(scores: pd.DataFrame, model_key: str, expected_model_id: str) -> dict:
    required = {"model_key", "model_id", "model_revision"}
    if not required.issubset(scores.columns):
        raise RuntimeError(f"{model_key} scores miss provenance columns: {sorted(required - set(scores.columns))}")
    shard_columns = [column for column in ("gpu", "gpu_shard") if column in scores.columns]
    if not shard_columns:
        raise RuntimeError(f"{model_key} scores miss GPU shard provenance (expected gpu or gpu_shard)")
    if len(shard_columns) == 2 and not scores.gpu.astype(int).eq(scores.gpu_shard.astype(int)).all():
        raise RuntimeError(f"{model_key} gpu and gpu_shard provenance columns disagree")
    for column, expected in (("model_key", model_key), ("model_id", expected_model_id)):
        values = set(scores[column].astype(str))
        if values != {expected}:
            raise RuntimeError(f"Unexpected {model_key} {column} values: {sorted(values)}")
    revisions = set(scores.model_revision.fillna("").astype(str))
    if len(revisions) != 1 or "" in revisions or "unknown" in revisions:
        raise RuntimeError(f"{model_key} model revision is missing or inconsistent: {sorted(revisions)}")
    shard_column = shard_columns[0]
    gpus = set(scores[shard_column].astype(int))
    if gpus != {0, 1}:
        raise RuntimeError(f"{model_key} scores do not include both expected GPU shards: {sorted(gpus)}")
    return {
        "model_key": model_key,
        "model_id": expected_model_id,
        "model_revision": next(iter(revisions)),
        "gpu_shards": sorted(gpus),
        "rows": len(scores),
    }


def main() -> None:
    args = parse_args()
    if args.audit_dir.exists() and not args.force:
        raise FileExistsError(f"Audit directory exists: {args.audit_dir}; use --force only after review")
    input_pairs, input_manifest = read_pack(args.input_pack)
    with zipfile.ZipFile(args.outputs) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Kaggle output ZIP CRC failed")
        names = set(archive.namelist())
        if names != REQUIRED:
            raise RuntimeError(f"Unexpected output members: missing={REQUIRED-names}, extra={names-REQUIRED}")
        raw = {name: archive.read(name) for name in REQUIRED}
    validation = json.loads(raw["validation.json"])
    run_manifest = json.loads(raw["run_manifest.json"])
    expected_models = {"bge": "BAAI/bge-reranker-v2-m3", "qwen4b": "Qwen/Qwen3-Reranker-4B"}
    if run_manifest.get("schema_version") != "p1_reranker_pilot_outputs_v1":
        raise RuntimeError("Unexpected P1 run-manifest schema")
    if {key: value.get("id") for key, value in run_manifest.get("models", {}).items()} != expected_models:
        raise RuntimeError("Run-manifest model IDs differ from the P1 pilot specification")
    if run_manifest["input_manifest"]["artifacts"]["p1_pairs.parquet"]["sha256"] != input_manifest["artifacts"]["p1_pairs.parquet"]["sha256"]:
        raise RuntimeError("Kaggle run used a different P1 pairs artifact")

    bge_scores = pd.read_parquet(io.BytesIO(raw["bge_scores.parquet"]))
    qwen_scores = pd.read_parquet(io.BytesIO(raw["qwen4b_scores.parquet"]))
    pair_scores = pd.read_parquet(io.BytesIO(raw["p1_pair_scores.parquet"]))
    doc_scores = pd.read_parquet(io.BytesIO(raw["p1_doc_scores.parquet"]))
    provenance = {
        "bge": model_provenance(bge_scores, "bge", expected_models["bge"]),
        "qwen4b": model_provenance(qwen_scores, "qwen4b", expected_models["qwen4b"]),
    }
    validate_score_frames(input_pairs, bge_scores, qwen_scores, pair_scores, doc_scores)
    metrics = {}
    for key in ("bge", "qwen"):
        column = f"{key}_score"
        metrics[key] = {
            "owner_proxy_auc": binary_auc(doc_scores.owner_doc.to_numpy(), doc_scores[column].to_numpy()),
            "owner_mean": float(doc_scores.loc[doc_scores.owner_doc, column].mean()),
            "non_owner_mean": float(doc_scores.loc[~doc_scores.owner_doc, column].mean()),
            "owner_p10": float(doc_scores.loc[doc_scores.owner_doc, column].quantile(0.10)),
            "non_owner_p90": float(doc_scores.loc[~doc_scores.owner_doc, column].quantile(0.90)),
            "mean_rank_correlation_with_control": float(
                doc_scores.groupby("query_id").apply(
                    lambda frame: (-frame.doc_rank).corr(frame[column], method="spearman"),
                    include_groups=False,
                ).mean()
            ),
        }
    model_agreement = float(
        doc_scores.groupby("query_id").apply(
            lambda frame: frame.bge_score.corr(frame.qwen_score, method="spearman"),
            include_groups=False,
        ).mean()
    )
    audit = {
        "schema_version": "p1_reranker_pilot_audit_v1",
        "input_pack_sha256": sha256_bytes(args.input_pack.read_bytes()),
        "output_zip_sha256": sha256_bytes(args.outputs.read_bytes()),
        "queries": int(doc_scores.query_id.nunique()),
        "pairs": len(pair_scores),
        "query_docs": len(doc_scores),
        "validation": validation,
        "proxy_warning": "owner_doc is Exp122 evidence coverage, not BTC relevance ground truth",
        "models": metrics,
        "model_provenance": provenance,
        "mean_model_spearman": model_agreement,
        "members": {name: {"bytes": len(data), "sha256": sha256_bytes(data)} for name, data in raw.items()},
        "gate": "requires blinded rank-stratified human review generated from p1_doc_scores.parquet before choosing a candidate-pool trial cutoff",
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.audit_dir / "p1_annotation_queue.parquet").write_bytes(raw["p1_annotation_queue.parquet"])
    doc_scores.to_parquet(args.audit_dir / "p1_doc_scores.parquet", index=False, compression="zstd")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
