"""Create a smaller, still-blinded P1 human review queue (2 rows/query).

The selection uses predefined pilot sampling reasons, never weak labels. This
is a sampled candidate pool, not BTC qrels or a representative corpus sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from generate_translation_submission import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / "data/audits/p1_reranker_pilot_audit/annotation_review"
REASONS = ("qwen_rank_1", "qwen_rank_180")


def select_ids(key: pd.DataFrame) -> pd.DataFrame:
    required = {"annotation_id", "query_id", "split", "sampling_reasons"}
    if not required.issubset(key.columns):
        raise ValueError(f"Missing key columns: {sorted(required - set(key.columns))}")
    if key.annotation_id.duplicated().any():
        raise ValueError("Duplicate annotation IDs in key")
    selected = key[key.sampling_reasons.map(
        lambda raw: any(reason in str(raw).split(";") for reason in REASONS)
    )].copy()
    if selected.query_id.nunique() != 120 or len(selected) != 240:
        raise ValueError("Expected exactly two selected rows for each of 120 queries")
    if not selected.groupby("query_id").size().eq(2).all():
        raise ValueError("Selection does not cover each query twice")
    split_counts = selected.drop_duplicates("query_id").split.value_counts().to_dict()
    if split_counts != {"development": 80, "holdout": 40}:
        raise ValueError(f"Unexpected query split: {split_counts}")
    return selected[["annotation_id", "query_id", "split"]]


def build_review(blind: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    visible = ["annotation_id", "query_id", "query", "doc_id", "passage_1", "passage_2", "label", "notes"]
    if set(blind.columns) != set(visible) or blind.annotation_id.duplicated().any():
        raise ValueError("Unexpected blinded CSV schema or duplicate IDs")
    if blind[["label", "notes"]].notna().any().any():
        raise ValueError("Source blinded CSV already contains reviewer annotations")
    review = blind.merge(selected[["annotation_id"]], on="annotation_id", how="inner", validate="one_to_one")
    if len(review) != len(selected) or set(review.annotation_id) != set(selected.annotation_id):
        raise ValueError("Blind CSV does not cover selected annotation IDs")
    return review[visible].sample(frac=1, random_state=20261024).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW / "compact_human_review")
    args = parser.parse_args()
    key_path = args.review_dir / "p1_annotation_key.parquet"
    blind_path = args.review_dir / "p1_annotation_blinded.csv"
    selected = select_ids(pd.read_parquet(key_path))
    review = build_review(pd.read_csv(blind_path), selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "p1_compact_human_review_240.csv"
    manifest_path = args.output_dir / "manifest.json"
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError("Compact review output exists; inspect it before replacing")
    review.to_csv(csv_path, index=False, encoding="utf-8-sig")
    selected_ids = sorted(selected.annotation_id.astype(str))
    manifest = {
        "schema_version": "p1_compact_human_review_v1",
        "selection_reasons": list(REASONS),
        "source_key_sha256": sha256_file(key_path),
        "source_blind_csv_sha256": sha256_file(blind_path),
        "review_csv_sha256": sha256_file(csv_path),
        "selected_annotation_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "rows": len(review), "queries": int(selected.query_id.nunique()),
        "development_queries": 80, "holdout_queries": 40,
        "contains_model_labels": False, "contains_scores_or_rank": False,
        "interpretation": "Targeted blind review only; not BTC qrels or corpus-wide relevance metrics.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
