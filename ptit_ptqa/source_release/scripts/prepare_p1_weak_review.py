"""Create a blinded human-review CSV from P1 Qwen3-14B weak judgments."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from generate_translation_submission import sha256_file
from prepare_p1_annotation_review import allocate_stratified_holdout, stable_hash


ROOT = Path(__file__).resolve().parents[1]
PILOT_QUERIES = ROOT / "data/kaggle_inputs/p1_reranker_pilot_v1/p1_queries.parquet"
OUT = ROOT / "data/audits/p1_reranker_pilot_v1/weak_review"
MEMBER = "p1_weak_judgments.parquet"
HIDDEN = {
    "label", "status", "evidence", "evidence_valid", "reason", "raw_output", "error",
    "owner_doc", "doc_rank", "bge_score", "qwen_score", "bge_rank", "qwen_rank",
    "rank_disagreement", "selection_reason", "dense_score", "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-outputs", type=Path, required=True)
    parser.add_argument("--queries", type=Path, default=PILOT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seed", default="p1-review-v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_weak(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or MEMBER not in archive.namelist():
            raise RuntimeError(f"Weak output ZIP is invalid or misses {MEMBER}")
        return pd.read_parquet(io.BytesIO(archive.read(MEMBER)))


def select_review_rows(frame: pd.DataFrame, seed: str) -> pd.DataFrame:
    """Select one core and up to two complementary rows per query."""
    selected = []
    for qid, group in frame.groupby("query_id", sort=True):
        group = group.copy()
        group["_hash"] = [stable_hash(seed, judgment_id) for judgment_id in group.judgment_id]
        needs_review = group[group.status.ne("ok")]
        core_pool = needs_review if not needs_review.empty else group
        core = core_pool.sort_values(["rank_disagreement", "_hash"], ascending=[False, True]).iloc[[0]].copy()
        core["review_tier"] = "core"
        selected.append(core)

        remaining = group[~group.judgment_id.isin(core.judgment_id)].copy()
        if remaining.empty:
            continue
        complementary = []
        random_pool = remaining[remaining.selection_reason.astype(str).str.contains("random", regex=False)]
        if not random_pool.empty:
            complementary.append(random_pool.sort_values("_hash").iloc[[0]])
        used = set(pd.concat(complementary).judgment_id) if complementary else set()
        anchor_pool = remaining[
            remaining.selection_reason.astype(str).str.contains("rank_anchor", regex=False)
            & ~remaining.judgment_id.isin(used)
        ]
        if not anchor_pool.empty:
            complementary.append(anchor_pool.sort_values(["rank_disagreement", "_hash"], ascending=[False, True]).iloc[[0]])
        used = set(pd.concat(complementary).judgment_id) if complementary else set()
        if len(complementary) < 2:
            fallback = remaining[~remaining.judgment_id.isin(used)].sort_values(
                ["rank_disagreement", "_hash"], ascending=[False, True]
            ).head(2 - len(complementary))
            if not fallback.empty:
                complementary.append(fallback)
        if complementary:
            extra = pd.concat(complementary).drop_duplicates("judgment_id").head(2).copy()
            extra["review_tier"] = "extended"
            selected.append(extra)
    result = pd.concat(selected, ignore_index=True).drop(columns="_hash")
    if result.duplicated("judgment_id").any() or result.groupby("query_id").size().max() > 3:
        raise RuntimeError("Weak review selector produced duplicate or oversized query groups")
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    paths = {
        "review": output / "p1_weak_review_blinded.csv",
        "key": output / "p1_weak_review_key.parquet",
        "manifest": output / "manifest.json",
    }
    if any(path.exists() for path in paths.values()) and not args.force:
        raise FileExistsError("Weak-review outputs exist; use --force only before labels are entered")
    weak = load_weak(args.weak_outputs)
    required = {"judgment_id", "query_id", "doc_id", "query", "passages", "status", "selection_reason", "rank_disagreement"}
    if not required.issubset(weak.columns) or weak.judgment_id.duplicated().any():
        raise RuntimeError("Weak judgments miss review fields or contain duplicate IDs")
    queries = pd.read_parquet(args.queries)
    holdout = allocate_stratified_holdout(queries, 40, args.seed)
    chosen = select_review_rows(weak, args.seed)
    chosen["split"] = chosen.query_id.astype(int).map(lambda qid: "holdout" if qid in holdout else "development")
    query_order = {qid: rank for rank, qid in enumerate(sorted(chosen.query_id.unique(), key=lambda value: stable_hash(args.seed, value)))}
    chosen["_query_order"] = chosen.query_id.map(query_order)
    chosen["_tier_order"] = chosen.review_tier.map({"core": 0, "extended": 1})
    chosen = chosen.sort_values(["_tier_order", "_query_order", "judgment_id"]).reset_index(drop=True)

    review = chosen[["judgment_id", "review_tier", "query_id", "query", "doc_id", "passages"]].copy()
    review["label"] = ""
    review["notes"] = ""
    if HIDDEN.intersection(review.columns):
        raise RuntimeError(f"Blinded review leaks hidden fields: {sorted(HIDDEN.intersection(review.columns))}")
    key = chosen.drop(columns=["_query_order", "_tier_order", "query", "passages"]).copy()

    output.mkdir(parents=True, exist_ok=True)
    review.to_csv(paths["review"], index=False, encoding="utf-8-sig")
    key.to_parquet(paths["key"], index=False, compression="zstd")
    manifest = {
        "schema_version": "p1_weak_review_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weak_outputs_sha256": sha256_file(args.weak_outputs),
        "queries_sha256": sha256_file(args.queries),
        "rows": len(review),
        "core_rows": int(review.review_tier.eq("core").sum()),
        "queries": int(review.query_id.nunique()),
        "development_queries": int(chosen.loc[chosen.split == "development", "query_id"].nunique()),
        "holdout_queries": int(chosen.loc[chosen.split == "holdout", "query_id"].nunique()),
        "allowed_labels": ["direct", "related", "irrelevant"],
        "hidden_features_absent": True,
        "artifacts": {
            "review": {"sha256": sha256_file(paths["review"]), "bytes": paths["review"].stat().st_size},
            "key": {"sha256": sha256_file(paths["key"]), "bytes": paths["key"].stat().st_size},
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
