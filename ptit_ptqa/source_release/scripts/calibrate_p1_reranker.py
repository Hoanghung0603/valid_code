"""Calibrate P1 reranker/model policy from weak and optional blinded human labels.

Weak labels can recommend what to inspect or run at full scale, but this script
does not call them verified relevance. BTC supplies no qrels/dev labels, and the
local synthetic dev set is intentionally not used. Human review covers only a
sampled candidate pool; its metrics are not corpus-wide recall or a BTC-score
guarantee. The operational human-review gate is not an official relevance oracle.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from generate_translation_submission import sha256_file
from prepare_p1_annotation_review import allocate_stratified_holdout


ROOT = Path(__file__).resolve().parents[1]
PILOT_QUERIES = ROOT / "data/kaggle_inputs/p1_reranker_pilot_v1/p1_queries.parquet"
OUT = ROOT / "data/audits/p1_reranker_pilot_v1/calibration"
ALLOWED_LABELS = {"direct", "related", "irrelevant"}


def json_finite(value):
    """Write undefined metrics as JSON null, never nonstandard NaN/Infinity."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker-outputs", type=Path, required=True)
    parser.add_argument("--weak-outputs", type=Path, help="Optional weak-label ZIP; never treated as BTC ground truth")
    parser.add_argument("--human-labels", type=Path)
    parser.add_argument("--model-labels", type=Path, help="Optional externally generated labels; kept exploratory and excluded from human-only metrics")
    parser.add_argument("--queries", type=Path, default=PILOT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seed", default="p1-review-v1")
    parser.add_argument("--min-direct-recall", type=float, default=0.97)
    parser.add_argument("--min-human-rows", type=int, default=100)
    parser.add_argument("--min-human-holdout-queries", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_zip_parquet(path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or member not in archive.namelist():
            raise RuntimeError(f"{path} is invalid or misses {member}")
        return pd.read_parquet(io.BytesIO(archive.read(member)))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return float((1 + beta2) * precision * recall / denominator) if denominator else 0.0


def rank_policy_metrics(frame: pd.DataFrame, rank_column: str, keep_top: int) -> dict:
    strict = frame[frame.label.isin(["direct", "irrelevant"])].copy()
    if strict.empty:
        return {"rows": 0, "positives": 0, "negatives": 0, "precision": None, "recall": None, "f2": None, "irrelevant_rejection": None}
    truth = strict.label.eq("direct").to_numpy()
    predicted = ((strict[rank_column].to_numpy() <= keep_top) | strict.owner_doc.to_numpy(dtype=bool))
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "rows": len(strict), "positives": int(truth.sum()), "negatives": int((~truth).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f2": fbeta(precision, recall),
        "irrelevant_rejection": tn / (tn + fp) if tn + fp else 0.0,
    }


def discrimination_metrics(frame: pd.DataFrame, score_column: str) -> dict:
    strict = frame[frame.label.isin(["direct", "irrelevant"])]
    broad = frame[frame.label.isin(ALLOWED_LABELS)]
    strict_labels = strict.label.eq("direct").to_numpy()
    broad_labels = broad.label.ne("irrelevant").to_numpy()
    return {
        "strict_rows": len(strict),
        "strict_direct": int(strict_labels.sum()),
        "strict_irrelevant": int((~strict_labels).sum()),
        "strict_auc": binary_auc(strict_labels, strict[score_column].to_numpy()),
        "strict_average_precision": average_precision(strict_labels, strict[score_column].to_numpy()),
        "broad_rows": len(broad),
        "broad_positive": int(broad_labels.sum()),
        "broad_auc": binary_auc(broad_labels, broad[score_column].to_numpy()),
        "broad_average_precision": average_precision(broad_labels, broad[score_column].to_numpy()),
    }


def choose_keep_top(frame: pd.DataFrame, rank_column: str, min_recall: float) -> tuple[int, list[dict]]:
    candidates = sorted(set(list(range(40, 216, 5)) + [216]))
    curve = [{"keep_top": value, **rank_policy_metrics(frame, rank_column, value)} for value in candidates]
    eligible = [row for row in curve if row["recall"] is not None and row["recall"] >= min_recall]
    if not eligible:
        return 216, curve
    winner = max(eligible, key=lambda row: (row["f2"], row["irrelevant_rejection"], -row["keep_top"]))
    return int(winner["keep_top"]), curve


def build_model_reports(development: pd.DataFrame, holdout: pd.DataFrame, min_recall: float) -> dict:
    models = {
        "bge": {"score": "bge_score", "rank": "bge_rank"},
        "qwen4b": {"score": "qwen_score", "rank": "qwen_rank"},
    }
    reports = {}
    for name, columns in models.items():
        keep_top, curve = choose_keep_top(development, columns["rank"], min_recall)
        reports[name] = {
            "columns": columns,
            "development_discrimination": discrimination_metrics(development, columns["score"]),
            "holdout_discrimination": discrimination_metrics(holdout, columns["score"]),
            "recommended_keep_top": keep_top,
            "development_policy": rank_policy_metrics(development, columns["rank"], keep_top),
            "holdout_policy": rank_policy_metrics(holdout, columns["rank"], keep_top),
            "development_policy_curve": curve,
        }
    return reports


def apply_human_labels(
    labels: pd.DataFrame,
    human: pd.DataFrame,
    *,
    label_source: str = "human",
) -> tuple[pd.DataFrame, int]:
    """Overlay external labels by judgment ID or query-doc key.

    P1 annotation packs use ``annotation_id`` plus ``query_id``/``doc_id``;
    weak-review packs use ``judgment_id``. External labels may also add
    reviewed query-docs that were not selected by the weak-label queue. Set
    ``label_source`` explicitly so model-generated labels cannot enter
    human-only metrics.
    """
    if label_source not in {"human", "model_weak"}:
        raise ValueError(f"Unsupported external label source: {label_source}")
    if "label" not in human.columns:
        raise RuntimeError("Human labels need a label column")
    human = human.copy()
    human["label"] = human.label.fillna("").astype(str).str.strip().str.lower()
    invalid = set(human.label) - ALLOWED_LABELS - {""}
    if invalid:
        raise RuntimeError(f"Invalid human labels: {sorted(invalid)}")

    if "judgment_id" in human.columns:
        if human.judgment_id.duplicated().any():
            raise RuntimeError("Human judgment_id values must be unique")
        valid = human[human.label.isin(ALLOWED_LABELS)]
        overrides = valid.set_index("judgment_id").label
        unknown = set(overrides.index) - set(labels.judgment_id)
        if unknown:
            raise RuntimeError(f"Human labels contain unknown judgment IDs: {sorted(unknown)[:5]}")
        mask = labels.judgment_id.isin(overrides.index)
        labels.loc[mask, "label"] = labels.loc[mask, "judgment_id"].map(overrides)
        labels.loc[mask, "label_source"] = label_source
        return labels, int(mask.sum())

    if not {"query_id", "doc_id"}.issubset(human.columns):
        raise RuntimeError("Human labels need judgment_id or query_id,doc_id identifiers")
    if human.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Human query-doc labels must be unique")
    valid = human[human.label.isin(ALLOWED_LABELS)].copy()
    valid["query_id"] = valid.query_id.astype(int)
    valid["doc_id"] = valid.doc_id.astype(int)
    label_pairs = set(zip(labels.query_id.astype(int), labels.doc_id.astype(int)))
    human_pairs = set(zip(valid.query_id, valid.doc_id))
    existing_pairs = label_pairs.intersection(human_pairs)
    added_pairs = human_pairs.difference(label_pairs)

    human_map = valid.set_index(["query_id", "doc_id"]).label
    mask = [
        (int(query_id), int(doc_id)) in existing_pairs
        for query_id, doc_id in zip(labels.query_id, labels.doc_id, strict=True)
    ]
    mask_series = pd.Series(mask, index=labels.index, dtype=bool)
    pair_index = pd.MultiIndex.from_arrays([labels.query_id.astype(int), labels.doc_id.astype(int)])
    labels.loc[mask_series, "label"] = [human_map.loc[key] for key in pair_index[mask_series.to_numpy()]]
    labels.loc[mask_series, "label_source"] = label_source

    additions = []
    for row in valid.itertuples(index=False):
        key = (int(row.query_id), int(row.doc_id))
        if key not in added_pairs:
            continue
        annotation_id = getattr(row, "annotation_id", "")
        judgment_id = f"{label_source}:{annotation_id}" if annotation_id else f"{label_source}:{key[0]}:{key[1]}"
        additions.append({
            "judgment_id": judgment_id,
            "query_id": key[0],
            "doc_id": key[1],
            "label": row.label,
            "status": f"{label_source}_review",
            "evidence_valid": None,
            "weak_label": "",
            "label_source": label_source,
        })
    if additions:
        labels = pd.concat([labels, pd.DataFrame(additions)], ignore_index=True)
    return labels, len(valid)


def choose_model(reports: dict) -> str:
    return max(
        reports,
        key=lambda name: (
            np.nan_to_num(reports[name]["development_discrimination"]["strict_auc"], nan=-1.0),
            np.nan_to_num(reports[name]["development_discrimination"]["broad_auc"], nan=-1.0),
        ),
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    report_path = output / "calibration.json"
    labeled_path = output / "labeled_pairs.parquet"
    if any(path.exists() for path in (report_path, labeled_path)) and not args.force:
        raise FileExistsError("Calibration outputs exist; pass --force only after reviewing labels/provenance")

    docs = read_zip_parquet(args.reranker_outputs, "p1_doc_scores.parquet")
    required = {"judgment_id", "query_id", "doc_id", "label", "status", "evidence_valid"}
    if docs.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Duplicate reranker query-doc rows")
    if not args.weak_outputs and not args.human_labels and not args.model_labels:
        raise RuntimeError("Provide --human-labels, --model-labels, --weak-outputs, or a combination")

    if args.weak_outputs:
        weak = read_zip_parquet(args.weak_outputs, "p1_weak_judgments.parquet")
        if not required.issubset(weak.columns):
            raise RuntimeError(f"Weak judgments miss columns: {sorted(required - set(weak.columns))}")
        if weak.duplicated(["judgment_id"]).any() or weak.duplicated(["query_id", "doc_id"]).any():
            raise RuntimeError("Duplicate weak judgment or query-doc rows")
        labels = weak[list(required)].copy()
        labels["label"] = labels.label.astype(str).str.strip().str.lower()
        labels.loc[~labels.label.isin(ALLOWED_LABELS), "label"] = ""
        labels["weak_label"] = labels["label"]
        labels["label_source"] = "weak"
    else:
        labels = pd.DataFrame(columns=[*required, "weak_label", "label_source"])

    human_rows = 0
    model_label_rows = 0
    if args.model_labels:
        model_labels = pd.read_csv(args.model_labels, encoding="utf-8-sig")
        labels, model_label_rows = apply_human_labels(labels, model_labels, label_source="model_weak")
    if args.human_labels:
        human = pd.read_csv(args.human_labels, encoding="utf-8-sig")
        labels, human_rows = apply_human_labels(labels, human)

    labeled = labels[labels.label.isin(ALLOWED_LABELS)].merge(
        docs, on=["query_id", "doc_id"], validate="one_to_one", suffixes=("_judge", "")
    )
    if len(labeled) != int(labels.label.isin(ALLOWED_LABELS).sum()):
        raise RuntimeError("Some labeled rows are absent from reranker doc scores")
    queries = pd.read_parquet(args.queries)
    holdout_ids = allocate_stratified_holdout(queries, 40, args.seed)
    labeled["split"] = labeled.query_id.astype(int).map(lambda qid: "holdout" if qid in holdout_ids else "development")
    development = labeled[labeled.split == "development"]
    holdout = labeled[labeled.split == "holdout"]

    model_reports = build_model_reports(development, holdout, args.min_direct_recall)
    has_nonhuman_labels = bool(args.weak_outputs or args.model_labels)
    winner = choose_model(model_reports) if has_nonhuman_labels else None
    human_holdout_queries = int(labeled.loc[(labeled.label_source == "human") & (labeled.split == "holdout"), "query_id"].nunique())
    human_development = development[development.label_source == "human"]
    human_holdout = holdout[holdout.label_source == "human"]
    human_reports = build_model_reports(human_development, human_holdout, args.min_direct_recall)
    human_winner = choose_model(human_reports)
    human_winner_holdout = human_reports[human_winner]["holdout_policy"]
    human_dev_strict = human_development[human_development.label.isin(["direct", "irrelevant"])]
    human_holdout_strict = human_holdout[human_holdout.label.isin(["direct", "irrelevant"])]
    human_review_gate = {
        "minimum_human_rows_met": human_rows >= args.min_human_rows,
        "minimum_human_holdout_queries_met": human_holdout_queries >= args.min_human_holdout_queries,
        "human_development_class_coverage_met": int(human_dev_strict.label.eq("direct").sum()) >= 20 and int(human_dev_strict.label.eq("irrelevant").sum()) >= 20,
        "human_holdout_class_coverage_met": int(human_holdout_strict.label.eq("direct").sum()) >= 10 and int(human_holdout_strict.label.eq("irrelevant").sum()) >= 10,
        "human_holdout_direct_recall_met": human_winner_holdout["recall"] is not None and human_winner_holdout["recall"] >= 0.95,
        "human_holdout_irrelevant_rejection_met": human_winner_holdout["irrelevant_rejection"] is not None and human_winner_holdout["irrelevant_rejection"] >= 0.05,
    }
    human_review_gate["eligible_for_pool_trial"] = all(human_review_gate.values())

    output.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(labeled_path, index=False, compression="zstd")
    report = {
        "schema_version": "p1_reranker_calibration_v1",
        "evaluation_scope": "sampled_candidate_pool_only_not_corpus_qrels",
        "btc_qrels_available": False,
        "synthetic_dev_set_used": False,
        "metric_interpretation": (
            "Human-only metrics compare labels within the reviewed candidate pool. "
            "They cannot estimate corpus-wide recall or guarantee a higher BTC score."
        ),
        "reranker_outputs_sha256": sha256_file(args.reranker_outputs),
        "weak_outputs_sha256": sha256_file(args.weak_outputs) if args.weak_outputs else None,
        "model_labels_sha256": sha256_file(args.model_labels) if args.model_labels else None,
        "human_labels_sha256": sha256_file(args.human_labels) if args.human_labels else None,
        "warning": (
            "Weak/model-generated labels are not BTC ground truth; mixed/weak recommendation is exploratory until the human gate passes."
            if has_nonhuman_labels else
            "Human-only metrics cover a sampled candidate pool, not BTC qrels or corpus-wide recall."
        ),
        "weak_labels_available": bool(args.weak_outputs),
        "model_labels_available": bool(args.model_labels),
        "exploratory_nonhuman_labels_available": has_nonhuman_labels,
        "label_provenance": {
            "weak_judgments": "LLM-assisted weak labels; exploratory only" if args.weak_outputs else None,
            "model_labels": "external model-generated annotations; exploratory only" if args.model_labels else None,
            "human_labels": "human reviewer labels; sampled candidate pool only" if args.human_labels else None,
        },
        "label_source_counts": {str(key): int(value) for key, value in labeled.label_source.value_counts().items()},
        "rows": len(labeled), "queries": int(labeled.query_id.nunique()), "human_rows": human_rows,
        "model_label_rows": model_label_rows,
        "development_queries": int(development.query_id.nunique()), "holdout_queries": int(holdout.query_id.nunique()),
        "human_holdout_queries": human_holdout_queries,
        "exploratory_recommended_model": winner,
        "exploratory_recommended_keep_top": model_reports[winner]["recommended_keep_top"] if winner else None,
        "mixed_or_weak_models": model_reports if has_nonhuman_labels else None,
        "human_only_recommended_model": human_winner if human_rows else None,
        "human_only_recommended_keep_top": human_reports[human_winner]["recommended_keep_top"] if human_rows else None,
        "human_only_models": human_reports,
        "human_review_gate": human_review_gate,
    }
    report = json_finite(report)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "queries", "human_rows", "exploratory_recommended_model", "exploratory_recommended_keep_top", "human_only_recommended_model", "human_only_recommended_keep_top", "human_review_gate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
