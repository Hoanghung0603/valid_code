from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/calibrate_p1_reranker.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("calibrate_p1_reranker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_discrimination_reference_metrics() -> None:
    labels = np.array([False, False, True, True])
    scores = np.array([0.0, 0.1, 0.8, 1.0])
    assert MODULE.binary_auc(labels, scores) == 1.0
    assert MODULE.average_precision(labels, scores) == 1.0


def test_undefined_metrics_serialize_as_strict_json_null() -> None:
    report = {"auc": float("nan"), "nested": [{"ap": float("inf")}]}
    serialized = json.dumps(MODULE.json_finite(report), allow_nan=False)
    assert json.loads(serialized) == {"auc": None, "nested": [{"ap": None}]}


def test_rank_policy_protects_owner_even_below_cutoff() -> None:
    frame = pd.DataFrame({
        "label": ["direct", "direct", "irrelevant", "irrelevant"],
        "rank": [1, 200, 2, 210],
        "owner_doc": [False, True, False, False],
    })
    metrics = MODULE.rank_policy_metrics(frame, "rank", 40)
    assert metrics["recall"] == 1.0
    assert metrics["irrelevant_rejection"] == 0.5
    assert metrics["precision"] == 2 / 3


def test_choose_keep_top_obeys_recall_floor() -> None:
    frame = pd.DataFrame({
        "label": ["direct", "direct", "irrelevant", "irrelevant"],
        "rank": [10, 100, 20, 180],
        "owner_doc": [False, False, False, False],
    })
    keep_top, curve = MODULE.choose_keep_top(frame, "rank", 1.0)
    assert keep_top >= 100
    chosen = next(row for row in curve if row["keep_top"] == keep_top)
    assert chosen["recall"] == 1.0


def test_empty_human_split_stays_non_promotable_without_crashing() -> None:
    empty = pd.DataFrame(columns=[
        "label", "owner_doc", "bge_score", "qwen_score", "bge_rank", "qwen_rank"
    ])
    reports = MODULE.build_model_reports(empty, empty, 0.97)
    assert MODULE.choose_model(reports) in {"bge", "qwen4b"}
    assert reports["bge"]["holdout_policy"]["recall"] is None
    assert reports["qwen4b"]["recommended_keep_top"] == 216


def test_human_annotation_csv_overlays_by_query_doc_and_adds_reviewed_rows() -> None:
    labels = pd.DataFrame({
        "judgment_id": ["WJ-00001", "WJ-00002"],
        "query_id": [1, 2],
        "doc_id": [10, 20],
        "label": ["related", "irrelevant"],
        "status": ["ok", "ok"],
        "evidence_valid": [True, False],
        "weak_label": ["related", "irrelevant"],
        "label_source": ["weak", "weak"],
    })
    human = pd.DataFrame({
        "annotation_id": ["P1-0001", "P1-0002"],
        "query_id": [1, 3],
        "doc_id": [10, 30],
        "label": ["direct", "irrelevant"],
    })

    merged, human_rows = MODULE.apply_human_labels(labels, human)
    assert human_rows == 2
    assert merged.loc[merged.judgment_id == "WJ-00001", "label"].item() == "direct"
    assert merged.loc[merged.judgment_id == "WJ-00001", "label_source"].item() == "human"
    assert merged.loc[merged.judgment_id == "WJ-00002", "label_source"].item() == "weak"
    added = merged.loc[merged.query_id == 3].iloc[0]
    assert added.judgment_id == "human:P1-0002"
    assert added.label_source == "human"


def test_human_weak_review_labels_still_map_by_judgment_id() -> None:
    labels = pd.DataFrame({
        "judgment_id": ["WJ-00001"], "query_id": [1], "doc_id": [10],
        "label": ["related"], "label_source": ["weak"],
    })
    human = pd.DataFrame({"judgment_id": ["WJ-00001"], "label": ["direct"]})
    merged, count = MODULE.apply_human_labels(labels, human)
    assert count == 1
    assert merged.label.iloc[0] == "direct"
    assert merged.label_source.iloc[0] == "human"


def test_model_generated_labels_keep_exploratory_provenance() -> None:
    labels = pd.DataFrame(columns=[
        "judgment_id", "query_id", "doc_id", "label", "status",
        "evidence_valid", "weak_label", "label_source",
    ])
    model_labels = pd.DataFrame({
        "annotation_id": ["P1-0001"], "query_id": [1], "doc_id": [10], "label": ["direct"],
    })

    merged, count = MODULE.apply_human_labels(labels, model_labels, label_source="model_weak")
    assert count == 1
    assert merged.label_source.iloc[0] == "model_weak"
    assert merged.status.iloc[0] == "model_weak_review"
    assert merged.judgment_id.iloc[0] == "model_weak:P1-0001"


def test_calibration_cli_accepts_blinded_p1_annotation_csv(tmp_path: Path, monkeypatch) -> None:
    docs = []
    weak = []
    human = []
    queries = []
    for query_id in range(1, 121):
        queries.append({"query_id": query_id, "stratum": "all"})
        for rank in range(1, 217):
            doc_id = query_id * 1000 + rank
            docs.append({
                "query_id": query_id, "doc_id": doc_id, "owner_doc": False,
                "bge_score": 1 / rank, "qwen_score": 1 / rank,
                "bge_rank": rank, "qwen_rank": rank,
            })
        weak.append({
            "judgment_id": f"WJ-{query_id:05d}", "query_id": query_id,
            "doc_id": query_id * 1000 + 1, "label": "related",
            "status": "ok", "evidence_valid": True,
        })
        human.append({
            "annotation_id": f"P1-{query_id:04d}", "query_id": query_id,
            "doc_id": query_id * 1000 + 2,
            "label": "direct" if query_id % 2 else "irrelevant",
        })
    docs_frame = pd.DataFrame(docs)
    weak_frame = pd.DataFrame(weak)
    queries_path = tmp_path / "queries.parquet"
    pd.DataFrame(queries).to_parquet(queries_path, index=False)
    human_path = tmp_path / "human.csv"
    pd.DataFrame(human).to_csv(human_path, index=False)
    docs_zip = tmp_path / "reranker.zip"
    weak_zip = tmp_path / "weak.zip"
    for path, member, frame in (
        (docs_zip, "p1_doc_scores.parquet", docs_frame),
        (weak_zip, "p1_weak_judgments.parquet", weak_frame),
    ):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member, buffer.getvalue())

    output = tmp_path / "calibration"
    monkeypatch.setattr(sys, "argv", [
        "calibrate_p1_reranker.py",
        "--reranker-outputs", str(docs_zip),
        "--weak-outputs", str(weak_zip),
        "--human-labels", str(human_path),
        "--queries", str(queries_path),
        "--output-dir", str(output),
    ])
    MODULE.main()

    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    labeled = pd.read_parquet(output / "labeled_pairs.parquet")
    assert report["human_rows"] == 120
    assert report["evaluation_scope"] == "sampled_candidate_pool_only_not_corpus_qrels"
    assert report["synthetic_dev_set_used"] is False
    assert len(labeled) == 240
    assert int(labeled.label_source.eq("human").sum()) == 120
    assert set(labeled.loc[labeled.label_source == "human", "doc_id"] % 1000) == {2}


def test_calibration_cli_supports_human_only_without_weak_labels(tmp_path: Path, monkeypatch) -> None:
    docs = []
    human = []
    queries = []
    for query_id in range(1, 121):
        queries.append({"query_id": query_id, "stratum": "all"})
        for rank in range(1, 217):
            docs.append({
                "query_id": query_id, "doc_id": query_id * 1000 + rank,
                "owner_doc": False, "bge_score": 1 / rank, "qwen_score": 1 / rank,
                "bge_rank": rank, "qwen_rank": rank,
            })
        human.append({
            "annotation_id": f"P1-{query_id:04d}", "query_id": query_id,
            "doc_id": query_id * 1000 + 2,
            "label": "direct" if query_id % 2 else "irrelevant",
        })

    docs_zip = tmp_path / "reranker.zip"
    buffer = io.BytesIO()
    pd.DataFrame(docs).to_parquet(buffer, index=False)
    with zipfile.ZipFile(docs_zip, "w") as archive:
        archive.writestr("p1_doc_scores.parquet", buffer.getvalue())
    queries_path = tmp_path / "queries.parquet"
    pd.DataFrame(queries).to_parquet(queries_path, index=False)
    human_path = tmp_path / "human.csv"
    pd.DataFrame(human).to_csv(human_path, index=False)
    output = tmp_path / "human_only_calibration"

    monkeypatch.setattr(sys, "argv", [
        "calibrate_p1_reranker.py",
        "--reranker-outputs", str(docs_zip),
        "--human-labels", str(human_path),
        "--queries", str(queries_path),
        "--output-dir", str(output),
    ])
    MODULE.main()

    report = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    labeled = pd.read_parquet(output / "labeled_pairs.parquet")
    assert report["weak_labels_available"] is False
    assert report["weak_outputs_sha256"] is None
    assert report["exploratory_recommended_model"] is None
    assert report["human_rows"] == 120
    assert report["btc_qrels_available"] is False
    assert report["synthetic_dev_set_used"] is False
    assert len(labeled) == 120
    assert set(labeled.label_source) == {"human"}
