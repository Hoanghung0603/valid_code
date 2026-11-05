from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_p1_verified_docs_submission.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_p1_verified_docs_submission", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_retained_docs_unions_top_ranked_with_internal_owners_in_control_order() -> None:
    docs = [10, 20, 30, 40, 50]
    ranked = [50, 40, 30, 20, 10]
    assert MODULE.retained_docs(docs, {10, 999}, ranked, 2) == [10, 40, 50]


def test_retained_docs_rejects_score_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="coverage"):
        MODULE.retained_docs([1, 2, 3], {1}, [1, 2, 4], 2)


def test_retained_docs_rejects_noop_keep_top() -> None:
    with pytest.raises(ValueError, match="keep_top"):
        MODULE.retained_docs([1, 2, 3], {1}, [1, 2, 3], 3)


def test_load_scores_normalizes_full_qwen4b_column_name(tmp_path: Path) -> None:
    import pandas as pd

    source = tmp_path / "full_qwen_scores.parquet"
    pd.DataFrame({"query_id": [1], "doc_id": [2], "qwen4b_score": [0.75]}).to_parquet(source)
    scores, payload_hash = MODULE.load_scores(source)
    assert "qwen_score" in scores.columns
    assert scores.loc[0, "qwen_score"] == 0.75
    assert len(payload_hash) == 64


def test_calibration_policy_uses_human_reviewed_candidate_pool_fields() -> None:
    calibration = {
        "human_review_gate": {"eligible_for_pool_trial": True},
        "human_only_recommended_model": "bge",
        "human_only_recommended_keep_top": 140,
        "exploratory_recommended_model": "qwen4b",
        "exploratory_recommended_keep_top": 120,
    }
    assert MODULE.resolve_calibration_policy(calibration, False) == ("human_reviewed_candidate_pool_trial", "bge", 140)


def test_calibration_policy_blocks_weak_only_without_explicit_override() -> None:
    calibration = {
        "human_review_gate": {"eligible_for_pool_trial": False},
        "exploratory_recommended_model": "qwen4b",
        "exploratory_recommended_keep_top": 160,
    }
    with pytest.raises(RuntimeError, match="operational human-review gate"):
        MODULE.resolve_calibration_policy(calibration, False)
    assert MODULE.resolve_calibration_policy(calibration, True) == ("unreviewed_exploratory", "qwen4b", 160)


def test_calibration_policy_rejects_noop_top216() -> None:
    calibration = {
        "human_review_gate": {"eligible_for_pool_trial": False},
        "exploratory_recommended_model": "bge",
        "exploratory_recommended_keep_top": 216,
    }
    with pytest.raises(RuntimeError, match="no-op"):
        MODULE.resolve_calibration_policy(calibration, True)


def test_uncalibrated_exploratory_override_requires_explicit_provenance() -> None:
    calibration = {
        "human_review_gate": {"eligible_for_pool_trial": False},
        "exploratory_recommended_model": "qwen4b",
        "exploratory_recommended_keep_top": 195,
    }
    with pytest.raises(RuntimeError, match="requires"):
        MODULE.select_requested_policy(calibration, "bge", 180, False, True, "preflight")
    with pytest.raises(RuntimeError, match="requires"):
        MODULE.select_requested_policy(calibration, "bge", 180, True, True, " ")
    assert MODULE.select_requested_policy(
        calibration, "bge", 180, True, True, "BTC exploratory docs-only test"
    ) == "uncalibrated_exploratory"


def test_override_cannot_impersonate_human_reviewed_policy() -> None:
    calibration = {"human_review_gate": {"eligible_for_pool_trial": True}}
    with pytest.raises(RuntimeError, match="human-reviewed"):
        MODULE.select_requested_policy(calibration, "bge", 180, True, True, "test")


def test_package_lzma64_has_one_root_submission_member(tmp_path: Path) -> None:
    if not shutil.which("7z"):
        pytest.skip("7z not installed")
    source = tmp_path / "candidate.json"
    source.write_text('[{"id":1}]', encoding="utf-8")
    staged = MODULE.package_lzma64(source, tmp_path / "candidate.zip")
    with zipfile.ZipFile(staged) as archive:
        assert archive.namelist() == ["submission.json"]
        assert archive.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert archive.read("submission.json") == source.read_bytes()
        assert archive.testzip() is None
