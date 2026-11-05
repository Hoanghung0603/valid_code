from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_p1_docs_policy_impact.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("audit_p1_docs_policy_impact", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def fixture_data() -> tuple[list[dict], pd.DataFrame]:
    control = [{"id": 1, "relevant_docs": list(range(1, 217)),
                "relevant_chunks": [{"doc_id": 216}]}]
    scores = pd.DataFrame({"query_id": [1] * 216, "doc_id": list(range(1, 217)),
                           "bge_score": [float(217 - doc_id) for doc_id in range(1, 217)]})
    return control, scores


def test_cutoff_impact_preserves_chunk_owner() -> None:
    control, scores = fixture_data()
    report = MODULE.audit_impact(control, scores, "bge", [1, 215])
    assert report["control_query_docs"] == 216
    assert report["cutoffs"]["1"]["remaining_query_docs"] == 2
    assert report["cutoffs"]["1"]["owner_docs_beyond_cutoff_retained"] == 1
    assert report["cutoffs"]["215"]["remaining_query_docs"] == 216
    assert report["cutoffs"]["215"]["changed_queries"] == 0


def test_score_coverage_mismatch_is_rejected() -> None:
    control, scores = fixture_data()
    with pytest.raises(ValueError, match="coverage"):
        MODULE.audit_impact(control, scores.iloc[:-1], "bge", [195])


def test_duplicate_cutoffs_are_rejected() -> None:
    control, scores = fixture_data()
    with pytest.raises(ValueError, match="unique"):
        MODULE.audit_impact(control, scores, "bge", [195, 195])
