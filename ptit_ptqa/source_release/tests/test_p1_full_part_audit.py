from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_p1_full_reranker_part.py"
SPEC = importlib.util.spec_from_file_location("audit_p1_full_reranker_part", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = np.arange(1, 217)
    raw = (217 - ids) / 100.0
    score = 1.0 / (1.0 + np.exp(-raw))
    expected = pd.DataFrame({
        "pair_id": [f"p{i}" for i in ids],
        "query_id": 1,
        "doc_id": ids,
        "doc_rank": ids,
        "owner_doc": False,
        "passage_rank": 0,
        "language": "vi",
        "dense_score": 0.1,
    })
    scored = expected.assign(
        bge_raw=raw, bge_score=score, input_tokens=16,
        model_revision=MODULE.MODEL_REVISIONS["bge"][1],
    )
    docs = expected[["query_id", "doc_id", "doc_rank", "owner_doc", "language"]].assign(
        bge_raw=raw, bge_score=score, passage_count=1, bge_rank=ids,
    )
    return expected, scored, docs


def test_full_part_audit_checks_lineage_and_aggregation() -> None:
    expected, scored, docs = sample_frames()
    report = MODULE.audit_frames(expected, scored, docs, "bge")
    assert report["pairs"] == 216
    assert report["query_docs"] == 216
    assert report["lineage_exact"] and report["doc_aggregation_exact"]


def test_full_part_audit_rejects_pair_source_mismatch() -> None:
    expected, scored, docs = sample_frames()
    scored.loc[0, "doc_id"] = 999
    with pytest.raises(RuntimeError, match="lineage"):
        MODULE.audit_frames(expected, scored, docs, "bge")


def test_full_part_audit_rejects_wrong_doc_maximum() -> None:
    expected, scored, docs = sample_frames()
    docs.loc[0, "bge_score"] -= 0.1
    with pytest.raises(RuntimeError, match="maximum passage"):
        MODULE.audit_frames(expected, scored, docs, "bge")


def test_partition_selection_is_query_disjoint() -> None:
    expected, _, _ = sample_frames()
    two = expected.copy()
    two["query_id"] = 2
    combined = pd.concat([expected, two], ignore_index=True)
    assert set(MODULE.expected_partition(combined, 0, 2).query_id) == {1}
    assert set(MODULE.expected_partition(combined, 1, 2).query_id) == {2}
