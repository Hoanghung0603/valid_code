from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_p1_compact_human_review.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_p1_compact_human_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    key_rows, blind_rows = [], []
    for qid in range(1, 121):
        for number, reason in enumerate(("qwen_rank_1", "qwen_rank_180"), 1):
            annotation_id = f"P1-{qid:04d}-{number}"
            key_rows.append({"annotation_id": annotation_id, "query_id": qid,
                             "split": "development" if qid <= 80 else "holdout",
                             "sampling_reasons": reason, "bge_score": 0.9})
            blind_rows.append({"annotation_id": annotation_id, "query_id": qid,
                               "query": "question", "doc_id": qid * 10 + number,
                               "passage_1": "source text", "passage_2": None,
                               "label": None, "notes": None})
    return pd.DataFrame(key_rows), pd.DataFrame(blind_rows)


def test_compact_selection_covers_each_query_and_remains_blind() -> None:
    key, blind = fixture_frames()
    selected = MODULE.select_ids(key)
    review = MODULE.build_review(blind, selected)
    assert len(review) == 240
    assert review.groupby("query_id").size().eq(2).all()
    assert set(review.columns) == set(blind.columns)
    assert not set(review.columns) & {"split", "bge_score", "qwen_score", "sampling_reasons"}


def test_compact_review_rejects_prelabelled_rows() -> None:
    key, blind = fixture_frames()
    blind.loc[0, "label"] = "direct"
    with pytest.raises(ValueError, match="already contains"):
        MODULE.build_review(blind, MODULE.select_ids(key))
