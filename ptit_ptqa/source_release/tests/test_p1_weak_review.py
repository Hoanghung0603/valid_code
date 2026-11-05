from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_p1_weak_review.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_p1_weak_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_select_review_rows_has_core_and_complementary_coverage() -> None:
    rows = []
    for query_id in (1, 2):
        for number, reason in enumerate(("hard", "random", "rank_anchor_64", "rank_anchor_128")):
            rows.append({
                "judgment_id": f"{query_id}-{number}",
                "query_id": query_id,
                "status": "needs_review" if number == 0 else "ok",
                "selection_reason": reason,
                "rank_disagreement": number * 10,
            })
    selected = MODULE.select_review_rows(pd.DataFrame(rows), "seed")
    assert len(selected) == 6
    assert selected.groupby("query_id").size().to_dict() == {1: 3, 2: 3}
    assert selected.groupby("query_id").review_tier.apply(lambda values: list(values).count("core")).eq(1).all()
    assert not selected.judgment_id.duplicated().any()


def test_hidden_set_covers_model_and_weak_judge_fields() -> None:
    assert {"label", "status", "split", "owner_doc", "bge_score", "qwen_score", "raw_output"}.issubset(MODULE.HIDDEN)
