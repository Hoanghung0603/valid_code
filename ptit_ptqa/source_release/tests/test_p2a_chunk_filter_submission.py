from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_p2a_chunk_filter_submission import choose_kept_ordinals


def test_keeps_ceiling_fraction_by_score_and_stable_tie_break() -> None:
    scores = pd.DataFrame({
        "chunk_ordinal": [3, 0, 2, 1],
        "bge_chunk_score": [0.7, 0.9, 0.8, 0.9],
    })
    assert choose_kept_ordinals(scores, 0.5) == {0, 1}


def test_always_keeps_at_least_one_chunk() -> None:
    scores = pd.DataFrame({"chunk_ordinal": [0], "bge_chunk_score": [0.01]})
    assert choose_kept_ordinals(scores, 0.1) == {0}


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_rejects_invalid_keep_fraction(fraction: float) -> None:
    scores = pd.DataFrame({"chunk_ordinal": [0], "bge_chunk_score": [0.5]})
    with pytest.raises(ValueError):
        choose_kept_ordinals(scores, fraction)
