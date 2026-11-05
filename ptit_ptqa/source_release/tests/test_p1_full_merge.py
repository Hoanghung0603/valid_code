from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/merge_p1_full_reranker_parts.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("merge_p1_full_reranker_parts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def valid_docs() -> pd.DataFrame:
    return pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2, 2],
        "doc_id": [10, 11, 12, 20, 21, 22],
        "bge_score": [0.9, 0.5, 0.1, 0.8, 0.4, 0.2],
        "bge_rank": [1, 2, 3, 1, 2, 3],
    })


def test_validate_merged_docs_accepts_exact_rank_permutation() -> None:
    MODULE.validate_merged_docs(valid_docs(), "bge", expected_queries=2, docs_per_query=3)


def test_validate_merged_docs_rejects_duplicate_rank_even_with_same_bounds() -> None:
    docs = valid_docs()
    docs.loc[2, "bge_rank"] = 2
    with pytest.raises(RuntimeError, match="each rank"):
        MODULE.validate_merged_docs(docs, "bge", expected_queries=2, docs_per_query=3)


def test_validate_merged_docs_rejects_overlapping_query_doc() -> None:
    docs = valid_docs()
    docs.loc[2, "doc_id"] = 11
    with pytest.raises(RuntimeError, match="overlap"):
        MODULE.validate_merged_docs(docs, "bge", expected_queries=2, docs_per_query=3)
