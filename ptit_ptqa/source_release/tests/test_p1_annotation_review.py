from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_p1_annotation_review.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_p1_annotation_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_allocate_stratified_holdout_is_exact_and_stable() -> None:
    queries = pd.DataFrame({
        "query_id": range(120),
        "stratum": ["a"] * 60 + ["b"] * 30 + ["c"] * 20 + ["rare"] * 10,
    })
    first = MODULE.allocate_stratified_holdout(queries, 40, "seed")
    second = MODULE.allocate_stratified_holdout(queries, 40, "seed")
    assert first == second
    assert len(first) == 40
    selected = queries[queries.query_id.isin(first)].stratum.value_counts().to_dict()
    assert selected == {"a": 20, "b": 10, "c": 7, "rare": 3}


def test_stable_hash_changes_with_seed() -> None:
    assert MODULE.stable_hash("a", 1) == MODULE.stable_hash("a", 1)
    assert MODULE.stable_hash("a", 1) != MODULE.stable_hash("b", 1)


def test_annotation_candidates_cover_both_rankers_and_cutoff_regions() -> None:
    docs = pd.DataFrame({
        "query_id": [7] * 216,
        "doc_id": range(1000, 1216),
        "qwen_rank": range(1, 217),
        "bge_rank": range(216, 0, -1),
        "rank_disagreement": [abs(rank - (217 - rank)) for rank in range(1, 217)],
        "owner_doc": [doc_id in {1002, 1100} for doc_id in range(1000, 1216)],
    })
    queries = pd.DataFrame({"query_id": [7], "query": ["question"]})

    selected = MODULE.select_annotation_candidates(docs, queries)
    assert set(selected.query_id) == {7}
    assert set(selected.loc[selected.sampling_reasons.str.contains("qwen_rank_"), "qwen_rank"]) >= {1, 40, 100, 180}
    assert selected.doc_id.nunique() == len(selected)
    assert set(selected.columns).issuperset({"query", "sampling_reasons", "bge_rank", "qwen_rank"})


def test_review_guide_defines_labels_and_disclaims_btc_qrels() -> None:
    assert "`direct`" in MODULE.REVIEW_GUIDE
    assert "`related`" in MODULE.REVIEW_GUIDE
    assert "`irrelevant`" in MODULE.REVIEW_GUIDE
    assert "không phải qrels/gold labels của BTC" in MODULE.REVIEW_GUIDE
    assert "để `label` trống" in MODULE.REVIEW_GUIDE
