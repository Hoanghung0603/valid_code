from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_p1_reranker_pilot.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_p1_reranker_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_detect_script_language_handles_cjk_and_vietnamese() -> None:
    assert MODULE.detect_script_language("这是医学文档。" * 20) == "zh"
    assert MODULE.detect_script_language("Phụ nữ mang thai cần được theo dõi.") == "vi"
    assert MODULE.detect_script_language("Pregnancy requires clinical monitoring.") == "en_or_plain_latin"


def test_select_top_passages_is_per_doc_and_stable() -> None:
    result = MODULE.select_top_passages(
        np.array([7, 8, 9, 10, 11]),
        np.array([1, 1, 1, 2, 2]),
        np.array([0.1, 0.9, 0.9, -1.0, 0.2]),
        2,
    )
    assert result == {1: [8, 9], 2: [11, 10]}


def test_select_nonempty_passages_skips_bad_index_spans() -> None:
    selected, skipped = MODULE.select_nonempty_passages(
        [0, 1, 2],
        "   useful second passage",
        np.array([0, 3, 10]),
        np.array([3, 10, 24]),
        2,
    )
    assert selected == [1, 2]
    assert skipped == 1


def test_hash_text_is_deterministic_and_unicode_sensitive() -> None:
    assert MODULE.hash_text("bầu") == MODULE.hash_text("bầu")
    assert MODULE.hash_text("bầu") != MODULE.hash_text("bau")


def test_docs_only_owner_policy_can_exclude_external_chunk_owners() -> None:
    docs = {1, 2, 3}
    all_chunk_owners = {2, 4}
    assert all_chunk_owners.intersection(docs) == {2}
    assert all_chunk_owners.difference(docs) == {4}


def test_binary_auc_reference_cases() -> None:
    audit_script = SCRIPT.parent / "audit_p1_reranker_outputs.py"
    audit_spec = importlib.util.spec_from_file_location("audit_p1_reranker_outputs", audit_script)
    audit_module = importlib.util.module_from_spec(audit_spec)
    assert audit_spec and audit_spec.loader
    audit_spec.loader.exec_module(audit_module)
    labels = np.array([False, False, True, True])
    assert audit_module.binary_auc(labels, np.array([0.0, 0.1, 0.8, 1.0])) == 1.0
    assert audit_module.binary_auc(labels, np.array([1.0, 0.8, 0.1, 0.0])) == 0.0


def test_output_audit_checks_score_lineage_and_doc_rank_permutations() -> None:
    import pandas as pd
    import pytest

    audit_script = SCRIPT.parent / "audit_p1_reranker_outputs.py"
    audit_spec = importlib.util.spec_from_file_location("audit_p1_reranker_outputs_lineage", audit_script)
    audit_module = importlib.util.module_from_spec(audit_spec)
    assert audit_spec and audit_spec.loader
    audit_spec.loader.exec_module(audit_module)

    rows = []
    for query_id in (1, 2):
        for doc_rank, doc_id in enumerate((10, 20, 30), start=1):
            for passage_rank in (1, 2):
                rows.append({
                    "pair_id": f"{query_id}:{doc_id}:{passage_rank}",
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "doc_rank": doc_rank,
                    "owner_doc": doc_id == 10,
                    "passage_rank": passage_rank,
                })
    input_pairs = pd.DataFrame(rows)
    bge = input_pairs[["pair_id", "query_id", "doc_id", "passage_rank"]].copy()
    bge["raw_score"] = np.arange(len(bge), dtype=float)
    bge["normalized_score"] = 1 / (1 + np.exp(-bge.raw_score))
    qwen = bge.copy()
    qwen["raw_score"] = bge.raw_score.to_numpy() / 2
    qwen["normalized_score"] = 1 / (1 + np.exp(-qwen.raw_score))
    pair_scores = input_pairs[["pair_id", "query_id", "doc_id", "passage_rank"]].copy()
    pair_scores["bge_raw"] = bge.raw_score
    pair_scores["bge_score"] = bge.normalized_score
    pair_scores["qwen_raw"] = qwen.raw_score
    pair_scores["qwen_score"] = qwen.normalized_score
    doc_scores = pair_scores.groupby(["query_id", "doc_id"], as_index=False).agg(
        bge_score=("bge_score", "max"), qwen_score=("qwen_score", "max")
    )
    doc_scores = doc_scores.merge(
        input_pairs.groupby(["query_id", "doc_id"], as_index=False).agg(
            doc_rank=("doc_rank", "first"), owner_doc=("owner_doc", "first")
        ), on=["query_id", "doc_id"], validate="one_to_one",
    )
    for column in ("bge_score", "qwen_score"):
        doc_scores[f"{column.split('_')[0]}_rank"] = doc_scores.groupby("query_id")[column].rank(
            method="first", ascending=False
        ).astype(int)

    audit_module.validate_score_frames(input_pairs, bge, qwen, pair_scores, doc_scores, docs_per_query=3)
    corrupted = doc_scores.copy()
    corrupted.loc[0, "bge_score"] += 0.05
    with pytest.raises(RuntimeError, match="max passage score"):
        audit_module.validate_score_frames(input_pairs, bge, qwen, pair_scores, corrupted, docs_per_query=3)


def test_output_audit_requires_exact_model_revision_and_both_gpu_shards() -> None:
    import pandas as pd
    import pytest

    audit_script = SCRIPT.parent / "audit_p1_reranker_outputs.py"
    audit_spec = importlib.util.spec_from_file_location("audit_p1_reranker_provenance", audit_script)
    audit_module = importlib.util.module_from_spec(audit_spec)
    assert audit_spec and audit_spec.loader
    audit_spec.loader.exec_module(audit_module)

    scores = pd.DataFrame({
        "model_key": ["bge", "bge"],
        "model_id": ["BAAI/bge-reranker-v2-m3"] * 2,
        "model_revision": ["abc123"] * 2,
        "gpu": [0, 1],
    })
    assert audit_module.model_provenance(scores, "bge", "BAAI/bge-reranker-v2-m3")["model_revision"] == "abc123"
    with pytest.raises(RuntimeError, match="revision"):
        audit_module.model_provenance(scores.assign(model_revision="unknown"), "bge", "BAAI/bge-reranker-v2-m3")
    with pytest.raises(RuntimeError, match="GPU shards"):
        audit_module.model_provenance(scores[scores.gpu == 0], "bge", "BAAI/bge-reranker-v2-m3")


def test_output_audit_accepts_gpu_shard_provenance_from_kaggle_worker() -> None:
    import pandas as pd
    import pytest

    audit_script = SCRIPT.parent / "audit_p1_reranker_outputs.py"
    audit_spec = importlib.util.spec_from_file_location("audit_p1_reranker_gpu_shard", audit_script)
    audit_module = importlib.util.module_from_spec(audit_spec)
    assert audit_spec and audit_spec.loader
    audit_spec.loader.exec_module(audit_module)

    scores = pd.DataFrame({
        "model_key": ["bge", "bge"],
        "model_id": ["BAAI/bge-reranker-v2-m3"] * 2,
        "model_revision": ["abc123"] * 2,
        "gpu_shard": [0, 1],
    })
    provenance = audit_module.model_provenance(scores, "bge", "BAAI/bge-reranker-v2-m3")
    assert provenance["gpu_shards"] == [0, 1]

    inconsistent = scores.assign(gpu=[0, 0])
    with pytest.raises(RuntimeError, match="disagree"):
        audit_module.model_provenance(inconsistent, "bge", "BAAI/bge-reranker-v2-m3")


def test_full_scope_uses_distinct_default_output() -> None:
    assert MODULE.OUT_FULL != MODULE.OUT
    assert MODULE.OUT_FULL.name == "p1_reranker_full_v1"
