from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_p2a_chunk_pairs.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("prepare_p2a_chunk_pairs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_window_ranges_preserve_source_and_coverage() -> None:
    text = "A" * 4100
    spans = MODULE.window_ranges(len(text), 1800, 4)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert MODULE.coverage_fraction(spans, len(text)) == 1.0
    assert all(text[start:end] == "A" * (end - start) for start, end in spans)
    assert MODULE.coverage_fraction(MODULE.window_ranges(10000, 1800, 4), 10000) < 1.0
    assert MODULE.window_ranges(2000, 1800, 1) == [(0, 1800)]


def test_pilot_pack_exact_submitted_chunks(tmp_path: Path, monkeypatch) -> None:
    control = tmp_path / "control.zip"
    text = "Y tế cho phụ nữ mang thai. " * 90
    rows = [
        {"id": qid, "relevant_docs": [qid],
         "relevant_chunks": [{"doc_id": qid, "chunk_text": text}]}
        for qid in range(1, 121)
    ]
    with zipfile.ZipFile(control, "w") as archive:
        archive.writestr("submission.json", json.dumps(rows, ensure_ascii=False))
    queries = tmp_path / "queries.parquet"
    pilot = tmp_path / "pilot.parquet"
    pd.DataFrame({"id": range(1, 1201), "query": [f"Câu hỏi {i}" for i in range(1, 1201)]}).to_parquet(queries)
    pd.DataFrame({"id": range(1, 121)}).to_parquet(pilot)
    monkeypatch.setattr(MODULE, "CONTROL_SHA256", MODULE.sha256_file(control))
    output = tmp_path / "pack"
    monkeypatch.setattr(sys, "argv", ["prepare", "--scope", "pilot", "--control", str(control),
                                     "--queries", str(queries), "--pilot", str(pilot),
                                     "--output-dir", str(output)])
    MODULE.main()
    pairs = pd.read_parquet(output / "p2a_pairs.parquet")
    chunks = pd.read_parquet(output / "p2a_chunks.parquet")
    assert len(chunks) == 120
    assert chunks.window_coverage.eq(1.0).all()
    assert pairs.pair_id.is_unique
    assert set(pairs.query_id) == set(range(1, 121))
    assert all(text[int(row.window_start):int(row.window_end)] == row.passage_text
               for row in pairs.itertuples(index=False))
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {"p2a_pairs.parquet", "p2a_chunks.parquet", "p2a_manifest.json"}

    # A synthetic score payload tests lineage/aggregation checks only. It is
    # never a relevance label and cannot estimate the BTC score.
    import audit_p2a_chunk_scores as audit

    scored = pairs.drop(columns=["query_text", "passage_text"]).copy()
    scored["raw_score"] = scored.passage_rank.astype(float)
    scored["normalized_score"] = scored.passage_rank.astype(float) / 10
    scored["input_tokens"] = 10
    scored["model_revision"] = audit.EXPECTED_REVISION
    scored["gpu_shard"] = 0
    aggregated = scored.groupby(["query_id", "chunk_ordinal", "doc_id", "chunk_sha256"], as_index=False).agg(
        bge_chunk_score=("normalized_score", "max"), bge_chunk_raw=("raw_score", "max"),
        windows_scored=("pair_id", "size"), max_input_tokens=("input_tokens", "max"))
    chunk_scores = chunks.merge(aggregated, on=["query_id", "chunk_ordinal", "doc_id", "chunk_sha256"], validate="one_to_one")
    chunk_scores["bge_chunk_rank"] = 1
    pair_buffer, chunk_buffer = io.BytesIO(), io.BytesIO()
    scored.to_parquet(pair_buffer, index=False)
    chunk_scores.to_parquet(chunk_buffer, index=False)
    input_manifest = json.loads((output / "p2a_manifest.json").read_text(encoding="utf-8"))
    validation = {
        "schema_version": "p2a_chunk_scores_v1", "model_key": "bge", "scope": "pilot",
        "model_revision": audit.EXPECTED_REVISION, "pairs": len(pairs), "chunks": len(chunks),
        "queries": 120, "control_zip_sha256": input_manifest["control_zip_sha256"],
        "input_pairs_sha256": input_manifest["artifacts"]["p2a_pairs.parquet"]["sha256"],
        "input_chunks_sha256": input_manifest["artifacts"]["p2a_chunks.parquet"]["sha256"],
    }
    run_manifest = {"outputs": {
        "p2a_pair_scores.parquet": hashlib.sha256(pair_buffer.getvalue()).hexdigest(),
        "p2a_chunk_scores.parquet": hashlib.sha256(chunk_buffer.getvalue()).hexdigest(),
    }}
    result = tmp_path / "scores.zip"
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("p2a_pair_scores.parquet", pair_buffer.getvalue())
        archive.writestr("p2a_chunk_scores.parquet", chunk_buffer.getvalue())
        archive.writestr("validation.json", json.dumps(validation))
        archive.writestr("run_manifest.json", json.dumps(run_manifest))
    report = audit.validate(output, result, control)
    assert report["exact_source_windows"] == len(pairs)
    assert report["window_to_chunk_max_exact"] is True
