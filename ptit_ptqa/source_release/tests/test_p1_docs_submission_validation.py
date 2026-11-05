from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_p1_docs_submission.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("validate_p1_docs_submission", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def make_zip(path: Path, rows: list[dict]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json.dumps(rows, separators=(",", ":")))


def fixture_rows() -> tuple[list[dict], list[dict]]:
    control = [{"id": qid, "relevant_docs": list(range(1, 217)),
                "relevant_chunks": [{"doc_id": 216, "text": "verbatim"}]}
               for qid in range(1, 1201)]
    candidate = [{"id": row["id"], "relevant_docs": list(range(1, 215)) + [216],
                  "relevant_chunks": row["relevant_chunks"]} for row in control]
    return control, candidate


def test_streaming_validator_accepts_docs_only_treatment(tmp_path: Path) -> None:
    control, candidate = fixture_rows()
    baseline_zip, candidate_zip = tmp_path / "baseline.zip", tmp_path / "candidate.zip"
    make_zip(baseline_zip, control)
    make_zip(candidate_zip, candidate)
    report = MODULE.validate(baseline_zip, candidate_zip)
    assert report["queries"] == 1200
    assert report["removed_query_docs"] == 1200
    assert report["chunks_exactly_equal_control"]


def test_streaming_validator_rejects_removed_chunk_owner(tmp_path: Path) -> None:
    control, candidate = fixture_rows()
    candidate[0]["relevant_docs"].remove(216)
    baseline_zip, candidate_zip = tmp_path / "baseline.zip", tmp_path / "candidate.zip"
    make_zip(baseline_zip, control)
    make_zip(candidate_zip, candidate)
    with pytest.raises(ValueError, match="Chunk-owning docs removed"):
        MODULE.validate(baseline_zip, candidate_zip)
