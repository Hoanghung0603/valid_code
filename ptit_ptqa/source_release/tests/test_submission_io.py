from __future__ import annotations

import hashlib
import json

import pytest

from src.utils.submission_io import iter_json_array, sha256_file, write_json


def test_streaming_json_array_round_trip(tmp_path):
    path = tmp_path / "submission.json"
    payload = [
        {"id": 1, "query": "mẹ bầu"},
        {"id": 2, "query": "妊娠"},
    ]

    write_json(path, payload)

    assert list(iter_json_array(path, block_size=7)) == payload
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_streaming_json_array_rejects_non_array(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"id": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level JSON array"):
        list(iter_json_array(path))
