import json
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.release.config import load_release_config
from src.release.io import iter_submission_rows, sha256_file
from src.release.replay import build_replay_cache, replay_submission, verify_release
from src.release.service import ReplayService


def make_release(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    queries = tmp_path / "data/query.parquet"
    corpus = tmp_path / "data/corpus.parquet"
    pd.DataFrame({"id": [1], "query": ["Tôi bị đau đầu?"]}).to_parquet(queries)
    pd.DataFrame({"id": [7], "url": ["https://example.org/7"]}).to_parquet(corpus)
    payload = [{"id": 1, "relevant_docs": [7],
                "relevant_chunks": [{"doc_id": 7, "chunk_text": "đoạn bằng chứng"}]}]
    submission = tmp_path / "submission.zip"
    with zipfile.ZipFile(submission, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json.dumps(payload, ensure_ascii=False))
    import hashlib
    digest = hashlib.sha256()
    for row in payload:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode() + b"\n")
    config = {
        "schema_version": "r2ai_release_v1", "release_id": "test", "models": [],
        "paths": {"queries": "data/query.parquet", "corpus": "data/corpus.parquet",
                  "submission": "submission.zip", "cache": "data/replay.sqlite",
                  "source_text_dir": "data/text"},
        "sha256": {"queries": sha256_file(queries), "corpus": sha256_file(corpus),
                   "submission": sha256_file(submission), "predictions": digest.hexdigest()},
        "counts": {"queries": 1, "predicted_docs": 1, "predicted_chunks": 1},
        "limits": {"api_page_size": 100, "evidence_context_chars": 4096},
    }
    path = tmp_path / "configs/release.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_verify_replay_cache_and_service(tmp_path):
    config = load_release_config(make_release(tmp_path))
    assert verify_release(config)["predicted_chunks"] == 1
    output = tmp_path / "copy.zip"
    assert replay_submission(config, output)["sha256"] == config.expected_hash("submission")
    assert build_replay_cache(config)["chunks"] == 1
    service = ReplayService(config)
    run = service.create_run(mode="replay", query_id=1, query="Tôi bị đau đầu?")
    assert service.get_results(run.run_id, "docs", 0, 20)["items"][0]["doc_id"] == 7
    assert service.get_results(run.run_id, "chunks", 0, 20)["items"][0]["chunk_text"] == "đoạn bằng chứng"


def test_release_fails_closed_on_changed_query(tmp_path):
    config = load_release_config(make_release(tmp_path))
    frame = pd.read_parquet(config.artifact("queries"))
    frame.loc[0, "query"] = "changed"
    frame.to_parquet(config.artifact("queries"))
    with pytest.raises(ValueError, match="SHA256"):
        verify_release(config)


def test_rejects_non_submission_zip(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("other.json", "[]")
    with pytest.raises(ValueError, match="only"):
        list(iter_submission_rows(path))
