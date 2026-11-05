from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ReleaseConfig
from .io import iter_submission_rows, sha256_file


def _canonical_digest(rows: Iterable[dict]) -> tuple[str, int, int, int]:
    digest = hashlib.sha256()
    query_count = docs = chunks = 0
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8") + b"\n")
        query_count += 1
        docs += len(row.get("relevant_docs", []))
        chunks += len(row.get("relevant_chunks", []))
    return digest.hexdigest(), query_count, docs, chunks


def verify_release(config: ReleaseConfig, *, verify_predictions: bool = True) -> dict:
    checked: dict[str, object] = {"release_id": config.release_id, "status": "pass"}
    for key in ("queries", "corpus", "submission"):
        path = config.artifact(key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing release artifact: {path}")
        actual = sha256_file(path)
        if actual != config.expected_hash(key):
            raise ValueError(f"SHA256 mismatch for {key}: {actual}")
        checked[f"{key}_sha256"] = actual
    queries = pd.read_parquet(config.artifact("queries"), columns=["id", "query"])
    if len(queries) != config.expected_count("queries") or queries.id.duplicated().any():
        raise ValueError("Official query coverage is invalid")
    checked["queries"] = len(queries)
    if verify_predictions:
        digest, count, docs, chunks = _canonical_digest(
            iter_submission_rows(config.artifact("submission"))
        )
        expected = (
            config.expected_hash("predictions"),
            config.expected_count("queries"),
            config.expected_count("predicted_docs"),
            config.expected_count("predicted_chunks"),
        )
        if (digest, count, docs, chunks) != expected:
            raise ValueError("Prediction fingerprint or counts do not match the release")
        checked.update(predictions_sha256=digest, predicted_docs=docs, predicted_chunks=chunks)
    return checked


def replay_submission(config: ReleaseConfig, output: Path) -> dict:
    verify_release(config, verify_predictions=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.artifact("submission"), output)
    actual = sha256_file(output)
    if actual != config.expected_hash("submission"):
        output.unlink(missing_ok=True)
        raise ValueError("Copied replay output failed its release hash")
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": actual}


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def build_replay_cache(config: ReleaseConfig, *, force: bool = False) -> dict:
    verify_release(config, verify_predictions=True)
    cache = config.artifact("cache")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        raise FileExistsError(f"Cache already exists: {cache}")
    staging = cache.with_suffix(".building.sqlite")
    staging.unlink(missing_ok=True)
    queries = pd.read_parquet(config.artifact("queries"), columns=["id", "query"])
    corpus = pd.read_parquet(config.artifact("corpus"), columns=["id", "url"])
    urls = dict(zip(corpus.id.astype(int), corpus.url.astype(str), strict=True))
    connection = _connect(staging)
    try:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE queries (id INTEGER PRIMARY KEY, query TEXT NOT NULL);
            CREATE TABLE documents (
                query_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                doc_id INTEGER NOT NULL, url TEXT,
                PRIMARY KEY(query_id, ordinal)
            );
            CREATE INDEX documents_query_doc ON documents(query_id, doc_id);
            CREATE TABLE chunks (
                chunk_ref INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL, ordinal INTEGER NOT NULL,
                doc_id INTEGER NOT NULL, text_zlib BLOB NOT NULL, text_sha256 TEXT NOT NULL,
                UNIQUE(query_id, ordinal)
            );
            CREATE INDEX chunks_query ON chunks(query_id, ordinal);
        """)
        connection.executemany(
            "INSERT INTO queries(id, query) VALUES (?, ?)",
            ((int(row.id), str(row.query)) for row in queries.itertuples(index=False)),
        )
        rows = doc_count = chunk_count = 0
        for row in iter_submission_rows(config.artifact("submission")):
            qid = int(row["id"])
            connection.executemany(
                "INSERT INTO documents(query_id, ordinal, doc_id, url) VALUES (?, ?, ?, ?)",
                ((qid, i, int(doc_id), urls.get(int(doc_id)))
                 for i, doc_id in enumerate(row["relevant_docs"])),
            )
            chunk_values = []
            for i, chunk in enumerate(row["relevant_chunks"]):
                text = str(chunk["chunk_text"])
                chunk_values.append((
                    qid, i, int(chunk["doc_id"]),
                    sqlite3.Binary(zlib.compress(text.encode("utf-8"), level=6)),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ))
            connection.executemany(
                "INSERT INTO chunks(query_id, ordinal, doc_id, text_zlib, text_sha256) "
                "VALUES (?, ?, ?, ?, ?)", chunk_values,
            )
            rows += 1
            doc_count += len(row["relevant_docs"])
            chunk_count += len(chunk_values)
            if rows % 100 == 0:
                connection.commit()
        metadata = {
            "schema_version": "replay_cache_v1",
            "release_id": config.release_id,
            "submission_sha256": config.expected_hash("submission"),
            "predictions_sha256": config.expected_hash("predictions"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "queries": str(rows), "documents": str(doc_count), "chunks": str(chunk_count),
        }
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
        connection.commit()
        if (rows, doc_count, chunk_count) != (
            config.expected_count("queries"), config.expected_count("predicted_docs"),
            config.expected_count("predicted_chunks"),
        ):
            raise ValueError("Replay cache counts do not match release config")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        connection.close()
        staging.unlink(missing_ok=True)
        raise
    finally:
        try:
            connection.close()
        except Exception:
            pass
    if cache.exists():
        cache.unlink()
    staging.replace(cache)
    return {"cache": str(cache), "bytes": cache.stat().st_size,
            "queries": rows, "documents": doc_count, "chunks": chunk_count}
