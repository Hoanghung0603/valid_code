from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import pandas as pd

from .config import ReleaseConfig


@dataclass
class Run:
    run_id: str
    query_id: int
    query: str
    mode: str
    created_at: float


class ReplayService:
    def __init__(self, config: ReleaseConfig):
        self.config = config
        self.cache = config.artifact("cache")
        if not self.cache.is_file():
            raise FileNotFoundError(
                f"Replay cache not found: {self.cache}. Run `python -m src.release.cli build-cache`."
            )
        self.runs: dict[str, Run] = {}
        self._source_cache: dict[int, tuple[str, str]] = {}
        self._source_lock = Lock()
        self._validate_cache()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.cache}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _validate_cache(self) -> None:
        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if (metadata.get("release_id") != self.config.release_id or
                metadata.get("submission_sha256") != self.config.expected_hash("submission")):
            raise ValueError("Replay cache fingerprint does not match release config")

    def list_queries(self, search: str = "", offset: int = 0, limit: int = 50) -> dict:
        limit = max(1, min(limit, int(self.config.payload["limits"]["api_page_size"])))
        offset = max(0, offset)
        with self._connect() as connection:
            if search.strip():
                token = f"%{search.strip()}%"
                total = connection.execute(
                    "SELECT count(*) FROM queries WHERE query LIKE ?", (token,)
                ).fetchone()[0]
                rows = connection.execute(
                    "SELECT id, query FROM queries WHERE query LIKE ? ORDER BY id LIMIT ? OFFSET ?",
                    (token, limit, offset),
                ).fetchall()
            else:
                total = connection.execute("SELECT count(*) FROM queries").fetchone()[0]
                rows = connection.execute(
                    "SELECT id, query FROM queries ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
                ).fetchall()
        return {"total": total, "offset": offset, "limit": limit,
                "items": [dict(row) for row in rows]}

    def get_query(self, query_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT id, query FROM queries WHERE id=?", (query_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown cached query_id")
        return dict(row)

    def get_graph_rows(self, query_id: int, scan_limit: int = 160) -> list[dict]:
        """Read cached chunk/document associations for UI graph, without starting a run."""
        with self._connect() as connection:
            documents = {int(row["doc_id"]): dict(row) for row in connection.execute(
                "SELECT doc_id, ordinal, url FROM documents WHERE query_id=? ORDER BY ordinal",
                (query_id,))}
            chunks = connection.execute(
                "SELECT chunk_ref, ordinal, doc_id, text_zlib FROM chunks WHERE query_id=? "
                "ORDER BY ordinal LIMIT ?", (query_id, scan_limit)).fetchall()
        rows = []
        for chunk in chunks:
            doc = documents.get(int(chunk["doc_id"]), {})
            rows.append({"doc_id": int(chunk["doc_id"]), "url": doc.get("url"),
                         "doc_rank": int(doc["ordinal"]) + 1 if doc else None,
                         "chunk_rank": int(chunk["ordinal"]) + 1,
                         "chunk_ref": int(chunk["chunk_ref"]),
                         "chunk_text": zlib.decompress(chunk["text_zlib"]).decode("utf-8")})
        if not rows:
            rows = [{"doc_id": doc_id, "url": doc.get("url"),
                     "doc_rank": int(doc["ordinal"]) + 1, "chunk_rank": None,
                     "chunk_ref": None, "chunk_text": ""}
                    for doc_id, doc in list(documents.items())[:scan_limit]]
        return rows

    def create_run(self, *, mode: str, query_id: int | None, query: str | None) -> Run:
        if mode != "replay":
            raise ValueError("Live retrieval is not enabled in this audited release")
        if query_id is None:
            raise ValueError("Replay requires query_id")
        with self._connect() as connection:
            row = connection.execute("SELECT id, query FROM queries WHERE id=?", (query_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown cached query_id")
        if query is not None and query != row["query"]:
            raise ValueError("Replay query text must exactly match the cached official query")
        run = Run(str(uuid.uuid4()), int(row["id"]), str(row["query"]), mode, time.time())
        self.runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> Run:
        if run_id not in self.runs:
            raise KeyError("Unknown run_id")
        return self.runs[run_id]

    def get_results(self, run_id: str, kind: str, offset: int, limit: int) -> dict:
        run = self.get_run(run_id)
        limit = max(1, min(limit, int(self.config.payload["limits"]["api_page_size"])))
        offset = max(0, offset)
        with self._connect() as connection:
            if kind == "docs":
                total = connection.execute(
                    "SELECT count(*) FROM documents WHERE query_id=?", (run.query_id,)
                ).fetchone()[0]
                rows = connection.execute(
                    "SELECT ordinal, doc_id, url FROM documents WHERE query_id=? "
                    "ORDER BY ordinal LIMIT ? OFFSET ?", (run.query_id, limit, offset)
                ).fetchall()
                items = [dict(row) for row in rows]
            elif kind == "chunks":
                total = connection.execute(
                    "SELECT count(*) FROM chunks WHERE query_id=?", (run.query_id,)
                ).fetchone()[0]
                rows = connection.execute(
                    "SELECT chunk_ref, ordinal, doc_id, text_zlib, text_sha256 FROM chunks "
                    "WHERE query_id=? ORDER BY ordinal LIMIT ? OFFSET ?",
                    (run.query_id, limit, offset),
                ).fetchall()
                items = [{
                    "chunk_ref": row["chunk_ref"], "ordinal": row["ordinal"],
                    "doc_id": row["doc_id"], "chunk_text": zlib.decompress(row["text_zlib"]).decode("utf-8"),
                    "text_sha256": row["text_sha256"],
                } for row in rows]
            else:
                raise ValueError("kind must be docs or chunks")
        return {"kind": kind, "total": total, "offset": offset, "limit": limit, "items": items}

    def _load_source(self, doc_id: int) -> tuple[str | None, str | None]:
        if doc_id in self._source_cache:
            return self._source_cache[doc_id]
        source_dir = self.config.artifact("source_text_dir")
        with self._source_lock:
            if doc_id in self._source_cache:
                return self._source_cache[doc_id]
            for path in sorted(source_dir.glob("part_*.parquet")):
                ids = pd.read_parquet(path, columns=["doc_id"])["doc_id"]
                positions = ids.index[ids.eq(doc_id)]
                if len(positions):
                    frame = pd.read_parquet(path, columns=["doc_id", "text"])
                    text = str(frame.loc[positions[0], "text"])
                    value = (text, path.name)
                    self._source_cache[doc_id] = value
                    return value
        return None, None

    def evidence(self, run_id: str, chunk_ref: int) -> dict:
        run = self.get_run(run_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT doc_id, text_zlib, text_sha256 FROM chunks WHERE query_id=? AND chunk_ref=?",
                (run.query_id, chunk_ref),
            ).fetchone()
            if row is None:
                raise KeyError("chunk_ref does not belong to this run")
            url_row = connection.execute(
                "SELECT url FROM documents WHERE query_id=? AND doc_id=? LIMIT 1",
                (run.query_id, row["doc_id"]),
            ).fetchone()
        chunk_text = zlib.decompress(row["text_zlib"]).decode("utf-8")
        source, part = self._load_source(int(row["doc_id"]))
        if source is None:
            return {"doc_id": row["doc_id"], "url": url_row["url"] if url_row else None,
                    "chunk_text": chunk_text, "text_sha256": row["text_sha256"],
                    "resolved": False, "reason": "Document is not present in the configured source snapshot"}
        start = source.find(chunk_text)
        if start < 0:
            return {"doc_id": row["doc_id"], "url": url_row["url"] if url_row else None,
                    "chunk_text": chunk_text, "text_sha256": row["text_sha256"],
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "source_part": part, "resolved": False,
                    "reason": "Exact chunk is not a contiguous substring of the source snapshot"}
        end = start + len(chunk_text)
        radius = int(self.config.payload["limits"]["evidence_context_chars"]) // 2
        context_start, context_end = max(0, start - radius), min(len(source), end + radius)
        return {"doc_id": row["doc_id"], "url": url_row["url"] if url_row else None,
                "chunk_text": chunk_text, "text_sha256": row["text_sha256"],
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "source_part": part, "resolved": True, "start": start, "end": end,
                "context_start": context_start, "context": source[context_start:context_end],
                "highlight_start": start - context_start, "highlight_end": end - context_start}

    def export(self, run_id: str) -> dict:
        run = self.get_run(run_id)
        with self._connect() as connection:
            docs = connection.execute(
                "SELECT doc_id FROM documents WHERE query_id=? ORDER BY ordinal", (run.query_id,)
            ).fetchall()
            chunk_rows = connection.execute(
                "SELECT doc_id, text_zlib FROM chunks WHERE query_id=? ORDER BY ordinal", (run.query_id,)
            ).fetchall()
        chunks = [{"doc_id": row["doc_id"],
                   "chunk_text": zlib.decompress(row["text_zlib"]).decode("utf-8")}
                  for row in chunk_rows]
        return {
            "scope": "single_query_review_export_not_btc_submission",
            "release_id": self.config.release_id, "mode": run.mode,
            "id": run.query_id, "query": run.query,
            "relevant_docs": [row["doc_id"] for row in docs],
            "relevant_chunks": chunks,
            "provenance": {"submission_sha256": self.config.expected_hash("submission"),
                           "predictions_sha256": self.config.expected_hash("predictions")},
        }
