"""Small persistent key-value cache (SQLite, stdlib) for model outputs: rerank scores, query vectors.

Keys are sha1 of (namespace, model, inputs), so a cache never mixes models or input formats. Values are bytes
(numpy arrays) or JSON. Safe to share between processes of one machine (SQLite locking); not meant for the corpus.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from pathlib import Path


def cache_key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


class KVCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB NOT NULL)")

    def get_many(self, keys: Iterable[str]) -> dict[str, bytes]:
        keys = list(keys)
        out: dict[str, bytes] = {}
        for i in range(0, len(keys), 900):  # SQLite variable limit
            part = keys[i : i + 900]
            q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(part))})"
            out.update(dict(self.con.execute(q, part).fetchall()))
        return out

    def put_many(self, items: Iterable[tuple[str, bytes]]) -> None:
        with self.con:
            self.con.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", list(items))
