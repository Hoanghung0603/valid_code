"""Metadata store: one DuckDB file joining documents, chunks, embedding inputs and URL aliases.

Tables (built from the parquet outputs of `medret parse / chunk / export-embed` and the crawl aliases):
  documents (doc_id PK, url, host, lang, title, text, n_chars)
  chunks    (chunk_id PK, doc_id, lang, start, end, heading_path, n_tokens, input_id)   input_id: row of the vector
  aliases   (id, fetched_id)   every corpus id -> the fetched id with the same canonical URL (NULL if not fetched)
chunk_text is always sliced in Python from documents.text with (start, end): Python slicing on code points is the
contract (architecture.md section 3), so DuckDB's substring is not used for it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DB_FILE = "meta.duckdb"


def _glob(p: Path) -> str:
    return (p / "*" / "*.parquet").as_posix()


def build_meta(db_path: Path, docs_dir: Path, chunks_dir: Path, inputs_dir: Path, aliases_file: Path | None,
               memory_limit: str = "6GB", log=print) -> dict:
    import duckdb

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    t0 = time.time()
    con = duckdb.connect(str(tmp))
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{(db_path.parent / 'duckdb_tmp').as_posix()}'")
    con.execute(f"""CREATE TABLE documents AS
        SELECT doc_id, url, host, lang, title, text, n_chars FROM read_parquet('{_glob(docs_dir)}')""")
    con.execute(f"""CREATE TABLE chunks AS
        SELECT c.chunk_id, c.doc_id, c.lang, c.start, c."end", c.heading_path, c.n_tokens, i.input_id
        FROM read_parquet('{_glob(chunks_dir)}') c
        LEFT JOIN read_parquet('{(inputs_dir / 'shard-*.parquet').as_posix()}') i USING (embed_sha1)""")
    if aliases_file is not None and aliases_file.exists():
        con.execute(f"CREATE TABLE aliases AS SELECT id, fetched_id FROM read_parquet('{aliases_file.as_posix()}')")
    con.execute("CREATE UNIQUE INDEX documents_pk ON documents (doc_id)")
    con.execute("CREATE UNIQUE INDEX chunks_pk ON chunks (chunk_id)")
    con.execute("CREATE INDEX chunks_input ON chunks (input_id)")
    stats = {
        "documents": con.execute("SELECT count(*) FROM documents").fetchone()[0],
        "chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0],
        "chunks_without_input": con.execute("SELECT count(*) FROM chunks WHERE input_id IS NULL").fetchone()[0],
    }
    con.close()
    tmp.replace(db_path)
    log(f"meta store: {stats} in {time.time() - t0:,.0f}s -> {db_path}")
    return stats


class MetaStore:
    """Read-only lookups used by retrieval and submission writing."""

    def __init__(self, db_path: Path, memory_limit: str | None = None):
        import duckdb

        from medret.config import get_config

        self.con = duckdb.connect(str(db_path), read_only=True)
        # DuckDB's buffer pool defaults to 80% of RAM and keeps the pages it read: on the 14 GB laptop the
        # rerank of a deep run drained the free memory in seconds (2026-10-20)
        self.con.execute(f"SET memory_limit = '{memory_limit or get_config().index.meta_memory_limit}'")

    def _by_ids(self, sql: str, ids) -> pd.DataFrame:
        frame = pd.DataFrame({"id": pd.Series(list(ids), dtype="int64")})
        if frame.empty:
            return pd.DataFrame()
        self.con.register("_ids", frame)
        try:
            return self.con.execute(sql.replace("{ids}", "(SELECT id FROM _ids)")).df()
        finally:
            self.con.unregister("_ids")

    def chunks(self, chunk_ids) -> pd.DataFrame:
        """chunk_id, doc_id, lang, start, end, heading_path, input_id, title, chunk_text (verbatim slice)."""
        # sliced in SQL (substr counts characters, as Python slicing does), so the whole document text is not
        # copied once per chunk: 100 queries x 1,000 chunks pulled ~100k full texts into pandas
        return self._by_ids("""SELECT c.*, d.title, substr(d.text, c.start + 1, c."end" - c.start) AS chunk_text
                               FROM chunks c JOIN documents d USING (doc_id) WHERE c.chunk_id IN {ids}""", chunk_ids)

    def chunks_of_inputs(self, input_ids) -> pd.DataFrame:
        """Every chunk sharing a vector: input_id, chunk_id, doc_id (one input can serve several chunks)."""
        return self._by_ids("SELECT input_id, chunk_id, doc_id FROM chunks WHERE input_id IN {ids}", input_ids)

    def documents(self, doc_ids, with_text: bool = False) -> pd.DataFrame:
        cols = "doc_id, url, host, lang, title, n_chars" + (", text" if with_text else "")
        return self._by_ids(f"SELECT {cols} FROM documents WHERE doc_id IN {{ids}}", doc_ids)

    def twins(self, doc_ids) -> pd.DataFrame:
        """Corpus ids that are exact URL duplicates of the given fetched docs (id, fetched_id)."""
        return self._by_ids("SELECT id, fetched_id FROM aliases WHERE fetched_id IN {ids}", doc_ids)
