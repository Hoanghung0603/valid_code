"""Build a clean document-level BM25F index for the active BTC corpus.

The index is deliberately independent from the old Exp20 SQLite database:

* source documents are exactly the validated active embedding index;
* Vietnamese diacritics are preserved;
* URL/title, lead, body and Chinese bigrams are separate BM25F fields;
* the FTS table is contentless because source text already lives in Parquet;
* a staging database is validated before atomically replacing the active DB.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import pyarrow.parquet as pq


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "indices" / "new_dataset" / "index"
TEXT_DIR = INDEX_DIR / "text"
REGISTRY = ROOT / "data" / "embedded_samples.parquet"
OFFICIAL = ROOT / "data" / "raw" / "btc_provided" / "links_corpus_vi_en_zh.parquet"
OUTPUT_DIR = ROOT / "data" / "indices" / "bm25_document"
ACTIVE_DB = OUTPUT_DIR / "bm25_documents.sqlite"
STAGING_DB = OUTPUT_DIR / "bm25_documents.staging.sqlite"

CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
TAG_RE = re.compile(r"<[^>]{1,200}>")
SPACE_RE = re.compile(r"\s+")
TITLE_SPLIT_RE = re.compile(r"[-_]+")
BOILERPLATE = {
    "trang chủ", "giới thiệu", "cơ cấu tổ chức", "đội ngũ bác sĩ", "đặt lịch khám",
    "gọi tổng đài", "tin tức", "nội dung bài viết", "toggle", "tác giả", "đăng nhập",
    "home", "menu", "contact", "about us", "skip to content", "table of contents",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-max-chars", type=int, default=16_000)
    parser.add_argument("--lead-chars", type=int, default=1_800)
    parser.add_argument("--title-scan-chars", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(str(value))).lower()
    value = TAG_RE.sub(" ", value)
    return SPACE_RE.sub(" ", value).strip()


def url_title(url: str) -> str:
    path = unquote(urlsplit(str(url)).path).rstrip("/")
    slug = path.rsplit("/", 1)[-1]
    slug = re.sub(r"\.(?:html?|shtml|php|aspx?)$", "", slug, flags=re.IGNORECASE)
    return SPACE_RE.sub(" ", TITLE_SPLIT_RE.sub(" ", slug)).strip()


def first_meaningful_line(text: str, scan_chars: int) -> str:
    for raw in str(text)[:scan_chars].splitlines():
        line = normalize(raw)
        if not (12 <= len(line) <= 300):
            continue
        if line in BOILERPLATE or any(line.startswith(item + " ") for item in BOILERPLATE):
            continue
        if line.count("|") > 3:
            continue
        return line
    return ""


def cjk_bigrams(text: str) -> str:
    grams: list[str] = []
    for sequence in CJK_SEQUENCE_RE.findall(text):
        if len(sequence) == 1:
            grams.append("z" + sequence)
        else:
            grams.extend("z" + sequence[index : index + 2] for index in range(len(sequence) - 1))
    return " ".join(grams)


def source_rows(args: argparse.Namespace, url_by_id: dict[int, str]):
    for path in sorted(TEXT_DIR.glob("part_*.parquet")):
        table = pq.read_table(path, columns=["doc_id", "text"]).to_pandas()
        for row in table.itertuples(index=False):
            doc_id = int(row.doc_id)
            raw = str(row.text)
            normalized_text = normalize(raw[: args.body_max_chars])
            lead = normalized_text[: args.lead_chars]
            # Keep BM25F fields disjoint: indexing the lead again in body both
            # doubles common terms and distorts document-length normalization.
            body = normalized_text[args.lead_chars :]
            title = normalize(
                " ".join(
                    part
                    for part in (
                        url_title(url_by_id[doc_id]),
                        first_meaningful_line(raw, args.title_scan_chars),
                    )
                    if part
                )
            )
            yield (doc_id, title, lead, body, cjk_bigrams(normalized_text))


def configure(connection: sqlite3.Connection) -> None:
    # This is a disposable staging DB rebuilt from immutable Parquet sources.
    # Disabling journaling avoids writing the multi-gigabyte FTS index twice;
    # the validated DB is still atomically swapped into place at the end.
    connection.execute("PRAGMA page_size=32768")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA mmap_size=8589934592")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if STAGING_DB.exists() or STAGING_DB.with_name(STAGING_DB.name + "-wal").exists():
        raise FileExistsError(f"Staging database already exists: {STAGING_DB}")

    index_stats = json.loads((INDEX_DIR / "stats.json").read_text(encoding="utf-8"))
    expected_docs = int(index_stats["total_docs"])
    registry = pd.read_parquet(REGISTRY, columns=["id", "url"])
    if len(registry) != expected_docs or registry["id"].duplicated().any():
        raise RuntimeError("Embedded registry does not match active index stats")
    url_by_id = dict(zip(registry["id"].astype(int), registry["url"].astype(str)))

    official = pq.read_table(OFFICIAL, columns=["id"])["id"].to_pylist()
    official_ids = set(map(int, official))
    if not set(url_by_id).issubset(official_ids):
        raise RuntimeError("Embedded registry contains out-of-corpus IDs")

    started = time.time()
    connection = sqlite3.connect(STAGING_DB)
    configure(connection)
    connection.execute(
        "CREATE VIRTUAL TABLE documents USING fts5("
        "title, lead, body, cjk, content='', "
        "tokenize='unicode61 remove_diacritics 0')"
    )
    connection.execute("CREATE TABLE build_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    insert_sql = "INSERT INTO documents(rowid,title,lead,body,cjk) VALUES(?,?,?,?,?)"
    batch: list[tuple[int, str, str, str, str]] = []
    written = 0
    connection.execute("BEGIN")
    for item in source_rows(args, url_by_id):
        batch.append(item)
        if len(batch) >= args.batch_size:
            connection.executemany(insert_sql, batch)
            written += len(batch)
            batch.clear()
            if written % 100_000 == 0:
                connection.commit()
                connection.execute("BEGIN")
                print(f"Indexed {written:,}/{expected_docs:,} documents", flush=True)
    if batch:
        connection.executemany(insert_sql, batch)
        written += len(batch)
    connection.commit()
    if written != expected_docs:
        raise RuntimeError(f"Indexed {written:,} rows, expected {expected_docs:,}")

    print("Optimizing FTS segments...", flush=True)
    connection.execute("INSERT INTO documents(documents) VALUES('optimize')")
    built_at = datetime.now().astimezone().isoformat()
    metadata = {
        "built_at": built_at,
        "documents": written,
        "source_index_stats": index_stats,
        "official_corpus_sha256": sha256_file(OFFICIAL),
        "body_max_chars": args.body_max_chars,
        "lead_chars": args.lead_chars,
        "title_scan_chars": args.title_scan_chars,
        "fields": {"title": 8.0, "lead": 3.0, "body": 1.0, "cjk": 1.0},
        "tokenizer": "unicode61 remove_diacritics 0 + explicit CJK bigrams",
    }
    connection.executemany(
        "INSERT INTO build_metadata(key,value) VALUES(?,?)",
        [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
    )
    connection.commit()

    row_count = int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
    smoke = connection.execute(
        "SELECT rowid FROM documents WHERE documents MATCH ? "
        "ORDER BY bm25(documents,8.0,3.0,1.0,1.0) LIMIT 5",
        ('"đái" OR "tháo" OR "đường" OR "diabetes"',),
    ).fetchall()
    if row_count != expected_docs or not smoke:
        raise RuntimeError("BM25 index validation failed")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")

    backup = None
    if ACTIVE_DB.exists():
        backup = ACTIVE_DB.with_name(
            "bm25_documents.backup_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".sqlite"
        )
        ACTIVE_DB.rename(backup)
    try:
        os.replace(STAGING_DB, ACTIVE_DB)
    except Exception:
        if backup and not ACTIVE_DB.exists() and backup.exists():
            backup.rename(ACTIVE_DB)
        raise

    print("=" * 80)
    print(f"BM25 DOCUMENT INDEX COMPLETE in {(time.time() - started) / 60:.1f} minutes")
    print(f"Database: {ACTIVE_DB}")
    print(f"Documents: {row_count:,}")
    print(f"Size: {ACTIVE_DB.stat().st_size / (1024 ** 3):.2f} GiB")
    print(f"Recoverable backup: {backup}")
    print("=" * 80)


if __name__ == "__main__":
    main()
