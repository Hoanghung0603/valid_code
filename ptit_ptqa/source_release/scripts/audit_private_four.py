"""Independent structural audit of four append-only private candidates."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from scripts.build_final_sprint_exp142_missing_parents import read_rows
from scripts.generate_translation_submission import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_private01_exp143_replay.zip"
NAMES = [
    "submission_private02_bm25_consensus_add2.zip",
    "submission_private03_p2a_add5.zip",
    "submission_private04_combined.zip",
    "submission_private05_bm25_broad_add5.zip",
]


def audit() -> dict:
    official = set(pd.read_parquet(ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet",
                                   columns=["id"])["id"].astype(int))
    files = [ROOT / name for name in NAMES]
    for path in files:
        if not path.exists() or path.stat().st_size >= 100_000_000:
            raise ValueError(f"Missing/oversize candidate: {path}")
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise ValueError(f"Invalid ZIP: {path}")
    stats = {name: {"queries": 0, "docs": 0, "chunks": 0, "added_docs": 0,
                    "added_chunks": 0, "changed_queries": 0} for name in NAMES}
    iterators = [read_rows(path) for path in files]
    for rows in zip(read_rows(CONTROL), *iterators, strict=True):
        old = rows[0]
        qid = int(old["id"])
        old_docs = set(map(int, old["relevant_docs"]))
        old_chunks = {(int(c["doc_id"]), str(c["chunk_text"])) for c in old["relevant_chunks"]}
        for name, row in zip(NAMES, rows[1:], strict=True):
            if int(row["id"]) != qid:
                raise ValueError(f"Query ID/order mismatch: {name}")
            docs = list(map(int, row["relevant_docs"]))
            chunks = [(int(c["doc_id"]), str(c["chunk_text"])) for c in row["relevant_chunks"]]
            if (len(docs) != len(set(docs)) or len(chunks) != len(set(chunks)) or
                    not set(docs).issubset(official) or not old_docs.issubset(docs) or
                    not old_chunks.issubset(chunks) or
                    not {d for d, _ in chunks}.issubset(official)):
                raise ValueError(f"ID/source/append-only audit failed: {name}, query {qid}")
            s = stats[name]
            s["queries"] += 1
            s["docs"] += len(docs)
            s["chunks"] += len(chunks)
            s["added_docs"] += len(docs) - len(old_docs)
            s["added_chunks"] += len(chunks) - len(old_chunks)
            old_chunk_list = [(int(c["doc_id"]), str(c["chunk_text"])) for c in old["relevant_chunks"]]
            s["changed_queries"] += int(docs != old["relevant_docs"] or chunks != old_chunk_list)
    for name, s in stats.items():
        if s["queries"] != 1200:
            raise ValueError(f"Query coverage failed: {name}")
        path = ROOT / name
        s["bytes"] = path.stat().st_size
        s["sha256"] = sha256_file(path)
    return stats


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
