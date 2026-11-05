"""Smaller copy of data/indexes/meta.duckdb holding only the documents the final assembly of S11 reads.

  python scripts/analysis/slim_meta.py --src <full meta.duckdb> --out data/indexes/meta.duckdb
Keeps the `documents` rows (doc_id, url, host, lang, title, text, n_chars) of every reranked candidate
(qwen3-0.6b-all, qwen3-4b-all-tw-long), of the base list's relevant_docs and of the http/https twin pages those
docs read (exact_equivalent in the team snapshot's corpus_annotated.parquet). fused_sub.py, rerank_proxy.py and
add_lang_docs.py only call MetaStore.documents, so they run unchanged on it; the full store (14 GB, with the chunks
and aliases tables) is rebuilt by `medret build-meta` from data/parsed and data/chunks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from medret.submit import iter_submission

R = Path("data/kaggle/rerank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/indexes/meta.duckdb"))
    ap.add_argument("--base", default="data/runs/sub-vi-rrfwp-zh30docs-v1/submission.json")
    ap.add_argument("--annotated", default="data/external/gloryvu_20261023/corpus_annotated.parquet")
    a = ap.parse_args()
    ids = set()
    for f in ("qwen3-0.6b-all", "qwen3-4b-all-tw-long"):
        ids |= set(pd.read_parquet(R / f / "scores.parquet", columns=["doc_id"])["doc_id"])
    for x in iter_submission(a.base):
        ids |= {int(d) for d in x["relevant_docs"]}
    ann = pd.read_parquet(a.annotated, columns=["id", "canonical_id", "dup_type"])
    ann = ann[(ann["dup_type"] == "exact_equivalent") & ann["id"].isin(ids)]
    ids |= set(ann["canonical_id"].astype("int64"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        a.out.unlink()
    # the ids go through a parquet file: scanning a registered DataFrame inside this cross-database query crashed
    # DuckDB 1.5 with pandas 3 (GIL error)
    ids_file = a.out.with_suffix(".ids.parquet")
    pd.DataFrame({"id": pd.Series(sorted(ids), dtype="int64")}).to_parquet(ids_file, index=False)
    con = duckdb.connect(str(a.out))
    con.execute("SET memory_limit = '2GB'")
    con.execute(f"ATTACH '{a.src.as_posix()}' AS src (READ_ONLY)")
    con.execute(f"""CREATE TABLE documents AS SELECT doc_id, url, host, lang, title, text, n_chars
                    FROM src.documents WHERE doc_id IN (SELECT id FROM read_parquet('{ids_file.as_posix()}'))
                    ORDER BY doc_id""")
    con.execute("CREATE UNIQUE INDEX documents_pk ON documents (doc_id)")
    n = con.execute("SELECT count(*) FROM documents").fetchone()[0]
    con.close()
    ids_file.unlink()
    print(f"{n:,} of {len(ids):,} requested docs -> {a.out} ({a.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
