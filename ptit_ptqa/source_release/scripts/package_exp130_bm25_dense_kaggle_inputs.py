"""Package public-query translations and cached multilingual BM25 candidates for Kaggle.

The archive contains test-query data and must be uploaded to a private Kaggle Dataset.
The full active text/embedding index is attached separately to the Kaggle notebook.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/kaggle_inputs"
OUT = OUT_DIR / "exp130_bm25_dense_passage_input.zip"
BM25 = ROOT / "data/indices/bm25_document/bm25_candidates.parquet"
QUERIES = ROOT / "data/raw/btc_provided/query_translated.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    candidates = pd.read_parquet(BM25, columns=["query_id", "branch", "rank", "doc_id", "search_version", "branch_k"])
    queries = pd.read_parquet(QUERIES, columns=["id", "query", "query_en", "query_zh"])
    if len(queries) != 1200 or queries.id.nunique() != 1200:
        raise RuntimeError("Expected 1,200 unique official queries")
    if not set(candidates.query_id.astype(int)) == set(queries.id.astype(int)):
        raise RuntimeError("BM25 cache and query IDs do not match")
    if set(candidates.search_version) != {"bm25f_idf_pruned_v1"}:
        raise RuntimeError("Unexpected BM25 search version")
    if set(candidates.branch) != {"vi", "en", "zh"}:
        raise RuntimeError("Expected VI/EN/ZH candidate branches")
    manifest = {
        "experiment": 130,
        "purpose": "BM25 candidate generation followed by BGE-M3 exact dense reranking over the active chunk vectors",
        "queries": len(queries),
        "bm25_rows": len(candidates),
        "bm25_search_version": "bm25f_idf_pruned_v1",
        "branch_k": sorted(map(int, candidates.branch_k.unique())),
        "branches": sorted(candidates.branch.unique().tolist()),
        "bm25_sha256": sha256(BM25),
        "queries_sha256": sha256(QUERIES),
        "index_expected_separately": "private Kaggle Dataset containing the active index root (chunks_metadata.parquet, emb_dense/, text/)",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "exp130_bm25_dense_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(BM25, "bm25_candidates.parquet")
            archive.write(QUERIES, "query_translated.parquet")
            archive.write(manifest_path, "input_manifest.json")
        with zipfile.ZipFile(OUT) as archive:
            if archive.namelist() != ["bm25_candidates.parquet", "query_translated.parquet", "input_manifest.json"]:
                raise RuntimeError("Unexpected input archive members")
            if archive.testzip() is not None:
                raise RuntimeError("Input archive CRC check failed")
    except Exception:
        OUT.unlink(missing_ok=True)
        raise
    result = {**manifest, "archive": str(OUT), "archive_bytes": OUT.stat().st_size, "archive_sha256": sha256(OUT), "crc_pass": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
