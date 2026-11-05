"""Build a fresh multilingual BM25 cache for a versioned translation treatment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from generate_bm25_icd_submissions import (  # noqa: E402
    connect,
    fts_query,
    load_document_frequencies,
    search_branch,
)


DEFAULT_QUERIES = ROOT / "data/query_translations/query_translated_nllb600m.parquet"
DEFAULT_FALLBACK = ROOT / "data/raw/btc_provided/query_translated.parquet"
DEFAULT_DB = ROOT / "data/indices/bm25_document/bm25_documents.sqlite"
DEFAULT_OUTPUT = ROOT / "data/indices/bm25_document/bm25_candidates_nllb600m_k700.parquet"
DEFAULT_EFFECTIVE = ROOT / "data/query_translations/query_translated_nllb600m_effective.parquet"
DEFAULT_AUDIT = ROOT / "data/audits/bm25_translation_nllb600m_k700.json"
DEFAULT_CONTROL = ROOT / "data/indices/bm25_document/bm25_candidates.parquet"
SEARCH_VERSION = "bm25f_idf_pruned_nllb600m_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--effective-output", type=Path, default=DEFAULT_EFFECTIVE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--branch-k", type=int, default=700)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def effective_queries(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int]]:
    treatment = pd.read_parquet(args.queries).sort_values("id").reset_index(drop=True)
    fallback = pd.read_parquet(args.fallback).set_index("id")
    if len(treatment) != 1200 or treatment["id"].duplicated().any():
        raise RuntimeError("Treatment must contain 1,200 unique query IDs")
    if set(treatment["id"].astype(int)) != set(map(int, fallback.index)):
        raise RuntimeError("Treatment/fallback query IDs differ")
    counts: dict[str, int] = {}
    for language in ("en", "zh"):
        query_col = f"query_{language}"
        status_col = f"translation_status_{language}"
        ok = treatment[status_col].eq("ok") & treatment[query_col].fillna("").str.strip().ne("")
        counts[f"nllb_{language}"] = int(ok.sum())
        counts[f"fallback_{language}"] = int((~ok).sum())
        fallback_values = treatment["id"].map(fallback[query_col])
        treatment[query_col] = treatment[query_col].where(ok, fallback_values)
        treatment[f"translation_source_{language}"] = np.where(ok, "nllb600m", "qwen2.5_1.5b_fallback")
    keep = [
        "id", "query", "query_en", "query_zh",
        "translation_source_en", "translation_source_zh",
    ]
    effective = treatment[keep].copy()
    args.effective_output.parent.mkdir(parents=True, exist_ok=True)
    effective.to_parquet(args.effective_output, index=False, compression="snappy")
    return effective, counts


def build_candidates(args: argparse.Namespace, queries: pd.DataFrame) -> pd.DataFrame:
    if args.output.exists() and not args.force:
        existing = pd.read_parquet(args.output)
        if (
            set(existing.get("search_version", [])) == {SEARCH_VERSION}
            and set(existing.get("branch_k", []).astype(int)) == {args.branch_k}
            and set(existing["query_id"].astype(int)) == set(queries["id"].astype(int))
        ):
            print(f"Loaded treatment cache: {len(existing):,} rows")
            return existing
        raise RuntimeError("Existing treatment cache does not match requested run; use --force deliberately")

    connection = connect(args.db)
    document_frequencies, total_documents = load_document_frequencies(connection, queries)
    rows: list[dict] = []
    for number, row in enumerate(queries.itertuples(index=False), 1):
        values = {"vi": str(row.query), "en": str(row.query_en), "zh": str(row.query_zh)}
        sources = {
            "vi": "official_vi",
            "en": str(row.translation_source_en),
            "zh": str(row.translation_source_zh),
        }
        for branch, value in values.items():
            sqlite_query = fts_query(value, branch, document_frequencies, total_documents)
            for rank, (doc_id, score) in enumerate(search_branch(connection, sqlite_query, args.branch_k), 1):
                rows.append(
                    {
                        "query_id": int(row.id),
                        "branch": branch,
                        "rank": rank,
                        "doc_id": doc_id,
                        "score": score,
                        "translation_source": sources[branch],
                        "search_version": SEARCH_VERSION,
                        "branch_k": args.branch_k,
                    }
                )
        if number % 100 == 0:
            print(f"BM25 treatment searched {number}/{len(queries)}", flush=True)
    connection.close()
    candidates = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(args.output, index=False, compression="snappy")
    print(f"Saved {len(candidates):,} rows -> {args.output}")
    return candidates


def compare_control(treatment: pd.DataFrame, control_path: Path) -> dict:
    if not control_path.exists():
        return {"available": False}
    control = pd.read_parquet(control_path, columns=["query_id", "branch", "rank", "doc_id"])
    result: dict = {"available": True, "control_rows": len(control)}
    for branch in ("vi", "en", "zh"):
        old = control[control.branch.eq(branch)].groupby("query_id").doc_id.apply(lambda x: set(map(int, x)))
        new = treatment[treatment.branch.eq(branch) & treatment["rank"].le(500)].groupby("query_id").doc_id.apply(lambda x: set(map(int, x)))
        overlaps: list[float] = []
        novel: list[float] = []
        for query_id in sorted(set(old.index) & set(new.index)):
            overlaps.append(len(old[query_id] & new[query_id]) / max(len(old[query_id]), 1))
            novel.append(float(len(new[query_id] - old[query_id])))
        result[branch] = {
            "queries": len(overlaps),
            "overlap_fraction_at500": percentiles(overlaps),
            "novel_docs_at500": percentiles(novel),
        }
    return result


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    queries, fallback_counts = effective_queries(args)
    candidates = build_candidates(args, queries)
    audit = {
        "created_at": datetime.now().astimezone().isoformat(),
        "search_version": SEARCH_VERSION,
        "branch_k": args.branch_k,
        "rows": len(candidates),
        "queries": int(candidates.query_id.nunique()),
        "branch_rows": candidates.branch.value_counts().to_dict(),
        "translation_sources": fallback_counts,
        "inputs": {
            "translations_sha256": sha256_file(args.queries),
            "effective_queries_sha256": sha256_file(args.effective_output),
            "bm25_db_sha256": sha256_file(args.db),
        },
        "output_sha256": sha256_file(args.output),
        "control_comparison": compare_control(candidates, args.control),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
