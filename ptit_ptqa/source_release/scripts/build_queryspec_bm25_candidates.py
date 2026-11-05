"""Build a full QuerySpec-expanded BM25 candidate pool for dense reranking.

This stage only expands candidate recall. The original Vietnamese query remains
the dense-reranking query in the downstream notebook, which limits semantic
drift from translations and atom decomposition.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sqlite3
import unicodedata
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECS = ROOT / "data/query_specs/queryspec_production_v1/query_specs.parquet"
DEFAULT_ATOMS = ROOT / "data/query_specs/queryspec_production_v1/evidence_atoms.parquet"
DEFAULT_QUERIES = ROOT / "data/raw/btc_provided/query.parquet"
DEFAULT_DB = ROOT / "data/indices/bm25_document/bm25_documents.sqlite"
DEFAULT_CONTROL = ROOT / "data/indices/bm25_document/bm25_candidates.parquet"
DEFAULT_TRANSLATIONS = ROOT / "data/query_translations/translation_production_v1.zip"
DEFAULT_OUTPUT = ROOT / "data/indices/bm25_document/queryspec_candidates_v1.parquet"
DEFAULT_AUDIT = ROOT / "data/audits/queryspec_candidates_v1/audit.json"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
TOKEN_RE = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lexical_expression(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    terms: list[str] = []
    for token in TOKEN_RE.findall(normalized):
        if CJK_RE.fullmatch(token):
            continue
        if len(token) > 1 and token not in terms:
            terms.append(token)
    cjk_terms: list[str] = []
    for sequence in CJK_RE.findall(normalized):
        if len(sequence) == 1:
            grams = ["z" + sequence]
        else:
            grams = ["z" + sequence[i:i + 2] for i in range(len(sequence) - 1)]
        for gram in grams:
            if gram not in cjk_terms:
                cjk_terms.append(gram)
    clauses = ['"' + term.replace('"', '""') + '"' for term in terms[:24]]
    clauses.extend('cjk:"' + term.replace('"', '""') + '"' for term in cjk_terms[:24])
    return " OR ".join(clauses)


def search(connection: sqlite3.Connection, text: str, k: int) -> list[tuple[int, float]]:
    expression = lexical_expression(text)
    if not expression:
        return []
    rows = connection.execute(
        "SELECT rowid,bm25(documents,8.0,3.0,1.0,1.0) AS score "
        "FROM documents WHERE documents MATCH ? ORDER BY score LIMIT ?",
        (expression, k),
    ).fetchall()
    return [(int(doc_id), float(score)) for doc_id, score in rows]


def query_views(query: str, spec: pd.Series | None, atoms: pd.DataFrame,
                translation: pd.Series | None) -> list[tuple[str, str, float]]:
    views: list[tuple[str, str, float]] = [("original_vi", query, 1.0)]
    if translation is not None:
        production_en = str(translation.get("query_en", "") or "").strip()
        production_zh = str(translation.get("query_zh", "") or "").strip()
        if production_en:
            views.append(("production_en", production_en, 0.35))
        if production_zh:
            views.append(("production_zh", production_zh, 0.35))
    if spec is None or str(spec.get("status", "")) != "ok":
        return views
    canonical = str(spec.get("canonical_vi", "") or "").strip()
    query_en = str(spec.get("query_en", "") or "").strip()
    query_zh = str(spec.get("query_zh", "") or "").strip()
    if canonical and canonical != query:
        views.append(("canonical_vi", canonical, 0.70))
    if query_en:
        views.append(("query_en", query_en, 0.35))
    if query_zh:
        views.append(("query_zh", query_zh, 0.35))
    atom_count = max(len(atoms), 1)
    # Keep total atom-family mass bounded so a long query cannot dominate merely
    # because the planner split it into more atoms.
    vi_weight = 0.75 / math.sqrt(atom_count)
    translated_weight = 0.25 / math.sqrt(atom_count)
    for atom in atoms.itertuples(index=False):
        for family, column, weight in (
            ("atom_vi", "retrieval_query_vi", vi_weight),
            ("atom_en", "retrieval_query_en", translated_weight),
            ("atom_zh", "retrieval_query_zh", translated_weight),
        ):
            value = str(getattr(atom, column, "") or "").strip()
            if value:
                views.append((family, value, weight))
    deduped: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for family, text, weight in views:
        key = unicodedata.normalize("NFKC", text).casefold().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append((family, text, weight))
    return deduped


def process_query(
    db_path: Path,
    query_id: int,
    query: str,
    spec: pd.Series | None,
    atoms: pd.DataFrame,
    translation: pd.Series | None,
    original_ranking: list[int],
    branch_k: int,
    output_k: int,
    rrf_k: int,
) -> tuple[list[dict], dict]:
    connection = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA mmap_size=8589934592")
    fused: dict[int, float] = defaultdict(float)
    hits: dict[int, int] = defaultdict(int)
    best_family: dict[int, tuple[float, str]] = {}
    views = query_views(query, spec, atoms, translation)
    try:
        for family, text, weight in views:
            ranking = ([(doc_id, 0.0) for doc_id in original_ranking[:branch_k]]
                       if family == "original_vi" else search(connection, text, branch_k))
            for rank, (doc_id, _bm25) in enumerate(ranking, 1):
                contribution = weight / (rrf_k + rank)
                fused[doc_id] += contribution
                hits[doc_id] += 1
                if contribution > best_family.get(doc_id, (-1.0, ""))[0]:
                    best_family[doc_id] = (contribution, family)
    finally:
        connection.close()
    ordered = sorted(fused, key=lambda doc_id: (-fused[doc_id], -hits[doc_id], doc_id))[:output_k]
    rows = [
        {
            "query_id": query_id,
            "rank": rank,
            "doc_id": doc_id,
            "rrf_score": fused[doc_id],
            "view_hits": hits[doc_id],
            "best_view_family": best_family[doc_id][1],
        }
        for rank, doc_id in enumerate(ordered, 1)
    ]
    spec_status = str(spec.get("status", "fallback_original")) if spec is not None else "fallback_original"
    return rows, {
        "query_id": query_id,
        "status": spec_status,
        "view_count": len(views),
        "candidate_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    parser.add_argument("--atoms", type=Path, default=DEFAULT_ATOMS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--branch-k", type=int, default=300)
    parser.add_argument("--output-k", type=int, default=2000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("Refusing to overwrite QuerySpec candidate output/audit")
    queries = pd.read_parquet(args.queries, columns=["id", "query"]).sort_values("id")
    specs = pd.read_parquet(args.specs).set_index("query_id", drop=False)
    atoms = pd.read_parquet(args.atoms)
    with zipfile.ZipFile(args.translations) as archive:
        if archive.testzip() is not None or "query_translated_production_v1.parquet" not in archive.namelist():
            raise RuntimeError("Production translation archive invalid")
        translations = pd.read_parquet(
            io.BytesIO(archive.read("query_translated_production_v1.parquet"))
        ).set_index("id")
    control = pd.read_parquet(args.control, columns=["query_id", "branch", "rank", "doc_id"])
    control = control[control.branch.eq("vi")].sort_values(["query_id", "rank"])
    original_rankings = {int(qid): frame.doc_id.astype(int).tolist()
                         for qid, frame in control.groupby("query_id", sort=False)}
    if (len(queries) != 1200 or queries.id.nunique() != 1200 or len(specs) != 1200
            or set(original_rankings) != set(queries.id.astype(int))
            or set(translations.index.astype(int)) != set(queries.id.astype(int))):
        raise RuntimeError("Expected 1,200 official queries and QuerySpecs")
    atoms_by_query = {int(qid): frame.copy() for qid, frame in atoms.groupby("query_id", sort=False)}
    all_rows: list[dict] = []
    diagnostics: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_query,
                args.db,
                int(row.id),
                str(row.query),
                specs.loc[int(row.id)] if int(row.id) in specs.index else None,
                atoms_by_query.get(int(row.id), pd.DataFrame()),
                translations.loc[int(row.id)],
                original_rankings[int(row.id)],
                args.branch_k,
                args.output_k,
                args.rrf_k,
            ): int(row.id)
            for row in queries.itertuples(index=False)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            rows, diagnostic = future.result()
            all_rows.extend(rows)
            diagnostics.append(diagnostic)
            if completed % 50 == 0 or completed == len(futures):
                print(f"BM25 QuerySpec {completed}/1200", flush=True)
    output = pd.DataFrame(all_rows).sort_values(["query_id", "rank"]).reset_index(drop=True)
    if output.query_id.nunique() != 1200 or output.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Candidate output coverage/uniqueness failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    diagnostics_frame = pd.DataFrame(diagnostics)
    audit = {
        "schema_version": "queryspec_bm25_candidates_v1",
        "purpose": "candidate expansion only; original VI query must rerank downstream",
        "queries": int(output.query_id.nunique()),
        "rows": len(output),
        "branch_k_per_view": args.branch_k,
        "output_k": args.output_k,
        "rrf_k": args.rrf_k,
        "status_counts": diagnostics_frame.status.value_counts().to_dict(),
        "view_count": diagnostics_frame.view_count.describe().to_dict(),
        "candidate_count": diagnostics_frame.candidate_count.describe().to_dict(),
        "inputs": {"specs_sha256": sha256_file(args.specs), "atoms_sha256": sha256_file(args.atoms),
                   "queries_sha256": sha256_file(args.queries), "db_sha256": sha256_file(args.db),
                   "original_bm25_cache_sha256": sha256_file(args.control),
                   "production_translations_sha256": sha256_file(args.translations)},
        "output_sha256": sha256_file(args.output),
        "limitation": "No qrels; candidate novelty is not relevance evidence.",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
