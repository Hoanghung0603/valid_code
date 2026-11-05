"""Compare original and QuerySpec query views on the 120-query BM25 pilot."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fts_expression(text: str) -> str:
    terms = re.findall(r"[\w\u3400-\u9fff]+", str(text).lower(), flags=re.UNICODE)
    terms = list(dict.fromkeys(t for t in terms if len(t) > 1))[:24]
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def search(conn: sqlite3.Connection, text: str, k: int) -> list[int]:
    expr = fts_expression(text)
    if not expr:
        return []
    rows = conn.execute(
        "SELECT rowid FROM documents WHERE documents MATCH ? ORDER BY bm25(documents, 3.0, 2.0, 1.0, 1.0) LIMIT ?",
        (expr, k),
    ).fetchall()
    return [int(row[0]) for row in rows]


def balanced_union(rankings: list[list[int]], k: int) -> list[int]:
    """Round-robin ranked lists so the first atom cannot consume the whole cap."""
    merged: list[int] = []
    seen: set[int] = set()
    depth = 0
    while len(merged) < k:
        added = False
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in seen:
                seen.add(ranking[depth])
                merged.append(ranking[depth])
                added = True
                if len(merged) >= k:
                    break
        if not added and depth >= max((len(r) for r in rankings), default=0):
            break
        depth += 1
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-dir", type=Path, default=ROOT / "data/audits/queryspec_pilot_v6_final")
    p.add_argument("--queries", type=Path, default=ROOT / "data/raw/btc_provided/query.parquet")
    p.add_argument("--db", type=Path, default=ROOT / "data/indices/bm25_document/bm25_documents.sqlite")
    p.add_argument("--control", type=Path, default=ROOT / "data/indices/bm25_document/bm25_candidates.parquet")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/audits/queryspec_bm25_pilot_v1")
    p.add_argument("--k", type=int, default=500)
    a = p.parse_args()
    specs = pd.read_parquet(a.pilot_dir / "query_specs.parquet")
    atoms = pd.read_parquet(a.pilot_dir / "evidence_atoms.parquet")
    originals = pd.read_parquet(a.queries).set_index("id")
    control = pd.read_parquet(a.control, columns=["query_id", "branch", "rank", "doc_id"])
    ids = sorted(map(int, specs.query_id.unique()))
    conn = sqlite3.connect(f"file:{a.db.resolve().as_posix()}?mode=ro", uri=True)
    rows = []
    for qid in ids:
        query = str(originals.loc[qid, "query"])
        spec = specs[specs.query_id.eq(qid)].iloc[0]
        qa = atoms[atoms.query_id.eq(qid)]
        canonical = str(spec.get("canonical_vi", "") or "").strip()
        views = {
            "original": [query],
            "normalized_vi": [canonical or query],
            "atoms_vi": qa.get("retrieval_query_vi", pd.Series(dtype=str)).dropna().astype(str).tolist(),
            "canonical_plus_atoms": [canonical or query]
                + qa.get("retrieval_query_vi", pd.Series(dtype=str)).dropna().astype(str).tolist(),
        }
        baseline = set(map(int, control[(control.query_id.eq(qid)) & (control.branch.eq("vi")) & (control["rank"].le(a.k))].doc_id))
        for variant, texts in views.items():
            rankings = [search(conn, text, a.k) for text in texts if text.strip()]
            if variant in {"atoms_vi", "canonical_plus_atoms"}:
                unique = balanced_union(rankings, a.k)
            else:
                unique = rankings[0][:a.k] if rankings else []
            cand = set(unique)
            rows.append({"query_id": qid, "variant": variant, "ranked_doc_ids": unique,
                         "candidate_count": len(cand), "control_overlap": len(cand & baseline),
                         "novel_vs_control": len(cand - baseline),
                         "overlap_rate": len(cand & baseline) / max(len(baseline), 1),
                         "atom_count": len(views["atoms_vi"])})
        if len(rows) % 40 == 0 or qid == ids[-1]:
            a.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(a.output_dir / "candidates.partial.parquet", index=False)
            print(f"Evaluated {len(rows) // 4}/{len(ids)} queries", flush=True)
    conn.close()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_parquet(a.output_dir / "candidates.parquet", index=False)
    summary = out.groupby("variant").agg(
        queries=("query_id", "nunique"), mean_candidates=("candidate_count", "mean"),
        mean_control_overlap=("overlap_rate", "mean"), mean_novel=("novel_vs_control", "mean"),
    ).round(4).reset_index().to_dict("records")
    (a.output_dir / "audit.json").write_text(json.dumps({"k": a.k, "pilot_queries": len(ids), "summary": summary,
        "metric_limit": "Candidate novelty/overlap only; no labeled retrieval recall computed."}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
