"""Measure concept-graph candidate coverage against Exp143, without qrels claims."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval.concept_graph import ConceptLexicon, retrieve  # noqa: E402
from scripts.build_final_sprint_exp142_missing_parents import read_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/reference/concept_graph_v1.sqlite")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "configs/medical_concepts_v1.json")
    parser.add_argument("--queries", type=Path, default=ROOT / "data/raw/btc_provided/query.parquet")
    parser.add_argument("--control", type=Path, default=ROOT / "submission_exp143_p2a_long_chunks_add3.zip")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    query_table = pq.read_table(args.queries, columns=["id", "query"])
    queries = {int(row["id"]): str(row["query"]) for row in query_table.to_pylist()}
    baseline = {int(row["id"]): set(map(int, row["relevant_docs"])) for row in read_rows(args.control)}
    if set(queries) != set(baseline) or len(queries) != 1200:
        raise RuntimeError("Official query/control ID mismatch")
    graph = sqlite3.connect(args.graph)
    lexicon = ConceptLexicon(args.lexicon)
    covered = total = novel = matched = 0
    per_query = []
    for qid, query in queries.items():
        candidates = retrieve(graph, lexicon, query, args.limit)
        ids = {int(item["doc_id"]) for item in candidates}
        covered += bool(ids)
        total += len(ids)
        novel += len(ids - baseline[qid])
        matched += len(ids & baseline[qid])
        per_query.append({"query_id": qid, "concepts": [m.concept_id for m in lexicon.mentions(query)],
                          "candidates": len(ids), "novel_vs_control": len(ids - baseline[qid]),
                          "overlap_with_control": len(ids & baseline[qid])})
    graph.close()
    report = {"queries": 1200, "queries_with_candidates": covered,
              "graph_candidates": total, "novel_vs_control": novel,
              "overlap_with_control": matched,
              "warning": "Structural coverage only; no BTC labels, no precision/recall or score estimate.",
              "rows": per_query}
    output = args.graph.with_suffix(".audit.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
