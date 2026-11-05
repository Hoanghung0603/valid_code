"""Build an optional, provenance-bearing concept graph from official corpus text.

Pilot with --limit first. The graph is not a replacement for the audited
private submission and must not be submitted without a controlled comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval.concept_graph import ConceptLexicon, add_document, create_graph, retrieve  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(text_path: Path, corpus_path: Path, lexicon_path: Path, output: Path,
          limit: int | None = None) -> dict:
    lexicon = ConceptLexicon(lexicon_path)
    official = set()
    for batch in pq.ParquetFile(corpus_path).iter_batches(columns=["id"], batch_size=100_000):
        official.update(batch.column(0).to_pylist())
    db = create_graph(output, lexicon)
    scanned = indexed = edges = outside = 0
    try:
        for batch in pq.ParquetFile(text_path).iter_batches(columns=["id", "url", "text"], batch_size=2_000):
            for row in batch.to_pylist():
                if limit is not None and scanned >= limit:
                    break
                scanned += 1
                doc_id = int(row["id"])
                if doc_id not in official:
                    outside += 1
                    continue
                count = add_document(db, lexicon, doc_id, str(row["url"]), str(row["text"] or ""))
                indexed += bool(count)
                edges += count
            db.commit()
            if scanned % 100_000 < 2_000:
                print(json.dumps({"scanned": scanned, "indexed": indexed, "edges": edges}), flush=True)
            if limit is not None and scanned >= limit:
                break
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity check failed")
    finally:
        db.close()
    result = {"scanned": scanned, "indexed_docs": indexed, "mention_edges": edges,
              "out_of_official_corpus": outside, "concepts": len(lexicon.concepts),
              "lexicon_sha256": sha256(lexicon_path), "corpus_sha256": sha256(corpus_path),
              "text_sha256": sha256(text_path), "graph_sha256": sha256(output),
              "graph_bytes": output.stat().st_size, "limit": limit,
              "status": "pilot" if limit is not None else "full"}
    output.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, default=ROOT / "data/embedded_samples.parquet")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet")
    parser.add_argument("--lexicon", type=Path, default=ROOT / "configs/medical_concepts_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reference/concept_graph_v1.sqlite")
    parser.add_argument("--limit", type=int, help="Document pilot size; omit for all documents")
    parser.add_argument("--query", help="Inspect graph candidates for a query after building")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    print(json.dumps(build(args.text, args.corpus, args.lexicon, args.output, args.limit), indent=2))
    if args.query:
        db = sqlite3.connect(args.output)
        print(json.dumps(retrieve(db, ConceptLexicon(args.lexicon), args.query, 10), ensure_ascii=False, indent=2))
        db.close()


if __name__ == "__main__":
    main()
