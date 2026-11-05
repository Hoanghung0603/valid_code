"""Submission files from a rerank run: top docs by reranked chunk score, one or more s400 chunks per doc.

Reads the run's rerank.pkl (chunk indices into the p150 files, in --chunks order) and maps each chosen p150 chunk
to the s400 chunk of the same doc that overlaps it most, as `pipeline.py eval` does, without loading chunk texts
that are not submitted. Writes <out>/sub_d{N}_c{K}.json and a .zip holding just that file.

    python retrieval/make_submission.py --run D:/r2ai_data/runs/sub1 --chunks <p150 dirs...> --s400 <s400 dirs...>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ranked_chunks, rerank_file  # noqa: E402


def files_of(paths: list[Path]) -> list[Path]:
    return [f for p in paths for f in (sorted(p.glob("*.parquet")) if p.is_dir() else [p])]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, nargs="+", required=True, help="p150 folders, same order as the run")
    ap.add_argument("--s400", type=Path, nargs="+", required=True)
    ap.add_argument("--queries", type=Path, default=Path("D:/r2ai_data/raw/query.parquet"))
    ap.add_argument("--reranker", default="bge")
    ap.add_argument("--mode", default="max", help="score for chunks also scored with the translated query")
    ap.add_argument("--n-docs", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--per-doc", type=int, nargs="+", default=[1])
    ap.add_argument("--chunk-docs", type=int, help="attach chunks only to the first N docs (keeps the zip small)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.run / "submit"
    out.mkdir(parents=True, exist_ok=True)

    p150 = pq.read_table(files_of(args.chunks), columns=["doc_id", "char_start", "char_end"])
    doc_of = p150.column("doc_id").to_numpy()
    p_start, p_end = p150.column("char_start").to_numpy(), p150.column("char_end").to_numpy()
    rr = pickle.loads((args.run / rerank_file(args.reranker)).read_bytes())["scores"]
    qids = pq.read_table(args.queries, columns=["id"]).column("id").to_pylist()
    missing = [q for q in qids if q not in rr]
    assert not missing, f"{len(missing)} queries have no rerank scores, e.g. {missing[:5]}"
    order = {q: ranked_chunks(rr[q], args.mode) for q in qids}

    # s400 chunks of the docs that can be submitted
    need = {int(doc_of[i]) for o in order.values() for i in o}
    s400 = pq.read_table(files_of(args.s400), columns=["doc_id", "char_start", "char_end", "text"])
    s400 = s400.filter(pc.is_in(s400.column("doc_id"), value_set=pa.array(sorted(need), pa.int64())))
    by_doc: dict[int, list[tuple[int, int, str]]] = {}
    for d, a, b, t in zip(*(s400.column(c).to_pylist() for c in ("doc_id", "char_start", "char_end", "text"))):
        by_doc.setdefault(d, []).append((a, b, t))
    print(f"{len(qids)} queries, {len(need)} candidate docs, {s400.num_rows} s400 chunks loaded", flush=True)

    def s400_text(i: int) -> str | None:
        a, b = p_start[i], p_end[i]
        best = max(by_doc.get(int(doc_of[i]), []), key=lambda s: min(b, s[1]) - max(a, s[0]), default=None)
        return best[2] if best and min(b, best[1]) > max(a, best[0]) else None

    for n_docs in args.n_docs:
        for per_doc in args.per_doc:
            sub = []
            for q in qids:
                docs: list[int] = []
                taken: dict[int, int] = {}
                seen: set[str] = set()
                chunks = []
                for i in order[q]:
                    d = int(doc_of[i])
                    if d not in taken:
                        if len(docs) == n_docs:
                            continue
                        docs.append(d)
                        taken[d] = 0
                    if taken[d] >= per_doc or (args.chunk_docs and docs.index(d) >= args.chunk_docs):
                        continue
                    t = s400_text(i)
                    if t is None or t in seen:
                        continue
                    seen.add(t)
                    taken[d] += 1
                    chunks.append({"doc_id": d, "chunk_text": t})
                sub.append({"id": int(q), "relevant_docs": docs, "relevant_chunks": chunks})
            name = f"sub_d{n_docs}_c{per_doc}" + (f"_k{args.chunk_docs}" if args.chunk_docs else "")
            js = out / f"{name}.json"
            js.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
            with zipfile.ZipFile(out / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.write(js, arcname=js.name)
            n_d = np.mean([len(s["relevant_docs"]) for s in sub])
            n_c = np.mean([len(s["relevant_chunks"]) for s in sub])
            print(f"{name}: {n_d:.1f} docs, {n_c:.1f} chunks per query -> {out / name}.zip", flush=True)


if __name__ == "__main__":
    main()
