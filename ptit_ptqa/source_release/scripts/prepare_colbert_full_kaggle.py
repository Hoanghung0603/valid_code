"""Package a full 1,200-query late-interaction run, pending pilot go/no-go."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

try:
    from scripts.prepare_colbert_pilot_kaggle import ROOT, SOURCE, QUERY, make_notebook, sha
except ModuleNotFoundError:
    from prepare_colbert_pilot_kaggle import ROOT, SOURCE, QUERY, make_notebook, sha

OUT = ROOT / "data/kaggle_inputs/colbert_full_20261103"
KERNEL = ROOT / "data/kaggle_inputs/colbert_full_kernel"
NOTEBOOK = "25_bgem3_colbert_full_kaggle.ipynb"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    KERNEL.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT / "colbert_full_pairs.parquet"
    manifest_path = OUT / "manifest.json"
    if pairs_path.exists() or manifest_path.exists():
        raise FileExistsError("Full input already exists; refusing overwrite")
    queries = pd.read_parquet(QUERY, columns=["id", "query"])
    if len(queries) != 1200 or queries.id.duplicated().any():
        raise RuntimeError("Official query coverage failed")
    source = pd.read_parquet(SOURCE, columns=["pair_id", "query_id", "doc_id", "passage_rank",
                                             "chunk_ordinal", "query_text", "passage_text"])
    source.sort_values(["query_id", "chunk_ordinal", "passage_rank", "pair_id"], inplace=True)
    source["doc_first_chunk_rank"] = source.groupby(["query_id", "doc_id"], sort=False)["chunk_ordinal"].transform("min")
    first_docs = (source[["query_id", "doc_id", "doc_first_chunk_rank"]]
                  .drop_duplicates(["query_id", "doc_id"])
                  .sort_values(["query_id", "doc_first_chunk_rank", "doc_id"]))
    first_docs["doc_pick_rank"] = first_docs.groupby("query_id").cumcount() + 1
    selected = first_docs[first_docs.doc_pick_rank <= 50][["query_id", "doc_id", "doc_pick_rank"]]
    full = source.merge(selected, on=["query_id", "doc_id"], how="inner", validate="many_to_one")
    full = full.groupby(["query_id", "doc_id"], sort=False).head(2).copy()
    full["passage_text_sha256"] = full.passage_text.map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    full.sort_values(["query_id", "doc_pick_rank", "passage_rank", "pair_id"], inplace=True)
    if set(full.query_id.astype(int)) != set(queries.id.astype(int)) or full.pair_id.duplicated().any():
        raise RuntimeError("Full input coverage or duplicate pair failure")
    full.to_parquet(pairs_path, index=False, compression="zstd")
    manifest = {"schema":"colbert_full_v1", "source":"P2a full exact source windows",
                "queries":1200, "docs_per_query_max":50, "passages_per_doc_max":2,
                "pairs":len(full), "unique_query_docs":len(selected),
                "source_sha256":sha(SOURCE), "query_sha256":sha(QUERY),
                "pairs_sha256":sha(pairs_path), "model":"BAAI/bge-m3",
                "model_revision":"5617a9f61b028005a4858fdac845db406aefb181",
                "max_query_tokens":256, "max_passage_tokens":384,
                "purpose":"full shortlist scores only; no relevance labels or BTC submission"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    notebook = make_notebook()
    notebook.cells[0].source = notebook.cells[0].source.replace("60 official queries", "1,200 official queries").replace(
        "throughput/stability pilot", "full shortlist scoring run, gated by pilot")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = (cell.source.replace("colbert_pilot_pairs.parquet", "colbert_full_pairs.parquet")
                           .replace("colbert_pilot_v1", "colbert_full_v1")
                           .replace("colbert_pilot_outputs.zip", "colbert_full_outputs.zip")
                           .replace("/colbert_pilot", "/colbert_full")
                           .replace("%100<8", "%1000<8"))
            if not cell.source.startswith("%pip"):
                compile(cell.source, "<colbert-full>", "exec")
    nbf.write(notebook, ROOT / "notebooks" / NOTEBOOK)
    nbf.write(notebook, KERNEL / NOTEBOOK)
    kernel_meta = {"id":"ptquanh/r2ai-bge-m3-colbert-full-20261103",
                   "title":"R2AI BGE-M3 ColBERT Full 20261103", "code_file":NOTEBOOK,
                   "language":"python", "kernel_type":"notebook", "is_private":"true",
                   "enable_gpu":"true", "enable_tpu":"false", "enable_internet":"true",
                   "machine_shape":"NvidiaTeslaT4",
                   "dataset_sources":["ptquanh/r2ai-bge-m3-colbert-full-input-20261103"],
                   "competition_sources":[], "kernel_sources":[], "model_sources":[]}
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")
    dataset_meta = {"title":"R2AI BGE-M3 ColBERT full input 20261103",
                    "id":"ptquanh/r2ai-bge-m3-colbert-full-input-20261103",
                    "licenses":[{"name":"other"}], "isPrivate":True}
    (OUT / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")
    print(json.dumps({"pairs":len(full), "queries":full.query_id.nunique(),
                      "query_docs":len(selected), "input_bytes":pairs_path.stat().st_size,
                      "input":str(OUT), "kernel":str(KERNEL)}, indent=2))


if __name__ == "__main__":
    main()
