"""Corpus scope: the parsed docs that belong to the current corpus file (config dataset.corpus_file).

Corpus v2 (2026-10-12) keeps every Vietnamese URL, only 240k of the 3.39M Chinese ones and adds 240k PubMed docs;
ids are unchanged. Docs outside the scope can never be gold, so indexes are built on the scope only and the
retriever filters by it as well. A parsed doc is in scope when its own id or any URL alias of it is in the corpus
(aliases: corpus id -> fetched doc id, from the crawl manifest).
  <index_dir>/scope_docs.npy     sorted doc ids in scope
  <index_dir>/scope_inputs.npy   sorted embedding input ids used by at least one in-scope chunk
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCOPE_DOCS, SCOPE_INPUTS = "scope_docs.npy", "scope_inputs.npy"


def build_scope(index_dir: Path, corpus: pd.DataFrame, log=print) -> tuple[np.ndarray, np.ndarray]:
    from medret.index.meta import DB_FILE, MetaStore

    meta = MetaStore(index_dir / DB_FILE)
    meta.con.register("_corpus", corpus[["id"]])
    docs = meta.con.execute("""
        SELECT doc_id FROM documents WHERE doc_id IN (SELECT id FROM _corpus)
        UNION SELECT fetched_id FROM aliases WHERE id IN (SELECT id FROM _corpus) AND fetched_id IS NOT NULL
    """).df()["doc_id"]
    docs = np.sort(docs.to_numpy(dtype=np.int64))
    meta.con.register("_docs", pd.DataFrame({"doc_id": docs}))
    inputs = meta.con.execute("""SELECT DISTINCT input_id FROM chunks
                                 WHERE doc_id IN (SELECT doc_id FROM _docs) AND input_id IS NOT NULL""").df()
    inputs = np.sort(inputs["input_id"].to_numpy(dtype=np.int64))
    by_lang = meta.con.execute("SELECT lang, count(*) n FROM documents WHERE doc_id IN (SELECT doc_id FROM _docs) "
                               "GROUP BY lang").df().set_index("lang")["n"].to_dict()
    np.save(index_dir / SCOPE_DOCS, docs)
    np.save(index_dir / SCOPE_INPUTS, inputs)
    log(f"scope: {len(docs):,} docs {by_lang}, {len(inputs):,} embedding inputs")
    return docs, inputs


def load_scope(index_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not (index_dir / SCOPE_DOCS).exists():
        return None
    return np.load(index_dir / SCOPE_DOCS), np.load(index_dir / SCOPE_INPUTS)
