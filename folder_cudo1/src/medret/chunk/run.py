"""Corpus-scale chunking: documents/<site>/<part>.parquet -> chunks/<site>/<part>.parquet (resumable).

Chunk columns follow docs/design/architecture.md section 3, plus embed_sha1: the hash of the BGE-M3 input
(`medret.index.embed_input.passage_input`), so identical inputs are embedded once and re-chunking only
re-embeds what changed.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from medret.chunk.packer import ChunkConfig, chunk_document, hf_counter
from medret.index.embed_input import passage_input

CHUNK_ID_STRIDE = 10_000  # chunk_id = doc_id * stride + index within the doc

CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.int64()),
    ("doc_id", pa.int64()),
    ("lang", pa.string()),
    ("start", pa.int32()),
    ("end", pa.int32()),
    ("heading_path", pa.string()),
    ("n_tokens", pa.int32()),
    ("chunker_version", pa.string()),
    ("embed_sha1", pa.string()),
])


def _chunk_file(src: str, dst: str, tokenizer: str) -> tuple[int, int]:
    count = hf_counter(tokenizer)
    cfg = ChunkConfig()
    t = pq.read_table(src, columns=["doc_id", "lang", "title", "text", "blocks"]).to_pylist()
    recs = []
    for d in t:
        units = chunk_document(d["text"], d["blocks"], count, cfg)
        if len(units) >= CHUNK_ID_STRIDE:
            raise ValueError(f"doc {d['doc_id']} has {len(units)} chunks, more than CHUNK_ID_STRIDE")
        for i, u in enumerate(units):
            body = d["text"][u.start : u.end]
            emb = passage_input(d["title"], u.heading_path, body)
            recs.append({
                "chunk_id": d["doc_id"] * CHUNK_ID_STRIDE + i, "doc_id": d["doc_id"], "lang": d["lang"],
                "start": u.start, "end": u.end, "heading_path": u.heading_path, "n_tokens": u.n_tokens,
                "chunker_version": cfg.version, "embed_sha1": hashlib.sha1(emb.encode("utf-8")).hexdigest(),
            })
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(recs, schema=CHUNK_SCHEMA), tmp, compression="zstd")
    tmp.replace(out)
    return len(t), len(recs)


def chunk_corpus(docs_dir: Path, chunks_dir: Path, tokenizer: str = "BAAI/bge-m3", workers: int = 8,
                 log=print) -> None:
    jobs = [(src, chunks_dir / src.relative_to(docs_dir)) for src in sorted(docs_dir.glob("*/*.parquet"))]
    jobs = [(s, d) for s, d in jobs if not d.exists()]
    log(f"{len(jobs)} document files to chunk")
    t0, done, n_docs, n_chunks = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_chunk_file, str(s), str(d), tokenizer) for s, d in jobs]
        for f in as_completed(futs):
            a, b = f.result()
            done, n_docs, n_chunks = done + 1, n_docs + a, n_chunks + b
            if done % 100 == 0 or done == len(jobs):
                log(f"{done}/{len(jobs)} files, {n_docs:,} docs, {n_chunks:,} chunks, "
                    f"{n_docs / (time.time() - t0):,.0f} docs/s")
