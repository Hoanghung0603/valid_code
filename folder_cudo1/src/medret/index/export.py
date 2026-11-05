"""Export deduplicated BGE-M3 inputs for the GPU embedding runs (scripts/kaggle/embed_runner).

Each distinct `embed_sha1` is written once: data/embed_inputs/shard-NNNNN.parquet with
  input_id   int64, global and sequential across shards (the id the runner writes to `<shard>.ids.npy`)
  embed_sha1 string, sha1 of the input (chunks.embed_sha1 points here, so one vector serves every chunk
             with the same input)
  text       string, `passage_input(title, heading_path, chunk text)`; never submitted, chunk_text is
             always cut from documents.text
  n_tokens   int32, chunk tokens (the input adds ~8% for the title / heading prefix)
A rerun or a crawl increment reads the existing shards, skips hashes already exported and continues the
input_id sequence, so earlier shards (and their vectors) stay valid.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from medret.index.embed_input import passage_input

SHARD_GLOB = "shard-*.parquet"
EXPORT_SCHEMA = pa.schema([
    ("input_id", pa.int64()), ("embed_sha1", pa.string()), ("text", pa.string()), ("n_tokens", pa.int32()),
])


def export_embed_inputs(docs_dir: Path, chunks_dir: Path, out_dir: Path, rows_per_shard: int = 50_000,
                        log=print) -> dict:
    """Walk chunk files with their document files and write each unseen embed input once."""
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(SHARD_GLOB))
    seen: set[str] = set()
    next_id = 0
    for f in existing:
        t = pq.read_table(f, columns=["input_id", "embed_sha1"])
        seen.update(t.column("embed_sha1").to_pylist())
        if t.num_rows:
            next_id = max(next_id, int(pc.max(t.column("input_id")).as_py()) + 1)
    shard_no = len(existing)
    buf: list[dict] = []
    stats = {"chunks": 0, "unique": 0, "tokens_unique": 0, "shards": 0}

    def flush() -> None:
        nonlocal shard_no
        if buf:
            pq.write_table(pa.Table.from_pylist(buf, schema=EXPORT_SCHEMA),
                           out_dir / f"shard-{shard_no:05d}.parquet", compression="zstd")
            shard_no += 1
            stats["shards"] += 1
            buf.clear()

    for cf in sorted(chunks_dir.glob("*/*.parquet")):
        chunks = pq.read_table(cf, columns=["doc_id", "start", "end", "heading_path", "n_tokens", "embed_sha1"])
        need = [r for r in chunks.to_pylist() if r["embed_sha1"] not in seen]
        stats["chunks"] += chunks.num_rows
        if not need:
            continue
        docs = pq.read_table(docs_dir / cf.relative_to(chunks_dir), columns=["doc_id", "title", "text"]).to_pylist()
        by_id = {d["doc_id"]: d for d in docs}
        for r in need:
            if r["embed_sha1"] in seen:
                continue
            seen.add(r["embed_sha1"])
            d = by_id[r["doc_id"]]
            buf.append({"input_id": next_id, "embed_sha1": r["embed_sha1"], "n_tokens": r["n_tokens"],
                        "text": passage_input(d["title"], r["heading_path"], d["text"][r["start"] : r["end"]])})
            next_id += 1
            stats["unique"] += 1
            stats["tokens_unique"] += r["n_tokens"]
            if len(buf) >= rows_per_shard:
                flush()
    flush()
    log(f"{stats['chunks']:,} chunks, {stats['unique']:,} new unique inputs in {stats['shards']} shards, "
        f"~{stats['tokens_unique'] / 1e9:.2f}B chunk tokens")
    return stats
