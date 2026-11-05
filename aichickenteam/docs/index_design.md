# Index design for the 4.4M-page VI/ZH corpus

Code: `src/chunk.py`, `src/index.py`, `src/embed.py`, `src/retrieve.py`, `src/rerank.py`. Tests: `tests/test_retrieval.py`.
Config: the `chunk:`, `index:`, `retrieve:` and `rerank:` blocks in `configs/default.yaml`.

## 1. Sizing (from links_corpus + the first crawled shard)

| | VI | ZH | total |
|---|---|---|---|
| URLs (by domain) | ~0.735M | ~3.66M (cnkang 0.96M, 120ask 0.92M, familydoctor 0.45M, ask.39 0.34M ...) | 4.39M |
| mean text chars/doc (data/docs/corpus-00000, 435 docs) | ~3,500 | ~920 | |
| corpus chars (if fully crawled) | ~2.6B | ~3.4B | **~6B** |
| mean chunk chars (default cfg) | 403 | 189 | |
| chunks | ~6.4M | ~18M | **~24M** |

## 2. Measured throughput (RTX 3050 8GB, shared with other jobs; fp16; 16 CPU)

| step | measured | notes |
|---|---|---|
| BGE-M3 embedding | **11k–17k tokens/s** | GPU is compute-bound (≈1.1 GFLOP/token). Peak VRAM 1.3–1.5 GB with `batch_tokens` 8–16k |
| ↳ ZH chunk ~300 chars (≈220 tok) | 50–80 chunks/s | |
| ↳ VI real chunks (avg 403 chars) | 85 chunks/s | crawled shard |
| ↳ doc "title + 150-char lead" (≈100 tok) | 87–130 docs/s | |
| ↳ title only (≈18 tok) | ~610/s | |
| bge-reranker-v2-m3, max_length 512 | **46–75 pairs/s** | 46 for ZH 300-char chunks, 65–75 for VI. Peak VRAM ~1.2 GB at 8k tokens/batch |
| chunking (8 procs) | 30k–55k docs/s | regex only |
| BM25 build (tokenize + postings, 8 procs) | ZH 11.5k chunks/s (jieba), VI 22.5k chunks/s | |
| BM25 postings | 0.33 postings/char of text; 3 B/posting (+ per-shard vocab) | ≈1.8–2.1 B/char on small shards, ~1.3 B/char expected at full scale |
| chunk parquet (zstd, with text) | 0.72 B/char | doc parquet is 0.54 B/char |
| FAISS IVF4096,PQ64 | train on 200k: 125 s; add 8.7k vec/s; search 0.3 ms/query (nprobe 64) | ~72 B/vector at scale (+16 MB centroids) |

## 3. Why full chunk-level dense does not fit

24M chunks is about 3.8B tokens. At 11–17k tok/s that is **62–96 GPU-hours**, and the fp16 vectors would take 47 GB. Even IVF-PQ storage (1.7 GB) cannot fix the compute cost. VI-only chunk-level dense (6.4M chunks, about 1.8B tokens) would still take 30–45 h.

## 4. Chosen design (2-level, bounded disk)

```
docs/*.parquet ──chunk_doc──► data/chunks/<file>-eNNNNN.parquet  (append-only manifest, global gids)
      │                          │
      │                          ├─► BM25 per language, CHUNK level, all chunks
      │                          │     hashed vocab (crc32, 2^24), shards ≤65k chunks:
      │                          │     terms uint32 | ptr int64 | rows uint16 | vals uint8 (quantised tf-part)
      │                          │     global IDF = Σ df over shards (computed at load), memmap'd
      │                          │
      └─title+lead(150 chars)──► dense DOC level (BGE-M3), fp16 .npy per file → IVF-PQ once ≥300k
                                  (fp16 files deleted after being added: disk stays bounded)
query: BM25(vi|zh|en rewrites) + dense(query_vi + all rewrites, max-sim over query vectors)
       doc-level hits (top 30 docs + top 10 BM25 docs) ─► their first ≤8 chunks embedded on the fly
       (sqlite cache) ─► chunk ranking ─► weighted RRF ─► bge-reranker-v2-m3 (sigmoid scores)
PubMed / extra docs per query: chunk + in-memory BM25 + embed (cached) ─► same RRF lists
```

* **Cost:** doc-level dense needs 4.4M × ~110 tok ≈ 0.5B tokens, so **≈9–13 GPU-hours**, resumable per file. Storage ≈ 0.35 GB (IVF-PQ) plus ≤0.6 GB of transient fp16. The chunk-level path (`index.dense.unit: {vi: chunk}`) is still supported, for the fixture, small subsets, or when spare GPU time exists. Units are set per language.
* **BM25 disk:** roughly 1.3–2 B/char, so **~8–12 GB** for 6B chars. Chunk parquet adds ~4.3 GB and docs ~3.2 GB. If disk is short, these steps keep it bounded:
  1. Set `chunk.max_chunks_per_doc` to 12. This caps long VI pages to about their first 4,800 chars, and answers on medical Q&A pages come first anyway.
  2. Set `bm25.vi_bigrams: false`. This removes about 45% of VI postings, at some loss of phrase precision.
  3. Delete `data/docs` after chunking and doc-level embedding. Chunks hold the verbatim text, so a submission only needs chunk text and doc_id.
* **BM25 RAM:** all files are memmapped. Resident memory is only the per-shard vocabulary plus the global idf (64 MB/lang). Building one shard needs about 0.3 GB.
* **Query cost (estimated, full scale):** BM25 runs over ~370 shards in about 0.1–0.3 s. IVF search takes milliseconds. Embedding expanded chunks (≤320) takes ≈3–4 s uncached. Reranking 200 candidates takes ≈3–4 s. **A first full run of 1,200 queries takes ≈2.5–3 h**, and later runs mostly hit the sqlite caches. `retrieve.expand_mode: inherit` skips the on-the-fly embedding.
* **Full build timeline:** chunking takes minutes. BM25 takes ~0.5–1 h on CPU and can run while the GPU does dense. Doc-level dense takes 9–13 h on GPU. IVF train takes ~2 min.

## 5. Incrementality & robustness

* `python -m src.index chunk|bm25|dense|all` only processes doc files it has not seen. A file whose size or mtime changed is tombstoned and re-added. Global ids never shift: chunk gid = entry.chunk_offset + row. The crawler may write a doc in several shards, so duplicates (same source+doc_id) are dropped at chunk time.
* BM25: the trailing shard stays open and is rebuilt when new files arrive. Full shards are sealed. IDF is global, and only `avgdl` is per shard (scores agree within ~1–3% with a single-shard build; see the test).
* Dense: one fp16 file per (doc file, unit, lang). `build_dense` can be stopped and resumed. Once ≥ `ivf.min_vectors` vectors exist, an IVF-PQ index is trained, and every 20 files new vectors are added to it and their fp16 files deleted.
* Embedding and reranking shrink the batch on CUDA OOM, and the smaller size sticks because the GPU is shared. Caches: `data/cache/embed.sqlite` and `data/cache/rerank.sqlite`.

## 6. Fixture results (tests/fixtures: 27 topics × 4 docs × {vi, zh, en})

Setup: 26 real ViBioMIR queries, **query_vi only (no LLM rewrites)**. VI/ZH are on disk; the 108 EN abstracts are passed as `extra_docs`.

| metric | value |
|---|---|
| fused list, topic-precision@5 among VI / ZH / EN chunks | 0.81 / 0.84 / 0.74 |
| ZH doc-level dense alone (cross-lingual VI→ZH), P@5 | 0.84 |
| after rerank: top-1 on topic | 1.00 |
| after rerank: P@5 | 0.93 |
| languages with on-topic chunks in reranked top-10 | 2.3 of 3 |
| retrieve latency | 33–100 ms/query |

With rewrites (test plans, 6 queries), the right topic is in the top 5 for every language and every query.

## 7. Open issues / next steps

* The thresholds and RRF weights are untuned until a dev set exists (see `retrieve.weights`, `select:`).
* Qwen3-Embedding-0.6B (MRL, 32–1024 dims) could halve dense cost through shorter sequences and low-dim int8 storage. `src/embed.py` supports it, but it is not benchmarked yet. The Qwen3-Reranker is not implemented.
* BGE-M3 sparse/colbert heads are unused. They could re-score candidates instead of the BM25 bigrams.
* Hashed vocabulary: with ~5M distinct jieba/VI-bigram types in 2^24 buckets, roughly a quarter of the (mostly rare) types share a bucket with another type. Frequent terms are rarely affected, and the cost is a slightly inflated df for some rare terms. Raise `hash_bits` to 26 (idf array 256 MB) if this matters.
