# Submission 3 and private phase (28/10 – 02/11)

Scripts that produced the public best (FINAL 0.2922) and the private files. Paths inside are the ones used on the team
machine (`D:/r2ai_data`) and on the lab (`~/sontung_veryIMPORTANT/r2ai_tmp_sub2`, deleted 02/11); `retrieval/pipeline.py`
provides the encoders, rerankers and candidate generation they call.

## Pipeline

1. **Candidates** (`job_sub3.sh` → `pipeline.py cands`): BM25 and BGE-M3 dense per language over s400 chunks, plus the
   whole-doc BM25 list (`bm25doc.py`, `bm25list.py`, `merge_list.py`).
2. **Rerank** (`pipeline.py rerank`): bge-reranker-v2-m3 on a 300-doc pool per language; `rerank_more.py` scores up to 3
   more chunks of each top-200 doc (`make_pool.py`).
3. **More docs from doc-level embeddings**: Qwen3-Embedding-0.6B and -8B over title + start of every doc (`qemb.py`),
   docs they find without a score get bge-reranked (`pool_qemb*.py` → `rerank_more.py`). New VOV/CC docs:
   `newcands.py`.
4. **Re-order the top 300**: z(bge, zh/en −3) + z(aux), aux = z(topic centroid cosine, `prf.py`) + 2·z(Qwen3-8B cosine,
   `qemb_pool.py`) + z(Qwen3-Reranker-8B, `q8rr.py`), combined by `combine_aux2.py`.
5. **Submission** (`make_sub3.py`): per-query cut at bge score ≥ −2.5, chunk = best s400 chunk ±1400 chars for the top
   155–185 docs; `merge_combo.py` joins a doc variant with a chunk variant (the two levels are scored apart);
   `lzbig.py` zips with LZMA and a 256MB dictionary, queries ordered by shared docs (about a third smaller than bzip2).
   `reproduce_submission.sh` rebuilds the private file from the cached scores; `same_submission.py` compares two files.

