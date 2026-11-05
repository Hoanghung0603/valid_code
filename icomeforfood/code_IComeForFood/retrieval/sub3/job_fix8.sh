#!/bin/bash
# Resume after the 8B embedding: bge-rerank the docs Qwen3-Embedding-8B found (2 chunks/doc), free the embedding model,
# pool top 300 docs per query, score their best chunk with Qwen3-Reranker-8B.
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/autolabel_det/bin/python
C="data/chunks_sub2/s400 data/chunks_sub3add/s400 data/chunks_sub4add/s400 data/chunks_sub5add/s400"
date
[ -f runs/sub3/rerank_qemb8.pkl ] || $PY -u rerank_more.py runs/sub3/pool_qemb8.pkl runs/sub3/rerank_qemb8.pkl 2 $C
date
if [ -d hf/hub/models--Qwen--Qwen3-Embedding-8B ]; then
  cd hf/hub && for f in $(find models--Qwen--Qwen3-Embedding-8B/snapshots -type l); do rm -f "$(readlink -f $f)"; done; rm -rf models--Qwen--Qwen3-Embedding-8B; cd $W
fi
df -h ~ | tail -1
$PY -u pool_best_lab.py 300 3 runs/sub3/pool_q8rr.pkl $C -- runs/sub3/rerank.pkl runs/sub3/rerank_more.pkl runs/sub3/rerank_qemb.pkl runs/sub3/rerank_new.pkl runs/sub3/rerank_qemb8.pkl
$PY -u q8rr.py runs/sub3/pool_q8rr.pkl runs/sub3/q8rr300.pkl Qwen/Qwen3-Reranker-8B $C
date; echo FIX8_DONE
