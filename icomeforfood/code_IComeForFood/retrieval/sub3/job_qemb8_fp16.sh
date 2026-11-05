#!/bin/bash
# Qwen3-Embedding-8B doc-level retrieval over all docs, then bge-rerank (2 chunks/doc) the docs it finds that have no score.
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4 BATCH=64 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/miniconda3/envs/autolabel_det/bin/python
C="data/chunks_sub2/s400 data/chunks_sub3add/s400 data/chunks_sub4add/s400 data/chunks_sub5add/s400"
date
$PY -u qemb.py Qwen/Qwen3-Embedding-8B runs/sub3/qemb8.pkl data/docs_sub2 data/docs_sub3add data/docs_sub4add data/docs_sub5add
date
$PY -u pool_qemb8.py 200 runs/sub3/pool_qemb8.pkl $C
$PY -u rerank_more.py runs/sub3/pool_qemb8.pkl runs/sub3/rerank_qemb8.pkl 2 $C
date; echo QEMB8_ALL_DONE
