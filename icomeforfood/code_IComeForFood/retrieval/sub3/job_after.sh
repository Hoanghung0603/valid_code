#!/bin/bash
# After job_bm25w.sh: rerank (bge, up to 3 chunks/doc) the docs bm25w finds that have no rerank score yet.
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4
PY=~/miniconda3/envs/autolabel_det/bin/python
C="data/chunks_sub2/s400 data/chunks_sub3add/s400 data/chunks_sub4add/s400"
while ! grep -q BM25W_DONE job_bm25w.log 2>/dev/null; do sleep 60; done
date
set -e
$PY -u pool_bm25w.py 100 runs/sub3/pool_bm25w.pkl $C
$PY -u rerank_more.py runs/sub3/pool_bm25w.pkl runs/sub3/rerank_bm25w.pkl 3 $C
date
