#!/bin/bash
# After job_qemb.sh: bge-rerank (2 chunks/doc) the docs Qwen3-Embedding finds that have no score yet, then score the
# best chunk of the top 400 docs per query with AITeamVN/Vietnamese_Reranker.
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4
PY=~/miniconda3/envs/autolabel_det/bin/python
C="data/chunks_sub2/s400 data/chunks_sub3add/s400 data/chunks_sub4add/s400"
while ! grep -q QEMB_DONE job_qemb.log 2>/dev/null; do grep -qE "Traceback|Killed" job_qemb.log && exit 1; sleep 60; done
date
set -e
$PY -u pool_qemb.py 200 runs/sub3/pool_qemb.pkl $C
$PY -u rerank_more.py runs/sub3/pool_qemb.pkl runs/sub3/rerank_qemb.pkl 2 $C
date
$PY -u vnrr.py runs/sub3/pool_best400.pkl runs/sub3/vnrr400.pkl AITeamVN/Vietnamese_Reranker $C
date
