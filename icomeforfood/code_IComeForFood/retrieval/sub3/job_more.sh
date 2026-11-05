#!/bin/bash
# Rerank up to 3 more chunks of each of the top 200 sub3 docs per query (rerank_more.py).
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=2
PY=~/miniconda3/envs/autolabel_det/bin/python
date
$PY -u rerank_more.py runs/sub3/pool_more.pkl runs/sub3/rerank_more.pkl 3 data/chunks_sub2/s400 data/chunks_sub3add/s400
date
