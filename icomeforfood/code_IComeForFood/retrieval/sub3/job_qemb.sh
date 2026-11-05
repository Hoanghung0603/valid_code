#!/bin/bash
# Qwen3-Embedding-0.6B doc-level retrieval over sub2 + sub3add + sub4add (qemb.py).
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4
PY=~/miniconda3/envs/autolabel_det/bin/python
date
$PY -u qemb.py Qwen/Qwen3-Embedding-0.6B runs/sub3/qemb06.pkl data/docs_sub2 data/docs_sub3add data/docs_sub4add
date
