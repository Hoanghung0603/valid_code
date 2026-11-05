#!/bin/bash
# New docs (CC 1029 + VOV 1030): encode their chunks, top 30 per query by dense similarity, bge-rerank 2 chunks/doc.
while ! grep -q VNRR_DONE job_after_qemb.log 2>/dev/null; do sleep 60; done
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1 OMP_NUM_THREADS=4
PY=~/miniconda3/envs/autolabel_det/bin/python
P="$PY -u code/retrieval/pipeline.py"
for s in sub4add sub5add; do
  [ -f data/index_$s/s400/dense.npy ] || $P encode --chunks data/chunks_$s/s400 --index data/index_$s/s400 --batch 64 --max-length 512
done
$PY -u newcands.py 30 runs/sub3/pool_new.pkl data/chunks_sub4add/s400 data/chunks_sub5add/s400 -- data/index_sub4add/s400 data/index_sub5add/s400
$PY -u rerank_more.py runs/sub3/pool_new.pkl runs/sub3/rerank_new.pkl 2 data/chunks_sub2/s400 data/chunks_sub3add/s400 data/chunks_sub4add/s400 data/chunks_sub5add/s400
date; echo VOV_DONE
