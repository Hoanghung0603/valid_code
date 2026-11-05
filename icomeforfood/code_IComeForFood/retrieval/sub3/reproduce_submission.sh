#!/bin/bash
# Rebuild the private submission (P2) from the cached model scores in $R2AI_DATA/runs/sub3: CPU only, about 30 minutes,
# 16GB RAM. Steps 3-4 of the pipeline (GPU) produced those caches; see README.md for running them from scratch.
#   R2AI_DATA=/path/to/r2ai_data PYTHON=python bash reproduce_submission.sh
# Private variants: P3  W_RR=2, P4  W_RR=0.5, P5  THR=-2, P6  W_Q8=3 W_RR=2, P7  THR=-3 (P2: W_Q8=2 W_RR=1 THR=-2.5).
set -e
cd "$(dirname "$0")"
DATA=${R2AI_DATA:-D:/r2ai_data}; PY=${PYTHON:-python}; D=$DATA/runs/sub3; O=${OUT:-$D/reproduce}
export R2AI_DATA=$DATA
mkdir -p "$O"
C="$DATA/chunks_sub2/s400 $DATA/chunks_sub3add/s400 $DATA/chunks_sub4add/s400 $DATA/chunks_sub5add/s400"
DOCS="$DATA/docs_sub2 $DATA/docs_sub3add $DATA/docs_sub4add $DATA/docs_sub5add"
$PY merge_q8cos.py $D/q8pool300.pkl $D/qemb8.pkl $O/q8cos_all.pkl
$PY rr2aux.py $D/q8rr300.pkl $O/q8rr_aux.pkl
$PY combine_aux2.py $O/aux.pkl $D/prf400q.pkl:20:1 $O/q8cos_all.pkl:0:${W_Q8:-2} $O/q8rr_aux.pkl:0:${W_RR:-1}
$PY make_sub3.py --run $D --queries $DATA/raw/query.parquet \
  --extra $D/rerank_more.pkl $D/rerank_qemb.pkl $D/rerank_new.pkl $D/rerank_qemb8.pkl \
  --chunks $C --docs $DOCS --bz2 --pen 3 --n-docs 400 --thr ${THR:--2.5} --prf $O/aux.pkl --prf-k 0 --prf-w 1 \
  --chunk-docs 185 --pad 1400 --out $O
ORDER=1 $PY lzbig.py $O/s3_p3_f0_prf0w1_d400_k185_w1_pad1400_t${THR:--2.5}_bz.zip $O/submission.zip 256
echo "-> $O/submission.zip"
