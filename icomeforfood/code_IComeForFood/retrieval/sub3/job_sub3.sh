#!/bin/bash
# Submission 3 (98.8% corpus): encode only the chunks sub3 adds, candidates over sub2 + sub3add, add the whole-doc BM25 list
# (vi_bm25doc, uploaded as runs/sub3/bm25doc_list.pkl) to the Vietnamese pool, bge rerank with a 300-doc pool per language.
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib HF_HOME=$W/hf TMPDIR=$W/tmp HF_HUB_DISABLE_TELEMETRY=1
PY=~/miniconda3/envs/autolabel_det/bin/python
D=$W/data
P="$PY -u code/retrieval/pipeline.py"
date
[ -f $D/index_sub3add/s400/dense.npy ] || $P encode --chunks $D/chunks_sub3add/s400 --index $D/index_sub3add/s400 --batch 64 --max-length 512
ARGS="--chunks $D/chunks_sub2/s400 $D/chunks_sub3add/s400 --index $D/index_sub2/s400 $D/index_sub3add/s400 --devset $D/devset/real --translations $D/qtrans/v1/translations.jsonl --split dev"
mkdir -p runs/sub3
[ -f runs/sub3/cands.pkl ] || $P cands --out $W/runs/sub3 $ARGS --bm25-max-tokens 300
echo "== waiting for bm25doc_list.pkl $(date)"
while [ ! -f runs/sub3/bm25doc_list.pkl ]; do sleep 60; done
[ -f runs/sub3/MERGED ] || { $PY merge_list.py runs/sub3/cands.pkl runs/sub3/bm25doc_list.pkl && touch runs/sub3/MERGED; }
[ -f runs/sub3/rerank.pkl ] || $P rerank --out $W/runs/sub3 $ARGS --n-docs-pool 300 --per-doc-pool 1 --batch 128
date; echo SUB3_DONE
