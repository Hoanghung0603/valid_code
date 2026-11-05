#!/bin/bash
# Word-segmented whole-doc BM25 (vi pyvi words, vi syllable bigrams, en, zh with translated queries), see bm25w.py.
set -e
W=~/sontung_veryIMPORTANT/r2ai_tmp_sub2; cd $W
export PYTHONNOUSERSITE=1 PYTHONPATH=$W/pylib TMPDIR=$W/tmp NPROC=16 OMP_NUM_THREADS=4
PY=~/miniconda3/envs/autolabel_det/bin/python
date
$PY -u bm25w.py seg data/seg data/docs_sub2 data/docs_sub3add data/docs_sub4add
date
$PY -u bm25w.py score data/seg data/qtrans/v1/translations.jsonl data/raw/query.parquet runs/sub3/bm25w.pkl
date
