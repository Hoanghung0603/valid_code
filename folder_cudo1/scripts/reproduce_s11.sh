#!/usr/bin/env bash
# Final assembly of the S11 submission (README section 4) from the shared intermediate data, CPU only, ~20 min.
#   bash scripts/reproduce_s11.sh            # -> data/runs/sub-long-kproxy-nproxy-v1/sub-long-kproxy-nproxy-v1.zip
# Needs the data package A unpacked into data/ (README section 3). Steps:
#   1. merged Qwen3-Reranker score files from the per-run Kaggle outputs
#   2. rerank proxy fitted to the LB doc P/R of the 8 earlier submissions, then k docs per query
#   3. fused doc order + whole-doc chunks (N per query from the same proxy)
#   4. check against the submitted S11, pack (LZMA, 256 MB dictionary: ~3 GB free RAM)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
NAME=sub-long-kproxy-nproxy-v1
mkdir -p data/dev   # rerank_proxy.py writes its fit and k file here

python scripts/analysis/merge_scores.py
python scripts/analysis/merge_long_scores.py
python scripts/analysis/rerank_proxy.py fit --deep --long --no-loo
python scripts/analysis/rerank_proxy.py k --deep --long
python scripts/analysis/fused_sub.py --docs-from sub-vi-rrfwp-zh30docs-v1 --n-from sub-vi-rrfwp-zh30c25snap-v1 \
  --scores data/kaggle/rerank/qwen3-0.6b-all/scores.parquet data/kaggle/rerank/qwen3-4b-all-tw-long/scores.parquet \
  --weights 1 0.5 1.5 --k-file data/dev/rerank_proxy_deep_long_k.parquet --n-proxy --deep --twins --long \
  --extra-docs 'data/parsed/documents/*/part-gloryvu-20261023-zh.parquet' --name "$NAME"
python scripts/compare_submissions.py "data/runs/$NAME/submission.json" "data/reference/$NAME/submission.json"
python scripts/analysis/allocate_sub.py pack --name "$NAME" --dict-mb "${DICT_MB:-256}"
