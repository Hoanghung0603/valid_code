#!/bin/bash
# Phiên mới: chỉ chạy tìm kiếm ZH trên shard đã có trên Drive (không crawl), kết quả ghi vào Drive
cd /content/r2ai
OUT=/content/drive/MyDrive/r2ai
LOC=/content/docs_zh
mkdir -p $LOC $OUT/logs
pkill -f "src.crawl"; pkill -f zh_search_fast; sleep 1
pip -q install jieba psutil >/dev/null 2>&1
cp -n $OUT/docs_zh/corpus-*.parquet $LOC/ 2>/dev/null
echo "shards local: $(ls $LOC | wc -l), $(du -sh $LOC | cut -f1)"
setsid nohup python colab/zh_search_fast.py --docs $LOC --plans query_plans.jsonl --out $OUT/zh_preds.jsonl --k 100 > $OUT/logs/zh_search.log 2>&1 < /dev/null &
echo started
