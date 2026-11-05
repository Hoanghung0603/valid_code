@echo off
rem v15: chunks from top-30 docs of cache_pool_rr (rerank logit order on BM25 top-200 U dense-full top-200).
rem 15 chunks/query x 600 units, same config as v12/v13 chunking -> chunks_pool_rr.json.
rem Log: D:\r2ai\data\sub\make_sub_v15.log. Lock heavy_job.lock do coordinator quan ly qua SSH (truoc/sau chay).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --cache-vi D:\r2ai\data\sub\cache_pool_rr.json --out D:\r2ai\data\sub\chunks_pool_rr.json --n-vi 30 --n-zh 0 --chunk-docs-vi 30 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 15 --chunks-zh 0 > D:\r2ai\data\sub\make_sub_v15.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v15.log
