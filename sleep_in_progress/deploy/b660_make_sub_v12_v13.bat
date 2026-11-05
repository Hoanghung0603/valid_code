@echo off
rem v12: rerank-only order (bge-reranker-v2-m3 logit) on 200 BM25 candidates -> cache_vi_rr200.json.
rem v13: RRF 3-way (BM25 + dense + rerank, k=60) -> cache_vi_rrf3_200.json.
rem Both: 60 docs, chunk 600 units x 15 chunks/query from first 30 docs (same config as v11).
rem Log: D:\r2ai\data\sub\make_sub_v12.log, make_sub_v13.log; out: submission_v12.json, submission_v13.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\locks mkdir D:\r2ai\locks
echo coordinator make_sub_v12_v13 %date% %time%> D:\r2ai\locks\heavy_job.lock
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\fuse.py --mode rerank --bm25 D:\r2ai\data\sub\cache_vi_k300.json --rerank D:\r2ai\data\sub\rerank_top200.json --n 200 --out D:\r2ai\data\sub\cache_vi_rr200.json > D:\r2ai\data\sub\make_sub_v12.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v12.log
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --cache-vi D:\r2ai\data\sub\cache_vi_rr200.json --out D:\r2ai\data\sub\submission_v12.json --n-vi 60 --n-zh 0 --chunk-docs-vi 30 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 15 --chunks-zh 0 >> D:\r2ai\data\sub\make_sub_v12.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v12.log
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\fuse.py --mode rrf --bm25 D:\r2ai\data\sub\cache_vi_k300.json --dense D:\r2ai\data\sub\dense_top200.json --rerank D:\r2ai\data\sub\rerank_top200.json --n 200 --k 60 --out D:\r2ai\data\sub\cache_vi_rrf3_200.json > D:\r2ai\data\sub\make_sub_v13.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v13.log
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --cache-vi D:\r2ai\data\sub\cache_vi_rrf3_200.json --out D:\r2ai\data\sub\submission_v13.json --n-vi 60 --n-zh 0 --chunk-docs-vi 30 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 15 --chunks-zh 0 >> D:\r2ai\data\sub\make_sub_v13.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v13.log
del D:\r2ai\locks\heavy_job.lock
