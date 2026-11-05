@echo off
rem v10: RRF(BM25 top-200, bge-m3 dense) -> 200 doc, chunk 600 tu x 15 chunk/cau tu 30 doc dau (cung cau hinh v9, chi doi thu hang).
rem Log: D:\r2ai\data\sub\make_sub_v10.log, ket qua: D:\r2ai\data\sub\submission_v10.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\fuse.py --bm25 D:\r2ai\data\sub\cache_vi_k300.json --dense D:\r2ai\data\sub\dense_top200.json --out D:\r2ai\data\sub\cache_vi_rrf200.json --n 200 --k 60 > D:\r2ai\data\sub\make_sub_v10.log 2>&1
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --cache-vi D:\r2ai\data\sub\cache_vi_rrf200.json --out D:\r2ai\data\sub\submission_v10.json --n-vi 200 --n-zh 0 --chunk-docs-vi 30 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 15 --chunks-zh 0 >> D:\r2ai\data\sub\make_sub_v10.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v10.log
