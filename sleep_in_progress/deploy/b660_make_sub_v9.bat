@echo off
rem v9: chi VI, dung cache top-300 (khong tim lai). 200 doc, chunk 600 tu x 15 chunk/cau lay tu 30 doc dau.
rem So voi v8 (150 doc, 8 chunk tu 20 doc): kiem tra Docs F2 quanh 200 doc va Chunks F2 khi tang so chunk.
rem Log: D:\r2ai\data\sub\make_sub_v9.log, ket qua: D:\r2ai\data\sub\submission_v9.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --cache-vi D:\r2ai\data\sub\cache_vi_k300.json --out D:\r2ai\data\sub\submission_v9.json --n-vi 200 --n-zh 0 --chunk-docs-vi 30 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 15 --chunks-zh 0 > D:\r2ai\data\sub\make_sub_v9.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v9.log
