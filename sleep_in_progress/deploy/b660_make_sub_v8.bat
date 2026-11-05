@echo off
rem v8: chi VI. 150 doc (v6 dung 60: recall tang 0,126 -> 0,159 khi 40 -> 60), chunk 600 tu x 8 chunk/cau lay tu 20 doc dau (v6: 3 chunk tu 10 doc).
rem Doi hai bien nhung chung anh huong hai cot diem khac nhau: Docs F2 <- so doc; Chunks F2 <- so chunk/doc nguon.
rem Log: D:\r2ai\data\sub\make_sub_v8.log, ket qua: D:\r2ai\data\sub\submission_v8.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --out D:\r2ai\data\sub\submission_v8.json --n-vi 150 --n-zh 0 --chunk-docs-vi 20 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 8 --chunks-zh 0 > D:\r2ai\data\sub\make_sub_v8.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v8.log
