@echo off
rem v6: chi VI, 60 doc, chunk 600 tu x 3 chunk/cau lay tu 10 doc dau (giong v4, chi tang so doc) -> do anh huong cua top-k doc.
rem v7: chi ZH, 40 doc, chunk 600 ky tu x 3 chunk/cau lay tu 10 doc dau -> do rieng chat luong nhanh ZH.
rem Chay lan luot (khong song song) de khong tranh dia. Log: D:\r2ai\data\sub\make_sub_v6.log, make_sub_v7.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --out D:\r2ai\data\sub\submission_v6.json --n-vi 60 --n-zh 0 --chunk-docs-vi 10 --chunk-docs-zh 0 --units-vi 600 --units-zh 600 --chunks-vi 3 --chunks-zh 0 > D:\r2ai\data\sub\make_sub_v6.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v6.log
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --out D:\r2ai\data\sub\submission_v7.json --n-vi 0 --n-zh 40 --chunk-docs-vi 0 --chunk-docs-zh 10 --units-vi 600 --units-zh 600 --chunks-vi 0 --chunks-zh 3 > D:\r2ai\data\sub\make_sub_v7.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v7.log
