@echo off
rem Bai nop v5: VI + ZH. 20 doc VI + 20 doc ZH (xen ke), chunk 600 tu/600 ky tu, 2 chunk VI + 2 chunk ZH moi cau.
rem Yeu cau: zh_queries.jsonl (dich xong). Log: D:\r2ai\data\sub\make_sub_v5.log, ket qua: D:\r2ai\data\sub\submission_v5.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bipipeline.py --index-vi D:\r2ai\data\index\vi --index-zh D:\r2ai\data\index\zh --queries D:\r2ai\data\query.parquet --zh-queries D:\r2ai\data\llm\zh_queries.jsonl --out D:\r2ai\data\sub\submission_v5.json --n-vi 20 --n-zh 20 --chunk-docs-vi 5 --chunk-docs-zh 5 --units-vi 600 --units-zh 600 --chunks-vi 2 --chunks-zh 2 > D:\r2ai\data\sub\make_sub_v5.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v5.log
