@echo off
rem Bien the bai nop v2: tra 40 doc/cau (bo rel_ratio) de tang Recall doc; chunk 80 tu, 10 chunk/cau, lay chunk tu 10 doc dau.
rem Log: D:\r2ai\data\sub\make_sub_v2.log, ket qua: D:\r2ai\data\sub\submission_v2.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bigpipeline.py --index D:\r2ai\data\index\vi --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\sub\submission_v2.json --top-docs 40 --rel-ratio 0 --chunk-docs 10 --chunk-units 80 --max-chunks 10 > D:\r2ai\data\sub\make_sub_v2.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v2.log
