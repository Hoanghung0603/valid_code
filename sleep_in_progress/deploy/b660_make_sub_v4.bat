@echo off
rem Bien the bai nop v4: chunk 600 tu x 3 chunk/cau (xu huong: 80 < 150 < 300 tu cang lon cang tot), 40 doc, lay chunk tu 10 doc dau.
rem Log: D:\r2ai\data\sub\make_sub_v4.log, ket qua: D:\r2ai\data\sub\submission_v4.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bigpipeline.py --index D:\r2ai\data\index\vi --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\sub\submission_v4.json --top-docs 40 --rel-ratio 0 --chunk-docs 10 --chunk-units 600 --max-chunks 3 > D:\r2ai\data\sub\make_sub_v4.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v4.log
