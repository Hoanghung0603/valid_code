@echo off
rem Bien the bai nop v3: chunk lon hon (300 tu, 4 chunk/cau) de thu gia thuyet "co chunk" (v2 chunk 80 tu x10 kem hon 150 x5).
rem Log: D:\r2ai\data\sub\make_sub_v3.log, ket qua: D:\r2ai\data\sub\submission_v3.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bigpipeline.py --index D:\r2ai\data\index\vi --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\sub\submission_v3.json --top-docs 40 --rel-ratio 0 --chunk-docs 10 --chunk-units 300 --max-chunks 4 > D:\r2ai\data\sub\make_sub_v3.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub_v3.log
