@echo off
rem Tao submission VI tu chi muc BigIndex. Log: D:\r2ai\data\sub\make_sub.log, ket qua: D:\r2ai\data\sub\submission.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\bigpipeline.py --index D:\r2ai\data\index\vi --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\sub\submission.json > D:\r2ai\data\sub\make_sub.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\make_sub.log
