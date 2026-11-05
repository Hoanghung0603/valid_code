@echo off
rem Dung chi muc BM25 tieng Trung tu shard trunglt198 (worker tu ha uu tien, 3 worker). Log: D:\r2ai\data\index\build_zh.log
cd /d D:\r2ai\scripts
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\r2ai\.venv\Scripts\python.exe bigindex.py build --shards D:\r2ai\data\corpus\trunglt198 --out D:\r2ai\data\index\zh --lang zh --workers 3 > D:\r2ai\data\index\build_zh.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\index\build_zh.log
