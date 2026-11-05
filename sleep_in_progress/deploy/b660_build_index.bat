@echo off
rem Dung chi muc BM25 tieng Viet tu shard trunglt198 (worker tu ha uu tien, 4 worker). Log: D:\r2ai\data\index\build_vi.log
cd /d D:\r2ai\scripts
set PYTHONIOENCODING=utf-8
if not exist D:\r2ai\data\index mkdir D:\r2ai\data\index
D:\r2ai\.venv\Scripts\python.exe bigindex.py build --shards D:\r2ai\data\corpus\trunglt198 --out D:\r2ai\data\index\vi --workers 4 > D:\r2ai\data\index\build_vi.log 2>&1
