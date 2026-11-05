@echo off
rem Kiem tra chi muc VI: toc do truy van + tu nhat quan. Ket qua: D:\r2ai\data\index\check_vi.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\check_index.py --index D:\r2ai\data\index\vi --queries D:\r2ai\data\query.parquet > D:\r2ai\data\index\check_vi.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\index\check_vi.log
