@echo off
rem Kiem tra chi muc ZH (chi tu nhat quan, khong co truy van ZH that). Ket qua: D:\r2ai\data\index\check_zh.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\check_index.py --index D:\r2ai\data\index\zh > D:\r2ai\data\index\check_zh.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\index\check_zh.log
