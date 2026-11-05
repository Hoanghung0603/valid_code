@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\vi41_offline_audit.py >> D:\r2ai\data\sub\vi41\offline_extra.log 2>&1
exit /b %errorlevel%
