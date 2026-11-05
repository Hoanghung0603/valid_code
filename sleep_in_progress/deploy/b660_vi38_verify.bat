@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi38_verify_export.py >> D:\r2ai\data\sub\vi38\verify.log 2>&1
exit /b %errorlevel%
