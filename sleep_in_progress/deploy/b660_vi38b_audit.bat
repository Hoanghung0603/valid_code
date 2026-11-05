@echo off
cd /d D:\r2ai
set PYTHONIOENCODING=utf-8
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi38_after_build.py > D:\r2ai\data\sub\vi38\vi38b_audit.log 2>&1
exit /b %errorlevel%
