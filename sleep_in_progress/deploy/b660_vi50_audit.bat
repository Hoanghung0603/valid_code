@echo off
cd /d D:\r2ai
set PYTHONIOENCODING=utf-8
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi50_prepare_audit.py > D:\r2ai\data\sub\vi50\independent_prepare_audit.log 2>&1
exit /b %ERRORLEVEL%
