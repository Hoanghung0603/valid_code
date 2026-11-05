@echo off
cd /d D:\r2ai
set PYTHONIOENCODING=utf-8
if not exist D:\r2ai\data\sub\vi54 mkdir D:\r2ai\data\sub\vi54
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi54_prepare.py > D:\r2ai\data\sub\vi54\prepare.log 2>&1
exit /b %ERRORLEVEL%
