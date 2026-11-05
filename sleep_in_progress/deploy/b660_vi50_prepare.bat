@echo off
cd /d D:\r2ai
set PYTHONIOENCODING=utf-8
if not exist D:\r2ai\data\sub\vi50 mkdir D:\r2ai\data\sub\vi50
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi50.py prepare --out D:\r2ai\data\sub\vi50 > D:\r2ai\data\sub\vi50\prepare.log 2>&1
exit /b %ERRORLEVEL%
