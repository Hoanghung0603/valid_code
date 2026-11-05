@echo off
cd /d D:\r2ai
set PYTHONIOENCODING=utf-8
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi56.py prepare --out D:\r2ai\data\sub\vi56 > D:\r2ai\data\sub\vi56\prepare.log 2>&1
exit /b %ERRORLEVEL%
