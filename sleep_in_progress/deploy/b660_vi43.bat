@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi43.py >> D:\r2ai\data\sub\vi43\vi43.log 2>&1
if errorlevel 1 exit /b 1
exit /b 0
