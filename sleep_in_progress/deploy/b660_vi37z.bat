@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\vi37z mkdir D:\r2ai\data\sub\vi37z
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi37z.py all >> D:\r2ai\data\sub\vi37z\vi37z.log 2>&1
exit /b %errorlevel%
