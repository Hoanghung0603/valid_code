@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\vi41 mkdir D:\r2ai\data\sub\vi41
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi41.py %1 >> D:\r2ai\data\sub\vi41\%1.log 2>&1
exit /b %errorlevel%
