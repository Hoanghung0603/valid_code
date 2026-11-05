@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi37.py qmerge --round 2 >> D:\r2ai\data\sub\vi37\vi37.log 2>&1
if errorlevel 1 exit /b 1
call D:\r2ai\b660_vi37.bat qselect
