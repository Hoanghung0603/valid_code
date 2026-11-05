@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\dchunk.py audit --sample 2000 >> D:\r2ai\data\sub\dchunk\audit_sample.log 2>&1
exit /b %errorlevel%
