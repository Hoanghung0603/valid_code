@echo off
cd /d D:\r2ai
set PYTHONIOENCODING=utf-8
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set OMP_NUM_THREADS=4
if not exist D:\r2ai\data\sub\vi40 mkdir D:\r2ai\data\sub\vi40
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi40.py build > D:\r2ai\data\sub\vi40\build.log 2>&1
exit /b %errorlevel%
