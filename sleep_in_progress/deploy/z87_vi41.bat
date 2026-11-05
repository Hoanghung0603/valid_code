@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set R2AI_ROOT=D:\r2ai
cd /d D:\r2ai
if not exist data\sub\vi41 mkdir data\sub\vi41
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\vi41.py dense >> D:\r2ai\data\sub\vi41\dense.log 2>&1"
exit /b %errorlevel%
