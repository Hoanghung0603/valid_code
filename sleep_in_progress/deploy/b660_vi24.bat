@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai
if not exist data\sub\diag24 mkdir data\sub\diag24
echo start %DATE% %TIME% > data\sub\diag24\%1.log
start "" /low /wait cmd /c "D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi24.py %1 --machine b660 >> data\sub\diag24\%1.log 2>&1"
set RC=%errorlevel%
echo exit %RC% %DATE% %TIME% >> data\sub\diag24\%1.log
exit /b %RC%
