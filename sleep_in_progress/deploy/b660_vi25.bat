@echo off
rem vi25 buoc %1 (b660, CPU): mo rong vi24 bang goi VI 20261023 (43.154 doc).
rem Log: D:\r2ai\data\sub\diag25\%1.log. Lock heavy_job.lock do vi25.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai
if not exist data\sub\diag25 mkdir data\sub\diag25
echo start %DATE% %TIME% > data\sub\diag25\%1.log
start "" /low /wait cmd /c "D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi25.py %1 --machine b660 >> data\sub\diag25\%1.log 2>&1"
set RC=%errorlevel%
echo exit %RC% %DATE% %TIME% >> data\sub\diag25\%1.log
exit /b %RC%
