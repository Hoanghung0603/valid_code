@echo off
rem vi34 buoc %1 (z87, GPU): docwin cho cap (cau, doc) moi cua doclist vi34.
rem Log: D:\r2ai\work\vi34\%1.log. Lock gpu.lock do vi34.py quan ly; do tre vinmec ghi trong log.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai
if not exist work\vi34 mkdir work\vi34
echo start %DATE% %TIME% > work\vi34\%1.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\vi34.py %1 --machine z87 >> work\vi34\%1.log 2>&1"
set RC=%errorlevel%
echo exit %RC% %DATE% %TIME% >> work\vi34\%1.log
exit /b %RC%
