@echo off
rem vi25 buoc %1 (z87, GPU bge-m3 + docwin): embed/rerank/docwin cho doc moi.
rem Log: D:\r2ai\work\vi25\%1.log. Lock gpu.lock do vi25.py quan ly. Do do tre vinmec trong log.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai
if not exist work\vi25 mkdir work\vi25
echo start %DATE% %TIME% > work\vi25\%1.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\vi25.py %1 --machine z87 >> work\vi25\%1.log 2>&1"
set RC=%errorlevel%
echo exit %RC% %DATE% %TIME% >> work\vi25\%1.log
exit /b %RC%
