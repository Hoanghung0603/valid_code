@echo off
rem vi30 (b660, CPU): vi29 + goi handoff_vi_20261025_full. Doi so: mot hoac nhieu stage
rem (prep sanity index exports bm25 pool doclist chunk build stats), chay lan luot, dung o stage loi.
rem Log: D:\r2ai\data\sub\diag30\vi30.log. Lock heavy_job.lock do vi30.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
if not exist D:\r2ai\data\sub\diag30 mkdir D:\r2ai\data\sub\diag30
set L=D:\r2ai\data\sub\diag30\vi30.log
echo start %DATE% %TIME% %* >> %L%
:loop
if "%1"=="" goto ok
echo === %1 %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi30.py %1 >> %L% 2>&1
if errorlevel 1 goto fail
shift
goto loop
:ok
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0
:fail
echo VI30 FAILED %1 >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
