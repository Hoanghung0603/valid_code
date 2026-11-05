@echo off
rem vi36 (b660, CPU): vi34 + goi handoff_vi_topup_20261026 (1.793 doc). Doi so: mot hoac nhieu stage
rem (prep sanity index bm25 pool doclist_pre q8b_select q8b_merge doclist chunk build stats), chay
rem lan luot, dung o stage loi. Log: D:\r2ai\data\sub\vi36\vi36.log.
rem Lock heavy_job.lock do vi36.py quan ly per-stage (nhao giua cac stage).
rem Luu y: giua q8b_select va q8b_merge can chay Modal (deploy/modal_q8b36.py) o may local.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
if not exist D:\r2ai\data\sub\vi36 mkdir D:\r2ai\data\sub\vi36
set L=D:\r2ai\data\sub\vi36\vi36.log
echo start %DATE% %TIME% %* >> %L%
:loop
if "%1"=="" goto ok
echo === %1 %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi36.py %1 >> %L% 2>&1
if errorlevel 1 goto fail
shift
goto loop
:ok
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0
:fail
echo VI36 FAILED %1 >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
