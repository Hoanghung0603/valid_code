@echo off
rem vi31 (b660, CPU): vi30 voi q8b_v31. Doi so: mot hoac nhieu stage
rem (sanity doclist chunk build stats), chay lan luot, dung o stage loi.
rem Log: D:\r2ai\data\sub\vi31\vi31.log. Lock heavy_job.lock tao dau / xoa cuoi.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
if not exist D:\r2ai\data\sub\vi31 mkdir D:\r2ai\data\sub\vi31
set L=D:\r2ai\data\sub\vi31\vi31.log
set K=D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% %* >> %L%
if exist %K% (
  echo LOCK BUSY %K% >> %L%
  echo exit fail >> %L%
  exit /b 1
)
echo vi31 %* %DATE% %TIME% > %K%
:loop
if "%1"=="" goto ok
echo === %1 %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi31.py %1 >> %L% 2>&1
if errorlevel 1 goto fail
shift
goto loop
:ok
del %K%
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0
:fail
del %K%
echo VI31 FAILED %1 >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
