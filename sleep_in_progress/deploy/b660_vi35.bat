@echo off
rem vi35 (b660, CPU): vi34 voi n_q theo diem fused fhyde (ratio), thr hieu chinh o stage calib.
rem Doi so: mot hoac nhieu stage (sanity calib doclist chunk build stats), chay lan luot,
rem dung o stage loi. Log: D:\r2ai\data\sub\vi35\vi35.log. Lock heavy_job.lock tao dau / xoa cuoi.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
if not exist D:\r2ai\data\sub\vi35 mkdir D:\r2ai\data\sub\vi35
set L=D:\r2ai\data\sub\vi35\vi35.log
set K=D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% %* >> %L%
if exist %K% (
  echo LOCK BUSY %K% >> %L%
  echo exit fail >> %L%
  exit /b 1
)
echo vi35 %* %DATE% %TIME% > %K%
:loop
if "%1"=="" goto ok
echo === %1 %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi35.py %1 >> %L% 2>&1
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
echo VI35 FAILED %1 >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
