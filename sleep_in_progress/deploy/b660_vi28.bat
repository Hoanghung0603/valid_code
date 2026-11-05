@echo off
rem vi28 (b660, CPU): vi25_z99 (#6757) gom 16.079 ban ghi nhiem cheo stale-tab (24/10).
rem Co doi so: verify | sanity | build | report (truyen thang cho vi28.py).
rem Khong doi so: chay day du verify -> sanity -> build -> report, dung o buoc loi.
rem Log: D:\r2ai\data\sub\diag28\vi28.log. Lock heavy_job.lock do vi28.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
if not exist D:\r2ai\data\sub\diag28 mkdir D:\r2ai\data\sub\diag28
set L=D:\r2ai\data\sub\diag28\vi28.log
echo start %DATE% %TIME% >> %L%

if not "%1"=="" goto one

echo === verify >> %L%
%P% %S%\vi28.py verify >> %L% 2>&1
if errorlevel 1 goto fail
echo === sanity >> %L%
%P% %S%\vi28.py sanity >> %L% 2>&1
if errorlevel 1 goto fail
echo === build >> %L%
%P% %S%\vi28.py build >> %L% 2>&1
if errorlevel 1 goto fail
echo === report >> %L%
%P% %S%\vi28.py report >> %L% 2>&1
if errorlevel 1 goto fail
goto ok

:one
echo === %* >> %L%
%P% %S%\vi28.py %* >> %L% 2>&1
if errorlevel 1 goto fail

:ok
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0

:fail
echo VI28 FAILED >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
