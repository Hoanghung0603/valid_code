@echo off
rem vi27 (b660, CPU): 2 zip LZMA chi doi b_cov (2.0 / 4.0) tren vi25_z99 (cung doclist + ung vien + delta 57).
rem Co doi so: truyen thang cho vi27.py (vd: b660_vi27.bat sanity / b660_vi27.bat build / b660_vi27.bat report).
rem Khong doi so: chay day du sanity -> build -> report.
rem Log: D:\r2ai\data\sub\diag27\vi27.log. Lock heavy_job.lock do vi27.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
if not exist D:\r2ai\data\sub\diag27 mkdir D:\r2ai\data\sub\diag27
set L=D:\r2ai\data\sub\diag27\vi27.log
echo start %DATE% %TIME% >> %L%

if not "%1"=="" goto one

echo === sanity >> %L%
%P% %S%\vi27.py sanity >> %L% 2>&1
if errorlevel 1 goto fail
echo === build >> %L%
%P% %S%\vi27.py build >> %L% 2>&1
if errorlevel 1 goto fail
echo === report >> %L%
%P% %S%\vi27.py report >> %L% 2>&1
if errorlevel 1 goto fail
goto ok

:one
echo === %* >> %L%
%P% %S%\vi27.py %* >> %L% 2>&1
if errorlevel 1 goto fail

:ok
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0

:fail
echo VI27 FAILED >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
