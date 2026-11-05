@echo off
rem vi26 (b660, CPU): ung vien chunk U900/U600/U450 cua vi25_z99 (cung docwin) + 2 zip LZMA <= 99MB.
rem Khong doi so: chay day du export-v23-docs -> cands 900 600 -> sanity -> build u600 -> cands 450
rem -> build u450. Co doi so: truyen thang cho vi26.py (vd: b660_vi26.bat cands --units 450).
rem Log: D:\r2ai\data\sub\diag26\vi26.log. Lock heavy_job.lock do vi26.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
if not exist D:\r2ai\data\sub\diag26 mkdir D:\r2ai\data\sub\diag26
set L=D:\r2ai\data\sub\diag26\vi26.log
echo start %DATE% %TIME% >> %L%

if not "%1"=="" goto one

echo === export-v23-docs >> %L%
%P% %S%\vi26.py export-v23-docs >> %L% 2>&1
if errorlevel 1 goto fail
echo === cands 900 600 >> %L%
%P% %S%\vi26.py cands --units 900 600 >> %L% 2>&1
if errorlevel 1 goto fail
echo === sanity >> %L%
%P% %S%\vi26.py sanity >> %L% 2>&1
if errorlevel 1 goto fail
echo === build u600 >> %L%
%P% %S%\vi26.py build --zips 0 --delta-start 120 >> %L% 2>&1
if errorlevel 1 goto fail
echo === cands 450 >> %L%
%P% %S%\vi26.py cands --units 450 >> %L% 2>&1
if errorlevel 1 goto fail
echo === build u450 >> %L%
%P% %S%\vi26.py build --zips 1 --delta-start 187 >> %L% 2>&1
if errorlevel 1 goto fail
goto ok

:one
echo === %* >> %L%
%P% %S%\vi26.py %* >> %L% 2>&1
if errorlevel 1 goto fail

:ok
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0

:fail
echo VI26 FAILED >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
