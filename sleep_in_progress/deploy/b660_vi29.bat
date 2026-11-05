@echo off
rem vi29 (b660, CPU): vi28 + 2 sua loi (dedup chunk theo (doc_id, text) sau filter_base;
rem loc id ngoai kho chuan links_corpus_vi_en_zh khoi moi dau vao chain).
rem Co doi so: count | sanity | build | report | tokmeasure (truyen thang cho vi29.py).
rem Khong doi so: chay count -> sanity -> build -> report, dung o buoc loi (tokmeasure chay rieng sau).
rem Log: D:\r2ai\data\sub\diag29\vi29.log. Lock heavy_job.lock do vi29.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
if not exist D:\r2ai\data\sub\diag29 mkdir D:\r2ai\data\sub\diag29
set L=D:\r2ai\data\sub\diag29\vi29.log
echo start %DATE% %TIME% >> %L%

if not "%1"=="" goto one

echo === count >> %L%
%P% %S%\vi29.py count >> %L% 2>&1
if errorlevel 1 goto fail
echo === sanity >> %L%
%P% %S%\vi29.py sanity >> %L% 2>&1
if errorlevel 1 goto fail
echo === build >> %L%
%P% %S%\vi29.py build >> %L% 2>&1
if errorlevel 1 goto fail
echo === report >> %L%
%P% %S%\vi29.py report >> %L% 2>&1
if errorlevel 1 goto fail
goto ok

:one
echo === %* >> %L%
%P% %S%\vi29.py %* >> %L% 2>&1
if errorlevel 1 goto fail

:ok
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0

:fail
echo VI29 FAILED >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
