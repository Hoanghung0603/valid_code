@echo off
rem vi23 buoc 13 (b660, CPU): bai phu newonly30 (30 doc moi/cau theo fhyde_v23, khong chunk) + zip + stats
rem buoc 8 (TB doc/chunk moi cau, top-30 trung fhyde cu, phan bo host doc moi). Khong nop bai.
rem Log: diag23\stats.log. Lock heavy_job.lock (ngan).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set L=%W%\stats.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_stats" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_stats > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === newonly30 >> %L%
%P% %S%\vi23.py newonly30 --fhyde %W%\cache_fhyde_v23.json --new-ids %W%\new_ids.json --n 30 --out %W%\submission_newonly30.json --report %W%\newonly30_report.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\evaluate.py validate %W%\submission_newonly30.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\zip_submission.py --json %W%\submission_newonly30.json --zip %W%\submission_newonly30.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
echo === stats bai chinh >> %L%
%P% %S%\vi23.py stats --submission %W%\submission_vi23.json --fhyde %W%\cache_fhyde_v23.json --fhyde-old D:\r2ai\data\sub\diag12\cache_fhyde.json --pool %W%\cache_pool_v23.json --new-ids %W%\new_ids.json --links %W%\new_host.json --report %W%\stats_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sha256 >> %L%
certutil -hashfile %W%\submission_newonly30.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_STATS FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
