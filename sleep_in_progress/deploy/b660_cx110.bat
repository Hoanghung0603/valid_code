@echo off
rem cx110: submission_cx110cov - giong cx93cov (doc nen split, ung vien docwin_ext, s = zc + 4*zd, b = 1,0)
rem nhung k'_q = min(k_q + delta, |C_q|), chon delta nho nhat de TB k' >= 110,0; zip LZMA <= 95.000.000 B.
rem Kiem tai lap truoc: delta = 0 phai trung y nguyen submission_cx93cov.json (khong 100% thi dung).
rem Log: D:\r2ai\data\sub\diag16\cx110.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
rem Zip vuot 95MB: check-zip ghi next_delta.txt -> chay lai run voi --delta nho hon.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag16
set W10=D:\r2ai\data\sub\diag10
set W12=D:\r2ai\data\sub\diag12
set W14=D:\r2ai\data\sub\diag14
set W15=D:\r2ai\data\sub\diag15
set DX=D:\r2ai\data\sub\docwin_ext
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\cx110.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_cx110" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_cx110 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

set DELTA_ARG=
:build
echo === run %DELTA_ARG% %DATE% %TIME% >> %L%
%P% %S%\cx110.py run --cands %DX%\chunk_dw_u900_ext.json.zst --split %W12%\submission_split.json --fused %W10%\cache_rrf3d2.json --ref-c120in %W14%\submission_c120in.json --pool %POOL% --ref-cx93cov %W15%\submission_cx93cov.json --dq %DX%\cache_docwin_ext.json --outdir %W% %DELTA_ARG% >> %L% 2>&1
if errorlevel 1 goto fail
echo === validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_cx110cov.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === zip lzma >> %L%
%P% %S%\zip_submission.py --json %W%\submission_cx110cov.json --zip %W%\submission_cx110cov.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
echo === check zip >> %L%
%P% %S%\cx110.py check-zip --json %W%\submission_cx110cov.json --zip %W%\submission_cx110cov.zip --report %W%\report.json >> %L% 2>&1
set RC=%errorlevel%
if %RC%==0 goto packed
if %RC%==3 goto reduce
goto fail
:reduce
set /p DELTA_ARG=<%W%\next_delta.txt
echo === zip vuot 95MB, giam delta %DELTA_ARG% >> %L%
goto build
:packed
echo === sub_stats >> %L%
%P% %S%\sub_stats.py %W%\submission_cx110cov.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_cx110cov.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo CX110 FAILED, khong nop bai >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
