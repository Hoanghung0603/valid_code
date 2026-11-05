@echo off
rem candgen14: sinh ung vien doc NGOAI pool (pool2 ngoai pool, HyDE, reason, dense top-300,
rem BM25 top-300), RRF k60 -> new_cands.json; dung 2 bai chan doan trong D:\r2ai\data\sub\diag18:
rem   submission_ceil2     docs = toan pool (thu tu cache_pool_rr) + NEW (RRF), chunks = y cx130cov
rem   submission_newonly30 docs = 30 NEW dau theo RRF, chunks = []
rem Dong goi zip LZMA (zip_submission.py), validate (evaluate.py), zipcheck + sha256. KHONG nop bai.
rem Log: D:\r2ai\data\sub\diag18\candgen14.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag18
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set P2=%W%\cache_pool2_rr.json
set HYDE=D:\r2ai\data\sub\hyde\dense_hyde_top200.json
set REASON=D:\r2ai\data\sub\hyde\dense_reason_top200.json
set DENSE=%W%\dense_full_top400.json
set BM25=D:\r2ai\data\sub\cache_vi_k300.json
set CX=D:\r2ai\data\sub\diag17\submission_cx130cov.json
set L=%W%\candgen14.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_cg14" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_cg14 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === stats >> %L%
%P% %S%\candgen14.py stats --pool %POOL% --pool2 %P2% --hyde %HYDE% --reason %REASON% --dense %DENSE% --bm25 %BM25% --dense-k 300 --bm25-k 300 --out-dir %W% --report %W%\stats_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === subs >> %L%
%P% %S%\candgen14.py subs --pool %POOL% --new %W%\new_cands.json --cx130 %CX% --out-dir %W% --report %W%\subs_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === validate ceil2 >> %L%
%P% %S%\evaluate.py validate %W%\submission_ceil2.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === validate newonly30 >> %L%
%P% %S%\evaluate.py validate %W%\submission_newonly30.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === zip lzma ceil2 >> %L%
%P% %S%\zip_submission.py --json %W%\submission_ceil2.json --zip %W%\submission_ceil2.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
echo === zip lzma newonly30 >> %L%
%P% %S%\zip_submission.py --json %W%\submission_newonly30.json --zip %W%\submission_newonly30.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
echo === zipcheck ceil2 >> %L%
%P% %S%\candgen14.py zipcheck --zip %W%\submission_ceil2.zip --max-zip 95000000 --report %W%\zipcheck_ceil2.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === zipcheck newonly30 >> %L%
%P% %S%\candgen14.py zipcheck --zip %W%\submission_newonly30.zip --max-zip 95000000 --report %W%\zipcheck_newonly30.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sub_stats ceil2 >> %L%
%P% %S%\sub_stats.py %W%\submission_ceil2.json >> %L% 2>&1
echo === sub_stats newonly30 >> %L%
%P% %S%\sub_stats.py %W%\submission_newonly30.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_ceil2.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_newonly30.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo CG14 FAILED, khong nop bai >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
