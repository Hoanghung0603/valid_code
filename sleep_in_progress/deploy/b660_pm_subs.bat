@echo off
rem 2 bai tham do PubMed (CHUA nop): submission_pm_dense30 (top-30 theo RRF k60 VI+EN), pm_rr30 (top-30 theo logit rerank).
rem Moi cau 30 doc, chunk = title + "\n" + abstract nguyen van. Validate + zip LZMA + sha256 + thong ke logit.
rem Out: D:\r2ai\data\sub\diag20\. Lock: heavy_job.lock (tao dau, xoa cuoi). Log: D:\r2ai\data\sub\diag20\pm_subs.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag20
set PM=D:\r2ai\data\pubmed
set L=%W%\pm_subs.log
if not exist %W% mkdir %W%
:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_pm_subs" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_pm_subs > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% %S%\pubmed_sub.py --cache %PM%\pm_rrf_top100.json --texts %PM%\pubmed_text.jsonl.zst --out %W%\submission_pm_dense30.json --top 30 >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\pubmed_sub.py --cache %PM%\pm_rr_top100.json --texts %PM%\pubmed_text.jsonl.zst --out %W%\submission_pm_rr30.json --top 30 >> %L% 2>&1
if errorlevel 1 goto fail
echo === validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_pm_dense30.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\evaluate.py validate %W%\submission_pm_rr30.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === zip lzma >> %L%
%P% %S%\zip_submission.py --json %W%\submission_pm_dense30.json --zip %W%\submission_pm_dense30.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\zip_submission.py --json %W%\submission_pm_rr30.json --zip %W%\submission_pm_rr30.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
echo === logit stats >> %L%
%P% %S%\pubmed_logit_stats.py --pm %PM%\pm_rr_top100.json --pool D:\r2ai\data\sub\cache_pool_rr.json --out %W%\logit_stats.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_pm_dense30.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_pm_rr30.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo PM SUBS FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
