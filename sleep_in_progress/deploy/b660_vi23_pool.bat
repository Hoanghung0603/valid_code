@echo off
rem vi23 buoc 5 (b660, CPU): gop dense top-200 cu + top-200 doc moi theo cosine (giu diem cu) cho cau goc
rem va HyDE; tinh doc MOI vao pool_v23 (BM25 vi23 top-200 | dense gop top-200) - pool cu -> cap can rerank.
rem Log: diag23\pool.log. Lock heavy_job.lock (ngan).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set L=%W%\pool.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_pool" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_pool > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === dense-merge cau goc >> %L%
%P% %S%\vi23.py dense-merge --cache-old D:\r2ai\data\sub\zhprobe\dense_full_top200.json --cache-new %W%\dense_new_top200.json --top 200 --out %W%\dense_v23_top200.json --report %W%\dense_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === dense-merge HyDE >> %L%
%P% %S%\vi23.py dense-merge --cache-old D:\r2ai\data\sub\hyde\dense_hyde_top200.json --cache-new %W%\hyde_new_top200.json --top 200 --out %W%\hyde_v23_top200.json --report %W%\hyde_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === pool-new >> %L%
%P% %S%\vi23.py pool-new --pool D:\r2ai\data\sub\cache_pool_rr.json --bm25-new %W%\cache_vi23_k300.json --dense-new %W%\dense_v23_top200.json --new-ids %W%\new_ids.json --idx %W%\new_idx.json --out-cands %W%\cands_rr_new.json --out-new-hits %W%\new_hits.json --report %W%\pool_new_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sha256 >> %L%
certutil -hashfile %W%\cands_rr_new.json SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_POOL FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
