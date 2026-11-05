@echo off
rem vi23 buoc 8 (b660, CPU, ngan): gop logit rerank doc moi vao pool_v23 (GIU nguyen doc cu + logit cu),
rem tinh rrf3d2_v23 + fhyde_v23. Log: diag23\fuse.log. Lock heavy_job.lock (ngan).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set L=%W%\fuse.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_fuse" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_fuse > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === pool-merge >> %L%
%P% %S%\vi23.py pool-merge --pool-old D:\r2ai\data\sub\cache_pool_rr.json --new-hits %W%\new_hits.json --rerank %W%\rerank_new.json --out %W%\cache_pool_v23.json --report %W%\pool_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === fuse rrf3d2 + fhyde >> %L%
%P% %S%\vi23.py fuse --pool %W%\cache_pool_v23.json --q8b D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json --dense %W%\dense_v23_top200.json --hyde %W%\hyde_v23_top200.json --out-rrf3d2 %W%\cache_rrf3d2_v23.json --out-fhyde %W%\cache_fhyde_v23.json --report %W%\fuse_report.json >> %L% 2>&1
if errorlevel 1 goto fail
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_FUSE FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
