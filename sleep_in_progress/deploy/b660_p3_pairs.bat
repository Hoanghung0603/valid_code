@echo off
rem p3 pairs: tim cap NEW thieu logit bge (rerank z87), toan bo cap NEW (dense z87),
rem 200 cap sanity da co logit (cham lai doi chieu). Out: D:\r2ai\data\sub\diag18\{cands_p3,cands_new_all,sanity_pairs,sanity_ref}.json
rem Log: D:\r2ai\data\sub\diag18\p3_pairs.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
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
set DENSE400=%W%\dense_full_top400.json
set BM25=D:\r2ai\data\sub\cache_vi_k300.json
set L=%W%\p3_pairs.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_p3pairs" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_p3pairs > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% %S%\pool3.py pairs --pool %POOL% --pool2 %P2% --new %W%\new_cands.json --idx-from %HYDE% %REASON% %DENSE400% %BM25% --out-cands %W%\cands_p3.json --out-new-all %W%\cands_new_all.json --out-sanity %W%\sanity_pairs.json --out-sanity-ref %W%\sanity_ref.json --report %W%\pairs_report.json >> %L% 2>&1
if errorlevel 1 goto fail
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo P3PAIRS FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
