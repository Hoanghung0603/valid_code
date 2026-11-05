@echo off
rem Do lai so doc n_q tren thu hang hop nhat rrf3d2. Chunk giu nguyen #5084 (chunks_rrf3d2_cf3.json).
rem n_q = adaptive_n(pool bge, thr, 40, 400). #5084 = -2.6; #5124 -2.0 (0,2727), #5125 -3.2 (0,2678).
rem Lan 2: thu -1.6, -1.2, -0.8 (it doc hon nua).
rem Log: D:\r2ai\data\sub\diag11\nq.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W10=D:\r2ai\data\sub\diag10
set W=D:\r2ai\data\sub\diag11
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\nq.log
echo r2ai_nq > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
call :one t16 -1.6
call :one t12 -1.2
call :one t08 -0.8
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
echo ===== %1 (thr %2) >> %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %W10%\cache_rrf3d2.json --top-ens 1000000 --thr %2 --chunks-from %W10%\chunks_rrf3d2_cf3.json --ref %W10%\submission_full_rrf3d2_cf3.json --out %W%\submission_%1_cf3.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1_cf3.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1_cf3.json --zip %W%\submission_%1_cf3.zip --level 9 >> %L% 2>&1
goto :eof
