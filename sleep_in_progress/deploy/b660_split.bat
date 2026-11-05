@echo off
rem Bai "split": danh sach doc theo rrf3d2 + HyDE x1 (cache_fhyde, #5301 Docs F2 0,3041),
rem chunk giu nguyen cua #5248 (chunks_rrf3d2_cf4.json, Chunks F2 0,2437). Chi doi nguon doc, T -2.0.
rem Log: D:\r2ai\data\sub\diag12\split.log. Lock: D:\r2ai\locks\heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W11=D:\r2ai\data\sub\diag11
set W=D:\r2ai\data\sub\diag12
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\split.log
echo r2ai_split > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_fhyde.json --top-ens 1000000 --thr -2.0 --chunks-from %W11%\chunks_rrf3d2_cf4.json --ref %W11%\submission_t20_cf4.json --out %W%\submission_split.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_split.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_split.json --zip %W%\submission_split.zip --level 9 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
