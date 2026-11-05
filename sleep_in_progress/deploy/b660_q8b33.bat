@echo off
rem q8b33 select: missing q8b pairs in full doclist_vi33 (+30 margin) of fhyde_v33, restricted to pool_v31.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
cd /d D:\r2ai\scripts
if not exist D:\r2ai\data\sub\diag33 mkdir D:\r2ai\data\sub\diag33
set L=D:\r2ai\data\sub\diag33\q8b33.log
echo start %DATE% %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\q8b33.py select --outdir D:\r2ai\data\sub\diag33\q8b33_in --shards 4 --bench 100 --index D:\r2ai\data\index\vi30 >> %L% 2>&1
if errorlevel 1 goto fail
echo exit done %DATE% %TIME% >> %L%
exit /b 0
:fail
echo exit fail %DATE% %TIME% >> %L%
exit /b 1
