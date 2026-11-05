@echo off
rem q8b31 select (b660, CPU): chon cap chua co diem 8B trong top-100 fhyde_v30 + xuat text/cands.
rem Log: D:\r2ai\data\sub\diag31\q8b31.log. Lock heavy_job.lock do q8b31.py quan ly.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
cd /d D:\r2ai\scripts
set L=D:\r2ai\data\sub\diag31\q8b31.log
echo start %DATE% %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\q8b31.py select --outdir D:\r2ai\data\sub\diag31\q8b31_in --shards 2 --bench 200 >> %L% 2>&1
if errorlevel 1 goto fail
echo exit done %DATE% %TIME% >> %L%
exit /b 0
:fail
echo exit fail %DATE% %TIME% >> %L%
exit /b 1
