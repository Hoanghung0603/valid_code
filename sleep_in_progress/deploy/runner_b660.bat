@echo off
rem runner_b660.bat <tag> <stage...>  -- chay runner_private tren b660 (CPU), tu snapshot
rem D:\r2ai\runner\scripts (khong dung D:\r2ai\scripts de khong anh huong phien vi36).
rem Log: D:\r2ai\data\sub\runner\<tag>\runner.log. Lock heavy_job.lock tao dau / xoa cuoi.
rem Vi du: runner_b660.bat rsmoke20 prep bm25 hyde
rem Tham so them cho moi stage dat vao bien moi truong RUNNER_ARGS (vd set RUNNER_ARGS=--delta 56 --smoke 20)
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set R2AI_ROOT=D:\r2ai
set TAG=%1
shift
cd /d D:\r2ai\runner\scripts
if not exist D:\r2ai\data\sub\runner\%TAG% mkdir D:\r2ai\data\sub\runner\%TAG%
set L=D:\r2ai\data\sub\runner\%TAG%\runner.log
set K=D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% %TAG% %* >> %L%
if exist %K% (
  echo LOCK BUSY %K% >> %L%
  echo exit fail >> %L%
  exit /b 1
)
echo runner %TAG% %DATE% %TIME% > %K%
:loop
if "%1"=="" goto ok
echo === %1 %TIME% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\runner\scripts\runner_private.py %1 --tag %TAG% %RUNNER_ARGS% >> %L% 2>&1
if errorlevel 1 goto fail
shift
goto loop
:ok
del %K%
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
exit /b 0
:fail
del %K%
echo RUNNER FAILED %1 >> %L%
echo end %DATE% %TIME% >> %L%
echo exit fail >> %L%
exit /b 1
