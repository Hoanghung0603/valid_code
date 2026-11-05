@echo off
rem runner_z87.bat <tag> <stage...>  -- chay runner_private tren z87 (GPU du phong: dense).
rem Log: D:\r2ai\work\runner\<tag>\<stage>.log. Lock gpu.lock do runner_private.py quan ly
rem (owned_lock); do tre vinmec ghi trong log, > 1s thi tu dung. Uu tien thap (start /low /wait).
rem Vi du: runner_z87.bat rsmoke20 dense
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set R2AI_ROOT=D:\r2ai
set TAG=%1
shift
cd /d D:\r2ai\runner\scripts
if not exist D:\r2ai\work\runner\%TAG% mkdir D:\r2ai\work\runner\%TAG%
:loop
if "%1"=="" goto ok
set L=D:\r2ai\work\runner\%TAG%\%1.log
echo start %DATE% %TIME% > %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\runner\scripts\runner_private.py %1 --tag %TAG% --machine z87 %RUNNER_ARGS% >> %L% 2>&1"
set RC=%errorlevel%
echo exit %RC% %DATE% %TIME% >> %L%
if not "%RC%"=="0" exit /b %RC%
shift
goto loop
:ok
echo exit done %DATE% %TIME%
exit /b 0
