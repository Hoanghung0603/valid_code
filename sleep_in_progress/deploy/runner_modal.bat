@echo off
rem runner_modal.bat <stage> <tag> [cands <file>] [out <file>]  -- khoi chay GPU Modal cho runner
rem (chay nen bang Start-Process, log ra file). Phai chay TREN MAY LOCAL (co modal.exe + token).
rem Stage: rr | docwin | q8b | bench_bge. Vi du:
rem   runner_modal.bat rr rsmoke20
rem   runner_modal.bat q8b rsmoke20 cands cands_q8b_shard0.json out q8b_shard0_out.json
rem   runner_modal.bat q8b rsmoke20 cands cands_q8b_bench.json out q8b_bench_out.json
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set STAGE=%1
set TAG=%2
shift
shift
set EXTRA=
:parse
if "%1"=="" goto run
set EXTRA=!EXTRA! --%1 %2
shift
shift
goto parse
:run
set M=%HIST_TMP%\modalenv\Scripts\modal.exe
set R=%WORKDIR%\r2ai
set LD=%HIST_TMP%\runner
if not exist %LD% mkdir %LD%
set L=%LD%\modal_%STAGE%_%TAG%.log
echo start %DATE% %TIME% %STAGE% %TAG% %EXTRA% > %L%
powershell -NoProfile -Command "Start-Process -FilePath '%M%' -ArgumentList 'run','--detach','%R%\deploy\modal_runner.py','--stage','%STAGE%','--tag','%TAG%'%EXTRA% -RedirectStandardOutput '%L%' -RedirectStandardError '%LD%\modal_%STAGE%_%TAG%.err' -WindowStyle Hidden"
echo exit started %DATE% %TIME% >> %L%
exit /b 0
