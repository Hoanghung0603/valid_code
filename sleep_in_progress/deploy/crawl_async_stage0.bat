@echo off
rem Stage 0 (single remote setup): extract async queue into NEW workdir only.
rem Never touches original state/pages. Safe to run once via schtasks /Run.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set WD=D:\r2ai\data\crawl\async_recovery_20261009
if exist "%WD%\queue\host-familydoctor_com_cn.jsonl.zst" (
  echo [stage0] queue already staged, refusing overwrite
  exit /b 2
)
mkdir "%WD%\queue" 2>nul
mkdir "%WD%\state" 2>nul
mkdir "%WD%\pages" 2>nul
mkdir "%WD%\audit" 2>nul
D:\r2ai\.venv\Scripts\python.exe -B -c "print('stage0-start')"
tar -xzf D:\r2ai\data\crawl\tools\async_queue.tar.gz -C "%WD%"
echo [stage0] exit=%ERRORLEVEL% %DATE% %TIME% >> "%WD%\audit\launcher.log"
dir "%WD%\queue" >> "%WD%\audit\launcher.log" 2>&1
dir "%WD%\queue_manifest.json" >> "%WD%\audit\launcher.log" 2>&1
exit /b
