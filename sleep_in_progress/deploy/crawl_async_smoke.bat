@echo off
rem Async smoke: bounded --max-urls run to validate supervisor wiring on b660.
rem Task r2ai_crawl_async_smoke ONCE 23:59, self-deletes. Deletes its own
rem smoke state after? No: keeps smoke_* workdir for parent inspection.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set WD=D:\r2ai\data\crawl\async_smoke_20261009
if exist "%WD%\queue\host-familydoctor_com_cn.jsonl.zst" (
  echo [smoke] queue already staged, refusing overwrite
  exit /b 2
)
mkdir "%WD%\queue" 2>nul
mkdir "%WD%\state" 2>nul
mkdir "%WD%\pages" 2>nul
mkdir "%WD%\audit" 2>nul
copy /Y "D:\r2ai\data\crawl\async_recovery_20261009\queue\smalls-00.jsonl.zst" "%WD%\queue\" >> "%WD%\audit\launcher.log" 2>&1
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_async_supervisor.py --workdir "%WD%" --crawl-root D:\r2ai\data\crawl --lock D:\r2ai\locks\heavy_job.lock --concurrency 4 --per-host 1 --delay 2.0 --max-mbps 20.0 --timeout 30 --extract-workers 1 --min-free-gb 5 --browser-max 0 --no-archive --max-urls 30 >> "%WD%\audit\launcher.log" 2>&1
echo [smoke] exit=%ERRORLEVEL% %DATE% %TIME% >> "%WD%\audit\launcher.log" 2>&1
start "" /min cmd /c "ping -n 8 localhost >nul & schtasks /Delete /TN r2ai_crawl_async_smoke /F >> D:\r2ai\data\crawl\async_smoke_20261009\audit\launcher.log 2>&1"
exit /b
