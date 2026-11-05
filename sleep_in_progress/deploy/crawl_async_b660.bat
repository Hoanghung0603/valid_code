@echo off
rem Async broad recovery: manual trigger ONLY (no ONCE 23:59 rerun risk).
rem Task r2ai_crawl_async_recovery trigger is non-repeating (2099). The
rem supervisor itself loops unattended cycles; this bat only launches it.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
powershell -NoProfile -ExecutionPolicy Bypass -File D:\r2ai\data\crawl\tools\crawl_async_unlimit.ps1 >> "D:\r2ai\data\crawl\async_recovery_20261009\audit\launcher.log" 2>&1
echo [async] start %DATE% %TIME% >> "D:\r2ai\data\crawl\async_recovery_20261009\audit\launcher.log"
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_async_supervisor.py --workdir D:\r2ai\data\crawl\async_recovery_20261009 --crawl-root D:\r2ai\data\crawl --lock D:\r2ai\locks\heavy_job.lock --proxy-file D:\r2ai\data\crawl\tools\proxy_secrets.txt --concurrency 16 --per-host 4 --vi-per-host 8 --delay 2.0 --max-mbps 20.0 --timeout 30 --extract-workers 3 --min-free-gb 5 --browser-max 8 --headed-hosts nhathuoclongchau.com.vn --browser-proxy-hosts nhathuoclongchau.com.vn --cycle-cooldown 60 >> "D:\r2ai\data\crawl\async_recovery_20261009\audit\launcher.log" 2>&1
echo [async] exit=%ERRORLEVEL% %DATE% %TIME% >> "D:\r2ai\data\crawl\async_recovery_20261009\audit\launcher.log"
exit /b
