@echo off
rem Realtime smoke: ONE public browser URL + ONE HTTPS proxy transport probe.
rem Task r2ai_crawl_async_smoke_rt ONCE 2099, self-deletes after run.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set LOG=D:\r2ai\data\crawl\async_smoke_realtime_20261010.log
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_async_realtime_smoke.py --proxy-file D:\r2ai\data\crawl\tools\proxy_secrets.txt --crawl-root D:\r2ai\data\crawl >> "%LOG%" 2>&1
echo [smoke-rt] exit=%ERRORLEVEL% %DATE% %TIME% >> "%LOG%" 2>&1
start "" /min cmd /c "ping -n 8 localhost >nul & schtasks /Delete /TN r2ai_crawl_async_smoke_rt /F >> D:\r2ai\data\crawl\async_smoke_realtime_20261010.log 2>&1"
exit /b
