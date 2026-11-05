@echo off
rem One-shot contract verify of async_recovery pages (HANDOFF gate).
rem Task r2ai_crawl_async_verify_rt ONCE 2099, self-deletes after run.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set LOG=D:\r2ai\data\crawl\async_verify_20261010.log
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_async_verify_output.py --pages D:\r2ai\data\crawl\async_recovery_20261009\pages --json >> "%LOG%" 2>&1
echo [verify-rt] exit=%ERRORLEVEL% %DATE% %TIME% >> "%LOG%" 2>&1
start "" /min cmd /c "ping -n 8 localhost >nul & schtasks /Delete /TN r2ai_crawl_async_verify_rt /F >> D:\r2ai\data\crawl\async_verify_20261010.log 2>&1"
exit /b
