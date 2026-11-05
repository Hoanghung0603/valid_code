@echo off
rem Nodriver pilot: single remaining ID 1900274 on b660 origin IP. Task crawl_nodriver_pilot ONCE 23:59, self-deletes.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
echo [pilot] start %DATE% %TIME% >> "D:\r2ai\data\crawl\browser_pilot.log"
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_nodriver_pilot.py --urls D:\r2ai\data\crawl\tools\pilot_urls_d.jsonl --out-root D:\r2ai\data\crawl --batch 20261009d --crawl-root D:\r2ai\data\crawl --lock-path D:\r2ai\locks\heavy_job.lock --delay 3.0 --timeout 45 >> "D:\r2ai\data\crawl\browser_pilot.log" 2>&1
echo [pilot] exit=%ERRORLEVEL% %DATE% %TIME% >> "D:\r2ai\data\crawl\browser_pilot.log"
start "" /min cmd /c "ping -n 8 localhost >nul & schtasks /Delete /TN crawl_nodriver_pilot /F >> D:\r2ai\data\crawl\browser_pilot.log 2>&1"
exit /b
