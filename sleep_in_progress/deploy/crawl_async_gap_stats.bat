@echo off
rem Per-host stats for shards not yet handed off (read-only). No crawler action.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
cd /d D:\r2ai\data\crawl\tools
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_async_gap_stats.py >> D:\r2ai\data\crawl\gap_stats.log 2>&1
echo [gap_stats] exit=%ERRORLEVEL% %DATE% %TIME% >> D:\r2ai\data\crawl\gap_stats.log
exit /b