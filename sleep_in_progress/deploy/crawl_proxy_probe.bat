@echo off
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_proxy_probe.py >> D:\r2ai\data\crawl\proxy_probe_20261021.log 2>&1
exit /b %ERRORLEVEL%
