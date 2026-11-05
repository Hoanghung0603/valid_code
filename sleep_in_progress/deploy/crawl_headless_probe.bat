@echo off
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_headless_probe.py --urls D:\r2ai\data\crawl\tools\probe_urls.jsonl --out D:\r2ai\data\crawl\tools\probe_headless_result.jsonl --crawl-root D:\r2ai\data\crawl --timeout 45 --delay 3 > D:\r2ai\data\crawl\tools\probe_headless.log 2>&1
echo EXIT=%ERRORLEVEL%>> D:\r2ai\data\crawl\tools\probe_headless.log
schtasks /Delete /TN r2ai_headless_probe /F >> D:\r2ai\data\crawl\tools\probe_headless.log 2>&1
