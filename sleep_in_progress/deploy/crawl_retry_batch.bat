@echo off
rem Bounded standalone retry batch for b660 leftovers (518 transient + 17 missing = 535).
rem Task: r2ai_crawl_retry_batch (ONCE /ST 23:59). Self-deletes when finished. Never touches r2ai_crawl.
rem Original state/pages/salvage/verification are read-only; output goes to data\crawl\retry_<VN date>\.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set RETRYDIR=D:\r2ai\data\crawl\retry_%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%
if not exist "D:\r2ai\data\crawl" mkdir "D:\r2ai\data\crawl"
echo [retry] start %DATE% %TIME% >> "D:\r2ai\data\crawl\retry_batch.log"
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_retry_batch.py --corpus D:\r2ai\data\links_corpus.parquet --state D:\r2ai\data\crawl\state --missing-summary D:\r2ai\data\crawl\verification\verify_summary.json --out-root D:\r2ai\data\crawl --concurrency 4 --per-host 2 --delay 1.0 --max-mbps 20 --timeout 30 --extract-workers 1 --min-free-gb 5 --lock D:\r2ai\locks\heavy_job.lock >> "D:\r2ai\data\crawl\retry_batch.log" 2>&1
echo [retry] exit=%ERRORLEVEL% %DATE% %TIME% >> "D:\r2ai\data\crawl\retry_batch.log"
start "" /min cmd /c "ping -n 8 localhost >nul & schtasks /Delete /TN r2ai_crawl_retry_batch /F >> D:\r2ai\data\crawl\retry_batch.log 2>&1"
exit /b
