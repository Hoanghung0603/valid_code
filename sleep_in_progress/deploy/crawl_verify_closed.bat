@echo off
rem Kiem tra read-only shard crawl da dong tren b660. Chay qua Task Scheduler: schtasks /Run /TN r2ai_crawl_verify
rem ONCE /ST 23:59; cuoi script tu xoa task de khong chay lai. Khong dung task r2ai_crawl.
setlocal
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set VERIFY=D:\r2ai\data\crawl\verification
if not exist "%VERIFY%" mkdir "%VERIFY%"
echo [verify] start %DATE% %TIME% >> "%VERIFY%\verify.log"
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\data\crawl\tools\crawl_verify_closed.py --pages D:\r2ai\data\crawl\pages --state D:\r2ai\data\crawl\state --out D:\r2ai\data\crawl\verification --lock D:\r2ai\locks\heavy_job.lock >> "%VERIFY%\verify.log" 2>&1
echo [verify] exit=%ERRORLEVEL% %DATE% %TIME% >> "%VERIFY%\verify.log"
rem Xoa task ONCE (tien trinh tach roi; task ket thuc truoc khi schtasks /Delete chay).
start "" /min cmd /c "ping -n 8 localhost >nul & schtasks /Delete /TN r2ai_crawl_verify /F >> D:\r2ai\data\crawl\verification\verify.log 2>&1"
exit /b
