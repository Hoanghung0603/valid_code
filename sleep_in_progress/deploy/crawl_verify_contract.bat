@echo off
rem Hop dong verify_shards.py cho manifest healthy (94 shard sach, loai crawl-00009).
rem Chay inline qua SSH: cmd /c D:\r2ai\data\crawl\tools\crawl_verify_contract.bat
rem Neu chay qua Task Scheduler (qua 2 phut) thi xoa task sau khi xong: schtasks /Delete /TN crawl_verify_contract /F
setlocal
set VERIFY=D:\r2ai\data\crawl\verification
echo [contract] start %DATE% %TIME% >> "%VERIFY%\verify_contract.log"
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\verify_shards.py D:\r2ai\data\crawl\pages "%VERIFY%\verify_manifest_healthy.sha256" >> "%VERIFY%\verify_contract.log" 2>&1
echo [contract] exit=%ERRORLEVEL% %DATE% %TIME% >> "%VERIFY%\verify_contract.log"
exit /b
