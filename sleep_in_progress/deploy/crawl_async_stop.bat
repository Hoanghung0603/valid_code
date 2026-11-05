@echo off
rem Stop async recovery gracefully (own STOP file; supervisor checkpoints).
setlocal
echo stop > "D:\r2ai\data\crawl\async_recovery_20261009\STOP"
echo [stop] wrote STOP %DATE% %TIME%
exit /b 0
