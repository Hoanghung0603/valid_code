@echo off
powershell -NoProfile -Command "$deadline=(Get-Date).AddHours(2); while (!(Test-Path D:\r2ai\data\sub\vi41\bm25_report.json)) {if ((Get-Date) -gt $deadline) {exit 2}; Start-Sleep -Seconds 10}"
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat dchunk
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat candidates
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat audit_inputs
exit /b %errorlevel%
