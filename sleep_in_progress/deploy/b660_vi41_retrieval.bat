@echo off
call D:\r2ai\b660_vi41.bat sanity
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat neardup
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat bm25
exit /b %errorlevel%
