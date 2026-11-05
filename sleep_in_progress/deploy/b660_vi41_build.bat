@echo off
call D:\r2ai\b660_vi41.bat merge
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat chunk
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat build
if errorlevel 1 exit /b %errorlevel%
call D:\r2ai\b660_vi41.bat stats
exit /b %errorlevel%
