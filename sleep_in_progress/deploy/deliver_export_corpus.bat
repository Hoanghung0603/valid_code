@echo off
set PYTHONIOENCODING=utf-8
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\deliver\deliver_export_corpus.py > D:\r2ai\deliver\export.log 2>&1
exit /b %ERRORLEVEL%
