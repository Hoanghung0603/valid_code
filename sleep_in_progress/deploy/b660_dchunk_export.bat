@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\dchunk mkdir D:\r2ai\data\sub\dchunk
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -m pip install transformers==4.51.3 sentencepiece==0.2.0 >> D:\r2ai\data\sub\dchunk\setup.log 2>&1
if errorlevel 1 exit /b 1
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\dchunk.py export >> D:\r2ai\data\sub\dchunk\export.log 2>&1
exit /b %errorlevel%
