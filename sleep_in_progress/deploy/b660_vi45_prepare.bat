@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\vi45 mkdir D:\r2ai\data\sub\vi45
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi45.py prepare >> D:\r2ai\data\sub\vi45\prepare.log 2>&1
exit /b %errorlevel%
