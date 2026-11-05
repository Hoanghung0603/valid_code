@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\vi41b mkdir D:\r2ai\data\sub\vi41b
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\vi41b_cpu.py build >> D:\r2ai\data\sub\vi41b\cpu.log 2>&1
exit /b %errorlevel%
