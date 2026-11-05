@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\diag37 mkdir D:\r2ai\data\sub\diag37
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\dwmax_diag.py > D:\r2ai\data\sub\diag37\dwmax_diag.log 2>&1
if errorlevel 1 goto failed
echo exit=0>> D:\r2ai\data\sub\diag37\dwmax_diag.log
exit /b 0
:failed
echo exit=1>> D:\r2ai\data\sub\diag37\dwmax_diag.log
exit /b 1
