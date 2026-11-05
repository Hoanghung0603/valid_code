@echo off
set PYTHONIOENCODING=utf-8
set R2AI_ROOT=D:\r2ai
if not exist D:\r2ai\data\sub\vi38 mkdir D:\r2ai\data\sub\vi38
cd /d D:\r2ai
for %%S in (%*) do (
  D:\r2ai\.venv\Scripts\python.exe -u D:\r2ai\scripts\vi38.py %%S >> D:\r2ai\data\sub\vi38\vi38.log 2>&1
  if errorlevel 1 goto failed
)
echo exit=0>> D:\r2ai\data\sub\vi38\vi38.log
exit /b 0
:failed
echo exit=1>> D:\r2ai\data\sub\vi38\vi38.log
exit /b 1
