@echo off
cd /d D:\r2ai
echo r2ai_vi23_build > D:\r2ai\locks\heavy_job.lock
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\vi23.py check-zip --json D:\r2ai\data\sub\diag23\submission_vi23.json --zip D:\r2ai\data\sub\diag23\submission_vi23.zip --report D:\r2ai\data\sub\diag23\report_chain_vi23.json --name vi23 --max-zip 93000000 > D:\r2ai\data\sub\diag23\resume_build.log 2>&1
if not errorlevel 3 goto fail
if errorlevel 4 goto fail
set /p DELTA_ARG=<D:\r2ai\data\sub\diag23\next_delta_vi23.txt
call D:\r2ai\b660_vi23_build.bat %DELTA_ARG%
exit /b
:fail
del D:\r2ai\locks\heavy_job.lock
exit /b 1
