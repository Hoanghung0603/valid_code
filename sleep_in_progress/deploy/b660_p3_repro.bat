@echo off
rem Repro sanity rieng (b660, CPU): chon chunk lai tren doc/ung vien CU (chunk_dw_u900_ext +
rem doc nen = relevant_docs cua cx130cov) voi cong thuc cx130cov phai tai lap submission_cx130cov
rem >= 99% cau. Log: D:\r2ai\data\sub\diag18\repro.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag18
set W10=D:\r2ai\data\sub\diag10
set W14=D:\r2ai\data\sub\diag14
set W17=D:\r2ai\data\sub\diag17
set DX=D:\r2ai\data\sub\docwin_ext
set L=%W%\repro.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_p3_repro" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_p3_repro > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% %S%\cxp3.py repro --cands %DX%\chunk_dw_u900_ext.json.zst --ref-cx130cov %W17%\submission_cx130cov.json --fused %W10%\cache_rrf3d2.json --ref-c120in %W14%\submission_c120in.json --report %W%\repro_cx130.json >> %L% 2>&1
set RC=%errorlevel%
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
if %RC%==0 (echo exit done >> %L%) else (echo exit fail rc=%RC% >> %L%)
goto :eof
