@echo off
rem cx93: 2 bai nhieu chunk tu ung vien mo rong docwin_ext (cx93in, cx93cov), zip LZMA (khong nop bai).
rem Kiem tai lap truoc: chon cx93in tren ung vien CU chunk_dw_u900.json.zst phai trung y nguyen
rem relevant_chunks cua submission_c120in.json (neu khong thi dung). k_q = so chunk cua cau trong c120in.
rem Doc nen = relevant_docs cua submission_split, giu y nguyen (0 doc them).
rem Log: D:\r2ai\data\sub\diag15\cx93.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag15
set W10=D:\r2ai\data\sub\diag10
set W12=D:\r2ai\data\sub\diag12
set W14=D:\r2ai\data\sub\diag14
set DW=D:\r2ai\data\sub\docwin
set DX=D:\r2ai\data\sub\docwin_ext
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\cx93.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_cx93" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_cx93 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% %S%\cx93.py repro --cands %DW%\chunk_dw_u900.json.zst --split %W12%\submission_split.json --fused %W10%\cache_rrf3d2.json --ref-c120in %W14%\submission_c120in.json --pool %POOL% --report %W%\repro_report.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\cx93.py run --cands %DX%\chunk_dw_u900_ext.json.zst --split %W12%\submission_split.json --fused %W10%\cache_rrf3d2.json --ref-c120in %W14%\submission_c120in.json --pool %POOL% --outdir %W% >> %L% 2>&1
if errorlevel 1 goto fail
call :pack cx93in
if errorlevel 1 goto fail
call :pack cx93cov
if errorlevel 1 goto fail
echo === sub_stats >> %L%
%P% %S%\sub_stats.py %W%\submission_cx93in.json %W%\submission_cx93cov.json >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:pack
echo === %1 validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_%1.json >> %L% 2>&1
if errorlevel 1 exit /b 1
echo === %1 zip lzma >> %L%
%P% %S%\zip_submission.py --json %W%\submission_%1.json --zip %W%\submission_%1.zip --method lzma >> %L% 2>&1
if errorlevel 1 exit /b 1
echo === %1 verify zip >> %L%
%P% %S%\cx93.py verify-zip --json %W%\submission_%1.json --zip %W%\submission_%1.zip >> %L% 2>&1
if errorlevel 1 exit /b 1
echo === %1 sha256 >> %L%
certutil -hashfile %W%\submission_%1.zip SHA256 >> %L% 2>&1
exit /b 0

:fail
echo CX93 FAILED, khong dung bai >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
