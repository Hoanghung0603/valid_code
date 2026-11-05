@echo off
rem diag14: 3 bai nhieu chunk hon (c120, c120in, ck105) tu ung vien chunk U900, zip LZMA (khong nop bai).
rem Kiem tai lap truoc: voi k_q goc phai trung y chunks_rrf3d2_cf4.json, neu khong thi dung.
rem Doc nen = relevant_docs cua submission_split (cache_fhyde, T -2.0) + doc nguon chunk con thieu.
rem Log: D:\r2ai\data\sub\diag14\c120.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag14
set W10=D:\r2ai\data\sub\diag10
set W11=D:\r2ai\data\sub\diag11
set W12=D:\r2ai\data\sub\diag12
set DW=D:\r2ai\data\sub\docwin
set L=%W%\c120.log
if not exist %W% mkdir %W%
echo r2ai_c120 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% %S%\diag14.py run --split %W12%\submission_split.json --rerank %DW%\chunk_dw_u900.json.zst --fused %W10%\cache_rrf3d2.json --ref-chunks %DW%\chunks_d900t015.json --target %W11%\chunks_rrf3d2_cf4.json --outdir %W% --wd 4 --delta 30 --cap 120 >> %L% 2>&1
if errorlevel 1 goto fail
call :pack c120
call :pack c120in
call :pack ck105
echo === sub_stats >> %L%
%P% %S%\sub_stats.py %W%\submission_c120.json %W%\submission_c120in.json %W%\submission_ck105.json >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:pack
echo === %1 validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_%1.json >> %L% 2>&1
echo === %1 zip lzma >> %L%
%P% %S%\zip_submission.py --json %W%\submission_%1.json --zip %W%\submission_%1.zip --method lzma >> %L% 2>&1
echo === %1 verify zip >> %L%
%P% %S%\diag14.py verify-zip --json %W%\submission_%1.json --zip %W%\submission_%1.zip >> %L% 2>&1
echo === %1 sha256 >> %L%
certutil -hashfile %W%\submission_%1.zip SHA256 >> %L% 2>&1
goto :eof

:fail
echo DIAG14 RUN FAILED, khong dung bai >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
