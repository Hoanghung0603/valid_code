@echo off
rem Phan B2 (b660, CPU): chon chunk lai tren chunk_dw_u900_p3 voi 3 doclist (t20/t16/nq).
rem Truoc het: repro sanity - code tren doc/ung vien CU (chunk_dw_u900_ext + doc nen cx130cov)
rem phai tai lap submission_cx130cov >= 99% cau (khong duoc thi dung).
rem Zd doc NEW: cache_rrf3d2_p3 = rrf3d2 TAI LAP tren pool3 (fuse3 bge + Qwen8B + dense*2, k=60,
rem 8B vang voi doc NEW). Build cache nay truoc run.
rem Zip LZMA <= 93MB: run tu dong giam delta (bat dau 57) theo uoc tinh; check-zip kiem zip THAT,
rem neu vuot ghi next_delta_<name>.txt -> chay lai rieng bai do voi --delta nho hon (toi da 3 lan).
rem Report rieng moi bai: report_<name>.json (khong de bai khac ghi de).
rem Log: D:\r2ai\data\sub\diag18\p3b.log. Lock heavy_job.lock.
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
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set L=%W%\p3b.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_p3b" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_p3b > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === build cache_rrf3d2_p3 (rrf3d2 tai lap tren pool3) >> %L%
%P% %S%\fuse3.py --pool %W%\cache_pool3_rr.json --q8b %Q8B% --dense %W%\cache_dense3.json --w-bge 1 --w-q8b 1 --w-dense 2 --k 60 --out %W%\cache_rrf3d2_p3.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === repro sanity tren doc/ung vien cu >> %L%
%P% %S%\cxp3.py repro --cands %DX%\chunk_dw_u900_ext.json.zst --ref-cx130cov %W17%\submission_cx130cov.json --fused %W10%\cache_rrf3d2.json --ref-c120in %W14%\submission_c120in.json --report %W%\repro_cx130.json >> %L% 2>&1
if errorlevel 1 goto fail

call :one submission_p3t20_cx %W%\doclist_p3_t20.json
if errorlevel 1 goto fail
call :one submission_p3t16_cx %W%\doclist_p3_t16.json
if errorlevel 1 goto fail
call :one submission_p3nq_cx %W%\doclist_p3_nq.json
if errorlevel 1 goto fail

echo === sub_stats >> %L%
%P% %S%\sub_stats.py %W%\submission_p3t20_cx.json %W%\submission_p3t16_cx.json %W%\submission_p3nq_cx.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_p3t20_cx.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_p3t16_cx.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_p3nq_cx.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

rem ---- :one <ten bai> <doclist>: run + validate + zip + check-zip, giam delta toi da 3 lan ----
:one
set NAME=%1
set DL=%2
set DELTA_ARG=
set TRIES=0
:one_build
echo === run %NAME% %DELTA_ARG% %DATE% %TIME% >> %L%
%P% %S%\cxp3.py run --cands %W%\chunk_dw_u900_p3.json.zst --doclist %NAME%=%DL% --fused %W10%\cache_rrf3d2.json --fused-new %W%\cache_rrf3d2_p3.json --ref-c120in %W14%\submission_c120in.json --new-cands %W%\new_cands.json --outdir %W% --report %W%\report_%NAME%.json --max-zip 93000000 %DELTA_ARG% >> %L% 2>&1
if errorlevel 1 exit /b 1
echo === validate %NAME% >> %L%
%P% %S%\evaluate.py validate %W%\%NAME%.json >> %L% 2>&1
if errorlevel 1 exit /b 1
echo === zip %NAME% >> %L%
%P% %S%\zip_submission.py --json %W%\%NAME%.json --zip %W%\%NAME%.zip --method lzma >> %L% 2>&1
if errorlevel 1 exit /b 1
echo === check-zip %NAME% >> %L%
%P% %S%\cxp3.py check-zip --json %W%\%NAME%.json --zip %W%\%NAME%.zip --report %W%\report_%NAME%.json --name %NAME% --max-zip 93000000 >> %L% 2>&1
set RC=%errorlevel%
if %RC%==0 exit /b 0
if %RC%==3 goto one_reduce
exit /b 1
:one_reduce
set /p DELTA_ARG=<%W%\next_delta_%NAME%.txt
set /a TRIES+=1
if %TRIES% GEQ 3 exit /b 1
echo === %NAME% vuot zip THAT, giam %DELTA_ARG% (lan %TRIES%) >> %L%
goto one_build

:fail
echo P3B FAILED, khong nop bai >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
