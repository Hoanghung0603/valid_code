@echo off
rem vi23 buoc 12 (b660, CPU): dung bai chinh theo cong thuc #5532 (chain --mode that: rrf3d2_v23, fhyde_v23,
rem n_q = adaptive_n(pool_v23, -2.0, 40, 400), doc nen = n_q doc dau fhyde + doc nguon chunk; chunk
rem cx130cov: s = zc + 4*zd, b = 1, k' = min(k_q + delta, |C_q|), delta 57). validate + zip LZMA +
rem check-zip; neu vuot 93MB thi giam delta theo next_delta_vi23.txt va lam lai.
rem Log: diag23\build.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set L=%W%\build.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_build" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_build > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
set DELTA_ARG=%*

:build
echo === chain that %DELTA_ARG% %DATE% %TIME% >> %L%
%P% %S%\vi23.py chain --mode that --name vi23 --pool %W%\cache_pool_v23.json --q8b D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json --dense %W%\dense_v23_top200.json --hyde %W%\hyde_v23_top200.json --cands %W%\chunk_dw_u900_v23.json.zst --ref-kq D:\r2ai\data\sub\diag14\submission_c120in.json --ref-chunks D:\r2ai\data\sub\diag11\chunks_rrf3d2_cf4.json --out-dir %W% %DELTA_ARG% >> %L% 2>&1
if errorlevel 1 goto fail
echo === validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_vi23.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === zip lzma >> %L%
%P% %S%\zip_submission.py --json %W%\submission_vi23.json --zip %W%\submission_vi23.zip --method lzma >> %L% 2>&1
if errorlevel 1 goto fail
echo === check zip >> %L%
%P% %S%\vi23.py check-zip --json %W%\submission_vi23.json --zip %W%\submission_vi23.zip --report %W%\report_chain_vi23.json --name vi23 --max-zip 93000000 >> %L% 2>&1
set RC=%errorlevel%
if %RC%==0 goto packed
if %RC%==3 goto reduce
goto fail
:reduce
set /p DELTA_ARG=<%W%\next_delta_vi23.txt
echo === vuot ngan sach zip, giam delta %DELTA_ARG% >> %L%
goto build
:packed
echo === sub_stats >> %L%
%P% %S%\sub_stats.py %W%\submission_vi23.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_vi23.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_BUILD FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
