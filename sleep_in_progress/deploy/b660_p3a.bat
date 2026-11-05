@echo off
rem Phan A (diag18): submission_p3t20_c130 / p3t16_c130 / p3nq_c130 - doc = doclist_p3
rem (+ doc nguon chunk cx130cov con thieu, noi cuoi), chunk GIU Y HET submission_cx130cov.
rem p3nq: top-n_q theo thu hang cache_fhyde3, n_q = so doc cua cau do trong submission_split.
rem Validate + zip LZMA + sha256. Log: D:\r2ai\data\sub\diag18\p3a.log
rem Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag18
set W12=D:\r2ai\data\sub\diag12
set W17=D:\r2ai\data\sub\diag17
set L=%W%\p3a.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_p3a" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_p3a > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === mknq doclist_p3_nq >> %L%
%P% %S%\cxp3.py mknq --rank %W%\cache_fhyde3.json --split %W12%\submission_split.json --new %W%\new_cands.json --out %W%\doclist_p3_nq.json --report %W%\doclist_p3_nq_report.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === swapdoc 3 bai >> %L%
%P% %S%\cxp3.py swapdoc --cx130 %W17%\submission_cx130cov.json --new %W%\new_cands.json --doclist submission_p3t20_c130=%W%\doclist_p3_t20.json --doclist submission_p3t16_c130=%W%\doclist_p3_t16.json --doclist submission_p3nq_c130=%W%\doclist_p3_nq.json --outdir %W% >> %L% 2>&1
if errorlevel 1 goto fail

for %%n in (submission_p3t20_c130 submission_p3t16_c130 submission_p3nq_c130) do (
  echo === validate %%n >> %L%
  %P% %S%\evaluate.py validate %W%\%%n.json >> %L% 2>&1
  if errorlevel 1 goto fail
  echo === zip %%n >> %L%
  %P% %S%\zip_submission.py --json %W%\%%n.json --zip %W%\%%n.zip --method lzma >> %L% 2>&1
  if errorlevel 1 goto fail
)
echo === sub_stats >> %L%
%P% %S%\sub_stats.py %W%\submission_p3t20_c130.json %W%\submission_p3t16_c130.json %W%\submission_p3nq_c130.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_p3t20_c130.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_p3t16_c130.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_p3nq_c130.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo P3A FAILED, khong nop bai >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
