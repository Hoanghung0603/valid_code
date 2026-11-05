@echo off
rem Phan B1 buoc 1 (b660, CPU): (q, doc) cua 3 doclist (t20/t16/nq) chua co ung vien chunk trong
rem chunk_dw_u900_ext (ngoai top-100 bge cache_pool_rr U relevant_docs submission_split) -> union
rem cands_docwin_p3.json + cache_docwin_p3.json, xuat text (export_docs --chars 200000 --no-title).
rem Out D:\r2ai\data\sub\diag18\docwin_p3\. Log: dwx_p3_export.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag18
set O=%W%\docwin_p3
set L=%O%\dwx_p3_export.log
if not exist %O% mkdir %O%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_dwx_p3_exp" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_dwx_p3_exp > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === pairs 3 doclist >> %L%
%P% %S%\dwx_p3.py pairs --doclist %W%\doclist_p3_t20.json --doclist %W%\doclist_p3_t16.json --doclist %W%\doclist_p3_nq.json --pool D:\r2ai\data\sub\cache_pool_rr.json --split D:\r2ai\data\sub\diag12\submission_split.json --idx-from %W%\cache_pool3_rr.json %W%\cache_fhyde3.json --new %W%\new_cands.json --out-dir %O% >> %L% 2>&1
if errorlevel 1 goto fail
echo === export_docs >> %L%
%P% %S%\export_docs.py --index D:\r2ai\data\index\vi --cache %O%\cache_docwin_p3.json --n 1000 --chars 200000 --no-title --part-mb 80 --out-docs %O%\docwin_p3.jsonl.zst --out-cands %O%\cands_from_export.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sha256 >> %L%
for %%f in (%O%\docwin_p3.part*.jsonl.zst) do certutil -hashfile %%f SHA256 >> %L% 2>&1
certutil -hashfile %O%\cands_docwin_p3.json SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo DWX_P3_EXPORT FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
