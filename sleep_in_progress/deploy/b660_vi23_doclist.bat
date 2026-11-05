@echo off
rem vi23 buoc 9 (b660, CPU): doclist (n_q theo logit bge pool_v23, nen = n_q doc dau fhyde_v23, them doc
rem nguon chunk cua chunks_rrf3d2_cf4 - luat full8b); tim cap (q, doc) chua co ung vien chunk can docwin
rem (tai dung cap da chon docwin_p3); xuat text: doc cu (index cu, bp cu) + doc moi (bp23).
rem Log: diag23\doclist.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set O=%W%\docwin_vi23
set L=%W%\doclist.log
if not exist %O% mkdir %O%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_doclist" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_doclist > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === doclist >> %L%
%P% %S%\vi23.py doclist --pool %W%\cache_pool_v23.json --fhyde %W%\cache_fhyde_v23.json --ref-chunks D:\r2ai\data\sub\diag11\chunks_rrf3d2_cf4.json --thr -2.0 --lo 40 --hi 400 --out %W%\doclist_v23.json --report %W%\doclist_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === docwin-cands >> %L%
%P% %S%\vi23.py docwin-cands --doclist %W%\doclist_v23.json --pool D:\r2ai\data\sub\cache_pool_rr.json --split D:\r2ai\data\sub\diag12\submission_split.json --p3-docwin D:\r2ai\data\sub\diag18\docwin_p3\docwin_p3.json.zst --p3-cands D:\r2ai\data\sub\diag18\docwin_p3\cands_docwin_p3.json --new-ids %W%\new_ids.json --idx-from %W%\cache_pool_v23.json %W%\cache_fhyde_v23.json %W%\new_idx.json --top-n 100 --out-dir %O% >> %L% 2>&1
if errorlevel 1 goto fail
echo === export text doc cu thieu docwin (index cu + bp cu) >> %L%
%P% %S%\export_docs.py --index D:\r2ai\data\index\vi --cache %O%\cache_docwin_old.json --n 1000 --chars 200000 --no-title --part-mb 80 --out-docs %O%\docs_vi23_dw_old.jsonl.zst --out-cands %O%\cands_from_export_old.json >> %L% 2>&1
echo === export text doc moi (bp23) >> %L%
%P% %S%\vi23.py export-new --shards %W%\shards --bp D:\r2ai\data\index\vi23\boilerplate.json --idx %W%\new_idx.json --mode docwin --out %O%\docs_vi23_dw_new.jsonl.zst --report %O%\export_dw_new_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sha256 >> %L%
certutil -hashfile %O%\cands_docwin_vi23.json SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_DOCLIST FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
