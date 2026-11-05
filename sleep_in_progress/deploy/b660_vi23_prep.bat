@echo off
rem vi23 buoc 0 (b660, CPU): copy goi VI moi ra ngoai thu muc crawl (CHI DOC), kiem sha256 manifest,
rem loc doc (status 200 / text >= 200 / detect_lang vi / trung id / id trong missing / khong trung id cu),
rem ghep boilerplate (giu host cu, fit host moi) -> boilerplate.json cua index vi23.
rem Out: D:\r2ai\data\sub\diag23\{prep_report.json,new_ids.json}, D:\r2ai\data\index\vi23\boilerplate.json
rem Log: D:\r2ai\data\sub\diag23\prep.log. Lock heavy_job.lock (ngan).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set SRC=D:\r2ai\data\crawl\handoff_vi_20261017
set SH=%W%\shards
set L=%W%\prep.log
if not exist %W% mkdir %W%
if not exist %SH% mkdir %SH%
if not exist D:\r2ai\data\index\vi23 mkdir D:\r2ai\data\index\vi23

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_prep" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_prep > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === copy shard ra ngoai (nguon CHI DOC) >> %L%
copy /y %SRC%\vi-recov-00000.jsonl.zst %SH%\vi-recov-00000.jsonl.zst >> %L% 2>&1
copy /y %SRC%\vi-recov-00001.jsonl.zst %SH%\vi-recov-00001.jsonl.zst >> %L% 2>&1
copy /y %SRC%\vi-recov-00002.jsonl.zst %SH%\vi-recov-00002.jsonl.zst >> %L% 2>&1
copy /y %SRC%\manifest.sha256 %SH%\manifest.sha256 >> %L% 2>&1
echo === kiem manifest >> %L%
%P% %S%\vi23.py manifest --shards %SH% --report %W%\manifest_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === prep >> %L%
%P% %S%\vi23.py prep --shards %SH% --old-bp D:\r2ai\data\index\vi\boilerplate.json --missing D:\r2ai\data\sub\diag21\missing_vi_by_host.json --old-ids D:\r2ai\data\index\vi\ids.npy --out-dir %W% --bp-out D:\r2ai\data\index\vi23 >> %L% 2>&1
if errorlevel 1 goto fail
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_PREP FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
