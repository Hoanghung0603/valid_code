@echo off
rem Docwin mo rong buoc 1 (b660, CPU): tinh D_q (doc submission_split nam NGOAI top-100 bge cua
rem cache_pool_rr), xuat text D_q (export_docs.py --chars 200000 --no-title, part <= 80MB).
rem Out D:\r2ai\data\sub\docwin_ext\: cands_docwin_ext.json (thu tu NOP = doc_rank), cache_docwin_ext.json,
rem docwin_ext.partNNN.jsonl.zst, dwx_cands_report.json. Log: dwx_export.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set W=D:\r2ai\data\sub\docwin_ext
set L=%W%\dwx_export.log
if not exist %W% mkdir %W%
echo r2ai_dwx_exp > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\dwx_cands.py --split D:\r2ai\data\sub\diag12\submission_split.json --pool D:\r2ai\data\sub\cache_pool_rr.json --fhyde D:\r2ai\data\sub\diag12\cache_fhyde.json --out-dir %W% >> %L% 2>&1
echo exit_cands=%ERRORLEVEL% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\export_docs.py --index D:\r2ai\data\index\vi --cache %W%\cache_docwin_ext.json --n 1000 --chars 200000 --no-title --part-mb 80 --out-docs %W%\docwin_ext.jsonl.zst --out-cands %W%\cands_from_export.json >> %L% 2>&1
echo exit_export=%ERRORLEVEL% >> %L%
echo === sha256 >> %L%
for %%f in (%W%\docwin_ext.part*.jsonl.zst) do certutil -hashfile %%f SHA256 >> %L% 2>&1
certutil -hashfile %W%\cands_docwin_ext.json SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
