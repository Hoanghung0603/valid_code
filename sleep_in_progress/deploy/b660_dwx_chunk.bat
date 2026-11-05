@echo off
rem Buoc 4-5 (b660, CPU): chunk U900 tu docwin_ext (doc_rank = hang trong relevant_docs submission_split),
rem gop voi chunk cu top-120 (doc_rank = hang top-100 cache_pool_rr) -> chunk_dw_u900_ext.json.zst
rem (KHONG cat top-120) + thong ke logit + uoc impact s = zc + 4*zd. Log: dwx_chunk.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set W=D:\r2ai\data\sub\docwin_ext
set L=%W%\dwx_chunk.log
echo r2ai_dwx_chunk > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
set DOCS=
for %%f in (%W%\docwin_ext.part*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\chunks_from_docwin.py --docwin %W%\docwin_ext.json.zst --docs %DOCS% --cands %W%\cands_docwin_ext.json --units 900 --top-chunks 100000 --top-docs 500 --out-dir %W% >> %L% 2>&1
echo exit_chunks=%ERRORLEVEL% >> %L%
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\dwx_merge.py --old D:\r2ai\data\sub\docwin\chunk_dw_u900.json.zst --new-dir %W% --docs-new %DOCS% --docs-old D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part000.jsonl.zst D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part001.jsonl.zst --fused D:\r2ai\data\sub\diag10\cache_rrf3d2.json --ref-chunks D:\r2ai\data\sub\diag12\submission_split.json --dq-cands %W%\cands_docwin_ext.json --pool D:\r2ai\data\sub\cache_pool_rr.json --new-out %W%\chunk_dw_u900_dq.json.zst --out %W%\chunk_dw_u900_ext.json.zst --report %W%\dwx_merge_report.json >> %L% 2>&1
echo exit_merge=%ERRORLEVEL% >> %L%
echo === sha256 >> %L%
certutil -hashfile %W%\chunk_dw_u900_ext.json.zst SHA256 >> %L% 2>&1
certutil -hashfile %W%\chunk_dw_u900_dq.json.zst SHA256 >> %L% 2>&1
for %%f in (%W%\chunk_dw_u900.json*) do certutil -hashfile %%f SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
