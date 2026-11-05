@echo off
rem Phan B1 buoc 3 (b660, CPU): chunk U900 tu docwin_p3 (doc_rank = hang trong cands_docwin_p3),
rem gop voi ung vien cu chunk_dw_u900_ext -> chunk_dw_u900_p3.json.zst (KHONG cat) + verify
rem 100% nguyen van (cach cu + moi) + thong ke. Log: dwx_p3_chunk.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag18
set O=%W%\docwin_p3
set W12=D:\r2ai\data\sub\diag12
set W17=D:\r2ai\data\sub\diag17
set DX=D:\r2ai\data\sub\docwin_ext
set C100=D:\r2ai\data\sub\chunkdocs100
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%O%\dwx_p3_chunk.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_dwx_p3_chunk" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_dwx_p3_chunk > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

set DOCS=
for %%f in (%O%\docwin_p3.part*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
echo === chunks_from_docwin U900 >> %L%
%P% %S%\chunks_from_docwin.py --docwin %O%\docwin_p3.json.zst --docs %DOCS% --cands %O%\cands_docwin_p3.json --units 900 --top-chunks 100000 --top-docs 500 --out-dir %O% >> %L% 2>&1
if errorlevel 1 goto fail
echo === dwx_merge gop voi chunk_dw_u900_ext >> %L%
%P% %S%\dwx_merge.py --old %DX%\chunk_dw_u900_ext.json.zst --new-dir %O% --docs-new %DOCS% --docs-old %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst %DX%\docwin_ext.part000.jsonl.zst %DX%\docwin_ext.part001.jsonl.zst --fused D:\r2ai\data\sub\diag10\cache_rrf3d2.json --ref-chunks %W17%\submission_cx130cov.json --dq-cands %O%\cands_docwin_p3.json --pool %POOL% --new-out %O%\chunk_dw_u900_p3_new.json.zst --out %W%\chunk_dw_u900_p3.json.zst --report %W%\dwx_p3_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sha256 >> %L%
certutil -hashfile %W%\chunk_dw_u900_p3.json.zst SHA256 >> %L% 2>&1
certutil -hashfile %O%\chunk_dw_u900_p3_new.json.zst SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo DWX_P3_CHUNK FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
