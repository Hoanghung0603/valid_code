@echo off
rem vi23 buoc 11 (b660, CPU): gop cands + docwin wins (tai dung cap p3 + cap moi) -> chunk U900
rem (chunks_from_docwin --units 900, giu day du ung vien) -> gop voi chunk_dw_u900_ext thanh
rem chunk_dw_u900_v23.json.zst + kiem nguyen van 100%. Log: diag23\chunk.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set O=%W%\docwin_vi23
set DX=D:\r2ai\data\sub\docwin_ext
set L=%W%\chunk.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_chunk" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_chunk > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === wins-merge (p3reuse + moi) >> %L%
%P% %S%\vi23.py wins-merge --cands-a %O%\cands_docwin_p3reuse.json --docwin-a D:\r2ai\data\sub\diag18\docwin_p3\docwin_p3.json.zst --cands-b %O%\cands_docwin_vi23.json --docwin-b %W%\docwin_vi23.json.zst --out-cands %O%\cands_docwin_v23all.json --out-docwin %O%\docwin_v23all.json --report %O%\wins_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
set DOCS=D:\r2ai\data\sub\diag18\docwin_p3\docwin_p3.part000.jsonl.zst D:\r2ai\data\sub\diag18\docwin_p3\docwin_p3.part001.jsonl.zst
for %%f in (%O%\docs_vi23_dw_old*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
for %%f in (%O%\docs_vi23_dw_new*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
%P% D:\r2ai\vi23_chunkcheck.py coverage --cands %O%\cands_docwin_v23all.json --docwin %O%\docwin_v23all.json --docs %DOCS% --report %O%\coverage_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === chunks_from_docwin U900 >> %L%
%P% %S%\chunks_from_docwin.py --docwin %O%\docwin_v23all.json --docs %DOCS% --cands %O%\cands_docwin_v23all.json --units 900 --top-chunks 100000 --top-docs 500 --out-dir %O% >> %L% 2>&1
if errorlevel 1 goto fail
echo === dwx_merge voi chunk_dw_u900_ext >> %L%
%P% %S%\dwx_merge.py --old %DX%\chunk_dw_u900_ext.json.zst --new-dir %O% --docs-new %DOCS% --docs-old D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part000.jsonl.zst D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part001.jsonl.zst %DX%\docwin_ext.part000.jsonl.zst %DX%\docwin_ext.part001.jsonl.zst --fused %W%\cache_rrf3d2_v23.json --ref-chunks D:\r2ai\data\sub\diag17\submission_cx130cov.json --dq-cands %O%\cands_docwin_v23all.json --pool D:\r2ai\data\sub\cache_pool_rr.json --new-out %O%\chunk_dw_u900_v23_new.json.zst --out %W%\chunk_dw_u900_v23.json.zst --report %W%\dwx_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% D:\r2ai\vi23_chunkcheck.py verbatim --report %W%\dwx_merge_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sha256 >> %L%
certutil -hashfile %W%\chunk_dw_u900_v23.json.zst SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_CHUNK FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
