@echo off
rem Chay lai rieng phan top30 cua b660_fuse3.bat (lan 1 loi vi export_docs khong tao file khi 0 doc thieu).
rem Cache cache_rrf3.json / cache_rrf3d2.json da co tu lan 1. Log: D:\r2ai\data\sub\diag10\fuse3_top30.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag10
set C100=D:\r2ai\data\sub\chunkdocs100
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set DOCS=%C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst %W%\dense30_extra.jsonl.zst
set L=%W%\fuse3_top30.log
echo r2ai_fuse3_top30 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
call :one rrf3
call :one rrf3d2
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
echo ===== %1 top30 >> %L%
set X=
if exist %W%\%1_30_extra.jsonl.zst set X=%W%\%1_30_extra.jsonl.zst
%P% %S%\chunks_wholedoc.py --docs %DOCS% %X% --cands %W%\cache_%1.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_%1_30 >> %L% 2>&1
%P% %S%\union_sub.py --docs-from %W%\cache_%1.json --chunks-from %W%\chunks_%1_30_rr_b10000k.json --out %W%\submission_%1_top30.json --docs-n 30 >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1_top30.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1_top30.json --zip %W%\submission_%1_top30.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_%1_top30.json >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %POOL% --n 30 --files %W%\submission_%1_top30.zip %W%\submission_full_%1.zip >> %L% 2>&1
goto :eof
