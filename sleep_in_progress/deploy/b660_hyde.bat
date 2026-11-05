@echo off
rem HyDE / reason (truy van = cau goc + doan LLM sinh, dense bge-m3 toan kho top-200).
rem (1) top30: doc = top-30 cua nguon, chunk = toan van 30 doc (so voi #5006 dense thuan P@30 0,340).
rem (2) full: rrf3d2 + nguon moi x1 (khe nguon thu 4 cua fuse3.py), cong thuc #5248 (T -2.0, wd 4).
rem Log: D:\r2ai\data\sub\diag12\hyde.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W10=D:\r2ai\data\sub\diag10
set W11=D:\r2ai\data\sub\diag11
set W=D:\r2ai\data\sub\diag12
set H=D:\r2ai\data\sub\hyde
set C100=D:\r2ai\data\sub\chunkdocs100
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set DENSE=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set DOCS=%C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst %W10%\dense30_extra.jsonl.zst
set L=%W%\hyde.log
if not exist %W% mkdir %W%
echo r2ai_hyde > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
call :one hyde
call :one reason
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
set SRC=%H%\dense_%1_top200.json
echo ===== %1 overlap top30 >> %L%
%P% %S%\bm25_chunk.py report --cache %SRC% --ref %DENSE% --n 30 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %SRC% --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
echo === %1 top30 >> %L%
%P% %S%\missing_cache.py --cache %SRC% --n 30 --docs %DOCS% --out %W%\cache_%1_30_missing.json >> %L% 2>&1
%P% %S%\export_docs.py --index D:\r2ai\data\index\vi --cache %W%\cache_%1_30_missing.json --n 30 --chars 200000 --no-title --out-docs %W%\%1_30_extra.jsonl.zst --out-cands %W%\%1_30_extra_cands.json >> %L% 2>&1
set X=
if exist %W%\%1_30_extra.jsonl.zst set X=%W%\%1_30_extra.jsonl.zst
%P% %S%\chunks_wholedoc.py --docs %DOCS% %X% --cands %SRC% --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_%1_30 >> %L% 2>&1
%P% %S%\union_sub.py --docs-from %SRC% --chunks-from %W%\chunks_%1_30_rr_b10000k.json --out %W%\submission_%1_top30.json --docs-n 30 >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1_top30.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1_top30.json --zip %W%\submission_%1_top30.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_%1_top30.json >> %L% 2>&1
echo === %1 full fused >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE% --w-bge 1 --w-q8b 1 --w-dense 2 --bm25 %SRC% --w-bm25 1 --bm25-n 200 --k 60 --out %W%\cache_f%1.json >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_f%1.json --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_f%1.json --ref-chunks %DW%\chunks_d900t015.json --wd 4 --compare-with %W11%\chunks_rrf3d2_cf4.json --out %W%\chunks_f%1_cf4.json >> %L% 2>&1
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_f%1.json --top-ens 1000000 --thr -2.0 --chunks-from %W%\chunks_f%1_cf4.json --ref %W11%\submission_t20_cf4.json --out %W%\submission_f%1.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_f%1.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_f%1.json --zip %W%\submission_f%1.zip --level 9 >> %L% 2>&1
goto :eof
