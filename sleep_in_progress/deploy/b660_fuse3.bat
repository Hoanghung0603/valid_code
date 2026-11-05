@echo off
rem Hop nhat 3 thu hang (bge pool, Qwen 8B, dense) bang RRF k=60: rrf3 (1,1,1) va rrf3d2 (1,1,2).
rem (1) top30: top-30 hop nhat, chunk toan van 30 doc (missing_cache + export_docs cho doc thieu text).
rem (2) full: n_q = adaptive_n(pool, -2.6, 40, 400), base = n_q doc dau theo hop nhat, chunk d900t015.
rem Log: D:\r2ai\data\sub\diag10\fuse3.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag10
set C100=D:\r2ai\data\sub\chunkdocs100
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set DW=D:\r2ai\data\sub\docwin
set DENSE=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set DOCS=%C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst %W%\dense30_extra.jsonl.zst
set L=%W%\fuse3.log
echo r2ai_fuse3 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
call :one rrf3 1
call :one rrf3d2 2
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
rem %1 = ten, %2 = trong so dense
echo ===== %1 (w dense %2) >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE% --w-bge 1 --w-q8b 1 --w-dense %2 --k 60 --out %W%\cache_%1.json >> %L% 2>&1
echo === overlap %1 vs bge, vs dense >> %L%
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %POOL% --n 30 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %DENSE% --n 30 >> %L% 2>&1
echo === %1 top30: missing_cache >> %L%
%P% %S%\missing_cache.py --cache %W%\cache_%1.json --n 30 --docs %DOCS% --out %W%\cache_%1_30_missing.json >> %L% 2>&1
%P% %S%\export_docs.py --index D:\r2ai\data\index\vi --cache %W%\cache_%1_30_missing.json --n 30 --chars 200000 --no-title --out-docs %W%\%1_30_extra.jsonl.zst --out-cands %W%\%1_30_extra_cands.json >> %L% 2>&1
rem export_docs khong tao file khi khong thieu doc nao: chi them extra neu ton tai
set X=
if exist %W%\%1_30_extra.jsonl.zst set X=%W%\%1_30_extra.jsonl.zst
%P% %S%\chunks_wholedoc.py --docs %DOCS% %X% --cands %W%\cache_%1.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_%1_30 >> %L% 2>&1
%P% %S%\union_sub.py --docs-from %W%\cache_%1.json --chunks-from %W%\chunks_%1_30_rr_b10000k.json --out %W%\submission_%1_top30.json --docs-n 30 >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1_top30.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1_top30.json --zip %W%\submission_%1_top30.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_%1_top30.json >> %L% 2>&1
echo === %1 full >> %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_%1.json --top-ens 1000000 --chunks-from %DW%\chunks_d900t015.json --ref %W%\submission_full8b.json --out %W%\submission_full_%1.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_full_%1.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_full_%1.json --zip %W%\submission_full_%1.zip --level 9 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %POOL% --n 30 --files %W%\submission_%1_top30.zip %W%\submission_full_%1.zip >> %L% 2>&1
goto :eof
