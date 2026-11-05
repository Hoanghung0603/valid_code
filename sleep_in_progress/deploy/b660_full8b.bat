@echo off
rem (1) full8b: n_q thich nghi theo logit bge (thr -2.6, [40,400]), base = ens8z top-100 + pool, chunks d900t015.
rem (2) dense_top30: export text doc thieu cua dense top-30, chunk toan van 30 doc, docs-n 30, overlap voi bge.
rem Log: D:\r2ai\data\sub\diag10\full8b.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag10
set C100=D:\r2ai\data\sub\chunkdocs100
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set ENS=D:\r2ai\data\sub\qwen8b\cache_ens8z.json
set DW=D:\r2ai\data\sub\docwin
set DENSE=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set L=%W%\full8b.log
echo r2ai_full8b > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
rem --- 1: full8b ---
echo === full8b_sub >> %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %ENS% --chunks-from %DW%\chunks_d900t015.json --ref %DW%\submission_d900t015.json --out %W%\submission_full8b.json >> %L% 2>&1
echo === validate full8b >> %L%
%P% %S%\evaluate.py validate %W%\submission_full8b.json >> %L% 2>&1
echo === zip full8b >> %L%
%P% %S%\zip_submission.py --json %W%\submission_full8b.json --zip %W%\submission_full8b.zip --level 9 >> %L% 2>&1
rem --- 2: dense_top30 ---
echo === missing_cache >> %L%
%P% %S%\missing_cache.py --cache %DENSE% --n 30 --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --out %W%\cache_dense30_missing.json >> %L% 2>&1
echo === export_docs >> %L%
%P% %S%\export_docs.py --index D:\r2ai\data\index\vi --cache %W%\cache_dense30_missing.json --n 30 --chars 200000 --no-title --out-docs %W%\dense30_extra.jsonl.zst --out-cands %W%\dense30_extra_cands.json >> %L% 2>&1
echo === chunks_wholedoc >> %L%
%P% %S%\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst %W%\dense30_extra.jsonl.zst --cands %DENSE% --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_dense30 >> %L% 2>&1
echo === union_sub >> %L%
%P% %S%\union_sub.py --docs-from %DENSE% --chunks-from %W%\chunks_dense30_rr_b10000k.json --out %W%\submission_dense_top30.json --docs-n 30 >> %L% 2>&1
echo === validate dense >> %L%
%P% %S%\evaluate.py validate %W%\submission_dense_top30.json >> %L% 2>&1
echo === zip dense >> %L%
%P% %S%\zip_submission.py --json %W%\submission_dense_top30.json --zip %W%\submission_dense_top30.zip --level 9 >> %L% 2>&1
echo === sub_stats dense >> %L%
%P% %S%\sub_stats.py %W%\submission_dense_top30.json >> %L% 2>&1
echo === report overlap >> %L%
%P% %S%\bm25_chunk.py report --cache %DENSE% --ref %POOL% --n 30 --files %W%\submission_full8b.zip %W%\submission_dense_top30.zip >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
