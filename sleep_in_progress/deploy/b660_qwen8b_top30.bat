@echo off
rem 2 bai top-30 tu diem Qwen3-Reranker-8B (ghep 4 shard): (A) q8b = top-30 theo diem 8B,
rem (B) ens8z = ensemble z bge + 8B. Chunk = toan van dung dung 30 doc (chunks_wholedoc --orders rr
rem --budgets 10000000 --doc-cap 40000 --top-docs 30). Log: D:\r2ai\data\sub\qwen8b\make.log
rem Output: D:\r2ai\data\sub\qwen8b\submission_q8b_top30.zip, submission_ens8z_top30.zip
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set W=D:\r2ai\data\sub\qwen8b
set C100=D:\r2ai\data\sub\chunkdocs100
set AA=D:\r2ai\data\sub\cache_pool_rr.json
echo start > %W%\make.log
rem --- A: q8b ---
%P% D:\r2ai\scripts\sort_cache.py --cache %W%\qwen8b_top100.json --out %W%\cache_qwen8b_sorted.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_qwen8b_sorted.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_q8b30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_qwen8b_sorted.json --chunks-from %W%\chunks_q8b30_rr_b10000k.json --out %W%\submission_q8b_top30.json --docs-n 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_q8b_top30.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_q8b_top30.json --zip %W%\submission_q8b_top30.zip --level 9 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_q8b_top30.json >> %W%\make.log 2>&1
rem --- B: ens8z ---
%P% D:\r2ai\scripts\ensemble_cache.py --a %AA% --b %W%\qwen8b_top100.json --mode z --out %W%\cache_ens8z.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_ens8z.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_ens8z30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_ens8z.json --chunks-from %W%\chunks_ens8z30_rr_b10000k.json --out %W%\submission_ens8z_top30.json --docs-n 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_ens8z_top30.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_ens8z_top30.json --zip %W%\submission_ens8z_top30.zip --level 9 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_ens8z_top30.json >> %W%\make.log 2>&1
echo exit done >> %W%\make.log
