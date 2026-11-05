@echo off
rem Ensemble 2 ranker bge (--a cache_pool_rr) + Qwen0.6B (--b qwen06_top100) top-30, 2 bien the:
rem rrf (k 60) va z. Moi bien the: ensemble -> chunks_wholedoc toan van 30 doc -> union_sub -> validate
rem -> zip level 9 -> sub_stats. Log: D:\r2ai\data\sub\qwen06\ens.log
rem Output: D:\r2ai\data\sub\qwen06\submission_ens_rrf_top30.zip, submission_ens_z_top30.zip
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set W=D:\r2ai\data\sub\qwen06
set C100=D:\r2ai\data\sub\chunkdocs100
set AA=D:\r2ai\data\sub\cache_pool_rr.json
echo start > %W%\ens.log
%P% D:\r2ai\scripts\ensemble_cache.py --a %AA% --b %W%\qwen06_top100.json --mode rrf --k 60 --out %W%\cache_ens_rrf.json >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_ens_rrf.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_ens_rrf >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_ens_rrf.json --chunks-from %W%\chunks_ens_rrf_rr_b10000k.json --out %W%\submission_ens_rrf_top30.json --docs-n 30 >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_ens_rrf_top30.json >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_ens_rrf_top30.json --zip %W%\submission_ens_rrf_top30.zip --level 9 >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_ens_rrf_top30.json >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\ensemble_cache.py --a %AA% --b %W%\qwen06_top100.json --mode z --out %W%\cache_ens_z.json >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_ens_z.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_ens_z >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_ens_z.json --chunks-from %W%\chunks_ens_z_rr_b10000k.json --out %W%\submission_ens_z_top30.json --docs-n 30 >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_ens_z_top30.json >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_ens_z_top30.json --zip %W%\submission_ens_z_top30.zip --level 9 >> %W%\ens.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_ens_z_top30.json >> %W%\ens.log 2>&1
echo exit done >> %W%\ens.log
