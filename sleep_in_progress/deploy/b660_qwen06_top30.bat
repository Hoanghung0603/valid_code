@echo off
rem Bai nop chan doan "qwen06 top-30" (doi chung voi #4872): relevant_docs = 30 doc dau theo diem Qwen,
rem chunk = toan van dung dung 30 doc do (chunks_wholedoc --orders rr --budgets 10000000 --doc-cap 40000).
rem Cache goc: qwen06_top100.json (Qwen3-Reranker-0.6B, z87). Log: D:\r2ai\data\sub\qwen06\make.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set W=D:\r2ai\data\sub\qwen06
set C100=D:\r2ai\data\sub\chunkdocs100
echo start > %W%\make.log
%P% D:\r2ai\scripts\sort_cache.py --cache %W%\qwen06_top100.json --out %W%\cache_qwen06_sorted.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_qwen06_sorted.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_q30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_qwen06_sorted.json --chunks-from %W%\chunks_q30_rr_b10000k.json --out %W%\submission_qwen06_top30.json --docs-n 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_qwen06_top30.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_qwen06_top30.json --zip %W%\submission_qwen06_top30.zip --level 9 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_qwen06_top30.json >> %W%\make.log 2>&1
echo exit done >> %W%\make.log
