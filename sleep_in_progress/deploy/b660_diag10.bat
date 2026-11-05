@echo off
rem 3 bai chan doan diag10: (A) maxp_top30 = top-30 theo max logit cua so docwin (bge 300/250),
rem (B) q8b_H_top30 + (C) q8b_N_top30 = top-30 cache_qwen8b_sorted chia theo host URL
rem (H = y te chuyen dung, N = bao tong hop / con lai). Chunk = toan van dung dung cac doc do
rem (chunks_wholedoc --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30).
rem Log: D:\r2ai\data\sub\diag10\make.log. Out: D:\r2ai\data\sub\diag10\submission_<ten>.json/.zip
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set W=D:\r2ai\data\sub\diag10
set C100=D:\r2ai\data\sub\chunkdocs100
echo start > %W%\make.log
rem --- A: maxp_top30 ---
%P% D:\r2ai\scripts\maxp_cache.py --docwin D:\r2ai\data\sub\docwin\docwin_top100.json.zst --cands %C100%\cands_chunkdocs100.json --out %W%\cache_maxp.json --bge D:\r2ai\data\sub\cache_pool_rr.json --top 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_maxp.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_maxp30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_maxp.json --chunks-from %W%\chunks_maxp30_rr_b10000k.json --out %W%\submission_maxp_top30.json --docs-n 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_maxp_top30.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_maxp_top30.json --zip %W%\submission_maxp_top30.zip --level 9 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_maxp_top30.json >> %W%\make.log 2>&1
rem --- host split ---
%P% D:\r2ai\scripts\host_split.py --cache D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json --links D:\r2ai\data\links_corpus.parquet --out-h %W%\cache_q8b_H.json --out-n %W%\cache_q8b_N.json --report %W%\host_report.txt --top 30 >> %W%\make.log 2>&1
rem --- B: q8b_H_top30 ---
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_q8b_H.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_q8bH30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_q8b_H.json --chunks-from %W%\chunks_q8bH30_rr_b10000k.json --out %W%\submission_q8b_H_top30.json --docs-n 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_q8b_H_top30.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_q8b_H_top30.json --zip %W%\submission_q8b_H_top30.zip --level 9 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_q8b_H_top30.json >> %W%\make.log 2>&1
rem --- C: q8b_N_top30 ---
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %W%\cache_q8b_N.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_q8bN30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\cache_q8b_N.json --chunks-from %W%\chunks_q8bN30_rr_b10000k.json --out %W%\submission_q8b_N_top30.json --docs-n 30 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_q8b_N_top30.json >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_q8b_N_top30.json --zip %W%\submission_q8b_N_top30.zip --level 9 >> %W%\make.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_q8b_N_top30.json >> %W%\make.log 2>&1
rem --- sha256 3 zip ---
certutil -hashfile %W%\submission_maxp_top30.zip SHA256 >> %W%\make.log 2>&1
certutil -hashfile %W%\submission_q8b_H_top30.zip SHA256 >> %W%\make.log 2>&1
certutil -hashfile %W%\submission_q8b_N_top30.zip SHA256 >> %W%\make.log 2>&1
echo exit done >> %W%\make.log
