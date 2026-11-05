@echo off
rem Chan doan precision dau bang (08/10): relevant_docs = top-k doc theo rerank (cache_pool_rr), chunk = toan van dung k doc do.
rem Docs P = P@k cua doc; Chunks P = ti le doc trong top-k co chua chunk vang. So voi #4844 (Chunks P 0,63).
rem Log: D:\r2ai\data\sub\wd\make_topk.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set WD=D:\r2ai\data\sub\wd
set C100=D:\r2ai\data\sub\chunkdocs100
echo start > %WD%\make_topk.log
for %%K in (10 30) do (
  echo ==== top%%K ==== >> %WD%\make_topk.log
  %P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %C100%\cands_chunkdocs100.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs %%K --out-prefix %WD%\chunks_top%%K >> %WD%\make_topk.log 2>&1
  %P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %WD%\chunks_top%%K_rr_b10000k.json --out %WD%\submission_top%%K.json --docs-n %%K >> %WD%\make_topk.log 2>&1
  %P% D:\r2ai\scripts\evaluate.py validate %WD%\submission_top%%K.json >> %WD%\make_topk.log 2>&1
  %P% D:\r2ai\scripts\zip_submission.py --json %WD%\submission_top%%K.json --zip %WD%\submission_top%%K.zip --level 9 >> %WD%\make_topk.log 2>&1
)
%P% D:\r2ai\scripts\sub_stats.py %WD%\submission_top10.json %WD%\submission_top30.json >> %WD%\make_topk.log 2>&1
echo exit done >> %WD%\make_topk.log
