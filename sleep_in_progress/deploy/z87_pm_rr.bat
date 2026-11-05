@echo off
rem Rerank top-100 (RRF k60 VI+EN) cua PubMed bang bge-reranker-v2-m3 (truy van VI goc, doc = title+abstract, max_len 512).
rem In: D:\r2ai\work\pm\docs\ + queries_vi.jsonl + pm_rrf_top100.json. Out: pm_rr_top100.json (da sap theo logit).
rem GPU lock: D:\r2ai\locks\gpu.lock (tao dau, xoa cuoi; job khac giu thi doi). Log: D:\r2ai\work\pm\rr.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
set L=D:\r2ai\work\pm\rr.log
:waitlock
if not exist D:\r2ai\locks\gpu.lock goto havelock
findstr /c:"r2ai_pm_rr" D:\r2ai\locks\gpu.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi gpu.lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_pm_rr > D:\r2ai\locks\gpu.lock
echo [%date% %time%] start pm_rr > %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\pubmed_rerank.py --docs D:\r2ai\work\pm\docs --queries D:\r2ai\work\pm\queries_vi.jsonl --cands D:\r2ai\work\pm\pm_rrf_top100.json --out D:\r2ai\work\pm\pm_rr_top100.unsorted.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 100 --resume-dir D:\r2ai\work\pm\rr_parts >> %L% 2>&1"
D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\sort_cache.py --cache D:\r2ai\work\pm\pm_rr_top100.unsorted.json --out D:\r2ai\work\pm\pm_rr_top100.json >> %L% 2>&1
del D:\r2ai\locks\gpu.lock
echo [%date% %time%] DONE >> %L%
echo exit done >> %L%
