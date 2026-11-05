@echo off
rem Smoke test pipeline PubMed tren z87 (du lieu gia nho): embed + topk + RRF + rerank + sort. Kiem bug truoc khi chay that.
rem In: D:\r2ai\work\pm_smoke\ (docs\, queries_vi.jsonl, queries_en.jsonl). Log: D:\r2ai\work\pm_smoke\smoke.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
set L=D:\r2ai\work\pm_smoke\smoke.log
if not exist D:\r2ai\work\pm_smoke\emb mkdir D:\r2ai\work\pm_smoke\emb
:waitlock
if not exist D:\r2ai\locks\gpu.lock goto havelock
findstr /c:"r2ai_pm_smoke" D:\r2ai\locks\gpu.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi gpu.lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_pm_smoke > D:\r2ai\locks\gpu.lock
echo [%date% %time%] start smoke > %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\pubmed_embed.py --docs D:\r2ai\work\pm_smoke\docs --queries-vi D:\r2ai\work\pm_smoke\queries_vi.jsonl --queries-en D:\r2ai\work\pm_smoke\queries_en.jsonl --work D:\r2ai\work\pm_smoke\emb --out-vi D:\r2ai\work\pm_smoke\vi.json --out-en D:\r2ai\work\pm_smoke\en.json --out-rrf D:\r2ai\work\pm_smoke\rrf.json --top 20 --batch 8 >> %L% 2>&1"
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\pubmed_rerank.py --docs D:\r2ai\work\pm_smoke\docs --queries D:\r2ai\work\pm_smoke\queries_vi.jsonl --cands D:\r2ai\work\pm_smoke\rrf.json --out D:\r2ai\work\pm_smoke\rr.unsorted.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 8 --top 20 --resume-dir D:\r2ai\work\pm_smoke\rr_parts >> %L% 2>&1"
D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\sort_cache.py --cache D:\r2ai\work\pm_smoke\rr.unsorted.json --out D:\r2ai\work\pm_smoke\rr.json >> %L% 2>&1
del D:\r2ai\locks\gpu.lock
echo [%date% %time%] DONE >> %L%
echo exit done >> %L%
