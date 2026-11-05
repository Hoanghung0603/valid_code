@echo off
rem Embed doc PubMed + truy van VI/EN bang bge-m3 (GPU only, uu tien thap), top-100 (a) VI, (b) EN, (c) RRF k60.
rem Input: D:\r2ai\work\pm\docs\docs_pm.partNNN.jsonl.zst + queries_vi.jsonl + queries_en.jsonl
rem Out: pm_dense_vi_top100.json, pm_dense_en_top100.json, pm_rrf_top100.json. Resume theo khoi blk_*.npz.
rem GPU lock: D:\r2ai\locks\gpu.lock (tao dau, xoa cuoi; job khac giu thi doi). Log: D:\r2ai\work\pm\embed.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
set L=D:\r2ai\work\pm\embed.log
if not exist D:\r2ai\work\pm mkdir D:\r2ai\work\pm
if not exist D:\r2ai\locks mkdir D:\r2ai\locks
:waitlock
if not exist D:\r2ai\locks\gpu.lock goto havelock
findstr /c:"r2ai_pm_embed" D:\r2ai\locks\gpu.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi gpu.lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_pm_embed > D:\r2ai\locks\gpu.lock
echo [%date% %time%] start pm_embed > %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\pubmed_embed.py --docs D:\r2ai\work\pm\docs --queries-vi D:\r2ai\work\pm\queries_vi.jsonl --queries-en D:\r2ai\work\pm\queries_en.jsonl --work D:\r2ai\work\pm\emb --out-vi D:\r2ai\work\pm\pm_dense_vi_top100.json --out-en D:\r2ai\work\pm\pm_dense_en_top100.json --out-rrf D:\r2ai\work\pm\pm_rrf_top100.json --doc-len 512 --query-len 256 --batch 16 --top 100 --chunk 131072 >> %L% 2>&1"
del D:\r2ai\locks\gpu.lock
echo [%date% %time%] DONE >> %L%
echo exit done >> %L%
