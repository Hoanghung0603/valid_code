@echo off
rem Embed ~112k doc top-200 + 1.200 cau bang bge-m3 tren z87 (GPU only, uu tien thap, resume theo khoi). Log: D:\r2ai\work\embed.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start embed >> work\embed.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\embed_run.py --docs D:\r2ai\work\docs_top200.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\cands_top200.json --work D:\r2ai\work\emb --out D:\r2ai\work\dense_top200.json --doc-len 512 --batch 16 >> D:\r2ai\work\embed.log 2>&1"
echo [%date% %time%] DONE >> work\embed.log
