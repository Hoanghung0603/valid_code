@echo off
rem Embed toan kho VI (~613k doc, tai dung blk cu trong work\emb) + 1.200 cau bang bge-m3 tren z87
rem (GPU only, uu tien thap, resume theo khoi). Log: D:\r2ai\work\embed_all.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start embed_all >> work\embed_all.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\embed_all.py --docs D:\r2ai\work\all --docs-old D:\r2ai\work\docs_top200.jsonl.zst --queries D:\r2ai\work\queries.jsonl --work-old D:\r2ai\work\emb --work D:\r2ai\work\emb_all --out D:\r2ai\work\dense_full_top200.json --doc-len 512 --query-len 256 --batch 16 --top 200 >> D:\r2ai\work\embed_all.log 2>&1"
echo [%date% %time%] DONE >> work\embed_all.log
