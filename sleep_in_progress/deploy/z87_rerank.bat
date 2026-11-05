@echo off
rem Rerank day du 1.200 cau x 200 ung vien bang bge-reranker-v2-m3 tren z87 (uu tien thap, resume theo phan).
rem Batch 16 theo benchmark 06/10: 60,6 cap/s (batch 32: 65,6), VRAM ~2,6GB (batch 32 ~5,5GB). Log: D:\r2ai\work\rerank.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start rerank > work\rerank.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\docs_top200.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\cands_top200.json --out D:\r2ai\work\rerank_top200.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 200 --resume-dir D:\r2ai\work\rr_parts >> D:\r2ai\work\rerank.log 2>&1"
echo [%date% %time%] DONE >> work\rerank.log
