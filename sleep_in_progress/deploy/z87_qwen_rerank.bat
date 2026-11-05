@echo off
rem Qwen3-Reranker-0.6B day du 1.200 cau x top-100 doc, max-len 1024, batch 8, fp16 (uu tien thap,
rem resume theo phan 50 cau; bench da chay phan 0). Log append: D:\r2ai\work\qrr06\qrr06.log
rem Out: D:\r2ai\work\qrr06\qwen06_top100.json  Resume: D:\r2ai\work\qrr06\parts  Watchdog mac dinh bat.
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set PYTHONPATH=D:\r2ai\scripts
cd /d D:\r2ai
echo [%date% %time%] start qrr06 >> work\qrr06\qrr06.log
start "" /low /wait cmd /c "D:\r2ai\venv_qwen\Scripts\python.exe D:\r2ai\work\qrr06\qwen_rerank_run.py --docs D:\r2ai\work\docwin\chunkdocs100.part000.jsonl.zst D:\r2ai\work\docwin\chunkdocs100.part001.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\docwin\cands_chunkdocs100.json --out D:\r2ai\work\qrr06\qwen06_top100.json --top 100 --max-len 1024 --batch 8 --resume-dir D:\r2ai\work\qrr06\parts >> work\qrr06\qrr06.log 2>&1"
echo [%date% %time%] exit done errorlevel %errorlevel% >> work\qrr06\qrr06.log
