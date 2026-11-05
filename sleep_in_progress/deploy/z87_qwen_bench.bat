@echo off
rem Benchmark Qwen3-Reranker-0.6B: 50 cau dau (phan 0) x top-100 doc, max-len 1024, batch 8, fp16.
rem Ghi vao cung resume dir de job that dung lai. Log: D:\r2ai\work\qrr06\bench.log
rem Out: D:\r2ai\work\qrr06\bench_out.json  Resume: D:\r2ai\work\qrr06\parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set PYTHONPATH=D:\r2ai\scripts
cd /d D:\r2ai
echo [%date% %time%] start qwen_bench > work\qrr06\bench.log
start "" /low /wait cmd /c "D:\r2ai\venv_qwen\Scripts\python.exe D:\r2ai\work\qrr06\qwen_rerank_run.py --docs D:\r2ai\work\docwin\chunkdocs100.part000.jsonl.zst D:\r2ai\work\docwin\chunkdocs100.part001.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\docwin\cands_chunkdocs100.json --out D:\r2ai\work\qrr06\bench_out.json --top 100 --max-len 1024 --batch 8 --limit-queries 50 --resume-dir D:\r2ai\work\qrr06\parts >> work\qrr06\bench.log 2>&1"
echo [%date% %time%] exit done errorlevel %errorlevel% >> work\qrr06\bench.log
