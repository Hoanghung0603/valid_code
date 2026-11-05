@echo off
rem Benchmark rerank bge-reranker-v2-m3 (20 cau x 200 cap = 4.000 cap) batch 16 roi batch 32 tren z87 (uu tien thap). Log: D:\r2ai\work\rerank_bench.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start rerank_bench > work\rerank_bench.log
echo [%date% %time%] === batch 16 === >> work\rerank_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\docs_top200.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\cands_top200.json --out D:\r2ai\work\rerank_bench.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 200 --limit-queries 20 --resume-dir D:\r2ai\work\rr_bench_parts >> D:\r2ai\work\rerank_bench.log 2>&1"
echo [%date% %time%] === batch 32 === >> work\rerank_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\docs_top200.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\cands_top200.json --out D:\r2ai\work\rerank_bench_b32.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 32 --top 200 --limit-queries 20 --resume-dir D:\r2ai\work\rr_bench_b32_parts >> D:\r2ai\work\rerank_bench.log 2>&1"
echo [%date% %time%] DONE >> work\rerank_bench.log
