@echo off
rem Benchmark chunk-rerank 20 cau dau (bge-reranker-v2-m3, max-len 512, batch 16, fp16) tren z87
rem (uu tien thap). Do so cap/s de uoc ETA truoc khi chay day du.
rem Log: D:\r2ai\work\chunkrerank_bench.log  Out: D:\r2ai\work\chunkrerank_bench.json
rem Resume: D:\r2ai\work\cr_bench_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start chunkrerank_bench > work\chunkrerank_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\chunk_rerank.py --docs D:\r2ai\work\chunkdocs\chunkdocs.part000.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\chunkdocs\cands_chunkdocs.json --out D:\r2ai\work\chunkrerank_bench.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 40 --top-chunks 40 --chunk-units 600 --limit-queries 20 --resume-dir D:\r2ai\work\cr_bench_parts >> D:\r2ai\work\chunkrerank_bench.log 2>&1"
echo [%date% %time%] DONE >> work\chunkrerank_bench.log
