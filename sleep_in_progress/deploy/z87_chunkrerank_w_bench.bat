@echo off
rem Benchmark chunk-rerank windowed 20 cau dau (bge-reranker-v2-m3, max-len 512, batch 16, fp16) tren z87
rem (uu tien thap): moi chunk 600 tu cat thanh cua so 300 tu buoc 250, diem chunk = max logit cua so.
rem Do so cap/s de uoc ETA truoc khi chay day du (top-docs 50).
rem Log: D:\r2ai\work\chunkrerank_w_bench.log  Out: D:\r2ai\work\chunkrerank_w_bench.json (+ .wfirst.json)
rem Resume: D:\r2ai\work\crw_bench_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start chunkrerank_w_bench > work\chunkrerank_w_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\chunk_rerank.py --docs D:\r2ai\work\chunkdocs50\chunkdocs50.part000.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\chunkdocs50\cands_chunkdocs50.json --out D:\r2ai\work\chunkrerank_w_bench.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 50 --top-chunks 40 --chunk-units 600 --window-words 300 --window-stride 250 --limit-queries 20 --resume-dir D:\r2ai\work\crw_bench_parts >> work\chunkrerank_w_bench.log 2>&1"
echo [%date% %time%] DONE >> work\chunkrerank_w_bench.log
