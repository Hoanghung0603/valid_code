@echo off
rem Benchmark docwin 20 cau dau (bge-reranker-v2-m3, max-len 512, batch 16, fp16) tren z87
rem (uu tien thap): cua so 300 tu buoc 250 o MUC DOC (top-100 doc cache_pool_rr).
rem Do so cap/s de uoc ETA truoc khi chay day du. Log: D:\r2ai\work\docwin\docwin_bench.log
rem Out: D:\r2ai\work\docwin\docwin_bench.json.zst  Resume: D:\r2ai\work\docwin\dw_bench_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start docwin_bench > work\docwin\docwin_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\docwin_run.py --docs D:\r2ai\work\docwin\chunkdocs100.part000.jsonl.zst D:\r2ai\work\docwin\chunkdocs100.part001.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\docwin\cands_chunkdocs100.json --out D:\r2ai\work\docwin\docwin_bench.json.zst --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 100 --window-words 300 --window-stride 250 --limit-queries 20 --resume-dir D:\r2ai\work\docwin\dw_bench_parts >> work\docwin\docwin_bench.log 2>&1"
echo [%date% %time%] DONE >> work\docwin\docwin_bench.log
