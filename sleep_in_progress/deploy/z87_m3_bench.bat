@echo off
rem Bench m3 (colbert+sparse+dense bge-m3) 20 cau dau: throughput doc/s, VRAM, sanity dense moi vs dense_full_top200.json.
rem Doc max_len 512, query max_len 256, fp16, batch 16. Log: D:\r2ai\work\cms\m3_bench.log
rem Out: D:\r2ai\work\cms\m3_bench_scores.json  Parts: D:\r2ai\work\cms\m3_bench_parts  Status: D:\r2ai\work\cms\m3_bench.status
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start m3 bench > work\cms\m3_bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\m3_multi_score.py --pool D:\r2ai\work\cms\cache_pool_rr.json --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --model D:\r2ai\hf\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181 --out D:\r2ai\work\cms\m3_bench_scores.json --parts-dir D:\r2ai\work\cms\m3_bench_parts --limit-queries 20 --bench-vs D:\r2ai\work\dense_full_top200.json --status D:\r2ai\work\cms\m3_bench.status --doc-len 512 --query-len 256 --batch 16 --chunk-docs 4096 >> D:\r2ai\work\cms\m3_bench.log 2>&1"
echo [%date% %time%] DONE >> work\cms\m3_bench.log
