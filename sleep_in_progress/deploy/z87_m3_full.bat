@echo off
rem Cham lai day du 1.200 cau x pool (~355 doc/cau, 164.648 doc duy nhat) bang bge-m3:
rem colbert MaxSim + sparse + dense CLS, doc max_len 512, query 256, fp16, batch 16.
rem Log: D:\r2ai\work\cms\m3_full.log  Out: D:\r2ai\work\cms\m3_scores.json
rem Parts resume: D:\r2ai\work\cms\m3_parts  Status: D:\r2ai\work\cms\m3_full.status
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start m3 full > work\cms\m3_full.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\m3_multi_score.py --pool D:\r2ai\work\cms\cache_pool_rr.json --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --model D:\r2ai\hf\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181 --out D:\r2ai\work\cms\m3_scores.json --parts-dir D:\r2ai\work\cms\m3_parts --status D:\r2ai\work\cms\m3_full.status --doc-len 512 --query-len 256 --batch 16 --chunk-docs 4096 >> D:\r2ai\work\cms\m3_full.log 2>&1"
echo [%date% %time%] DONE >> work\cms\m3_full.log
