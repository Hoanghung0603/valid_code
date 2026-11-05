@echo off
rem Top-400 cosine toan kho (1.200 cau x 613.087 doc) tu embedding DA CO trong work\emb + work\emb_all,
rem KHONG embed lai doc. idx doc lay tu docs_vi_all.part000..006; truy van nhung bge-m3 (GPU).
rem Giong cau hinh z87_embed_all.bat: fp16, doc-len/query-len 256. Log: D:\r2ai\work\dense_topk.log
rem Out: D:\r2ai\work\dense_full_top400.json
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start dense_topk > work\dense_topk.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dense_topk.py --emb D:\r2ai\work\emb D:\r2ai\work\emb_all --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --out D:\r2ai\work\dense_full_top400.json --model BAAI/bge-m3 --query-len 256 --top 400 --chunk 131072 --device cuda >> D:\r2ai\work\dense_topk.log 2>&1"
echo [%date% %time%] DONE >> work\dense_topk.log
