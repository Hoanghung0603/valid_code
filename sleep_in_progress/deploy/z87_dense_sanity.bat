@echo off
rem Sanity: chay lai dense top-200 voi truy van GOC (work\queries.jsonl), query-len 256 (giong dense_full_top200),
rem doi chieu top-30 voi D:\r2ai\work\dense_full_top200.json de chac cau hinh dung.
rem Log: D:\r2ai\work\hyde\dense_sanity.log ; out: D:\r2ai\work\hyde\dense_sanity_top200.json
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start dense_sanity > D:\r2ai\work\hyde\dense_sanity.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dense_topk.py --emb D:\r2ai\work\emb D:\r2ai\work\emb_all --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --out D:\r2ai\work\hyde\dense_sanity_top200.json --model BAAI/bge-m3 --query-len 256 --top 200 --chunk 131072 --device cuda >> D:\r2ai\work\hyde\dense_sanity.log 2>&1"
echo [%date% %time%] DONE >> D:\r2ai\work\hyde\dense_sanity.log
