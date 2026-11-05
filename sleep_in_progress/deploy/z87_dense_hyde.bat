@echo off
rem Dense top-200 toan kho (1200 x 613.087 doc) voi truy van HyDE = cau goc + doan sinh, query-len 512,
rem embedding doc DA CO (work\emb + work\emb_all), truy van nhung bge-m3 (GPU). Giong z87_dense_topk.bat.
rem Log: D:\r2ai\work\hyde\dense_hyde.log ; out: D:\r2ai\work\hyde\dense_hyde_top200.json
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start dense_hyde > D:\r2ai\work\hyde\dense_hyde.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dense_topk.py --emb D:\r2ai\work\emb D:\r2ai\work\emb_all --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\hyde\q_hyde.jsonl --out D:\r2ai\work\hyde\dense_hyde_top200.json --model BAAI/bge-m3 --query-len 512 --top 200 --chunk 131072 --device cuda >> D:\r2ai\work\hyde\dense_hyde.log 2>&1"
echo [%date% %time%] DONE >> D:\r2ai\work\hyde\dense_hyde.log
