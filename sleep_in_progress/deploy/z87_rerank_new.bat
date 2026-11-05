@echo off
rem Rerank 186.325 cap MOI (dense-full \ BM25 pool, data/cands_new.json) bang bge-reranker-v2-m3 tren z87
rem (uu tien thap, resume theo phan). Giong cau hinh z87_rerank.bat: fp16, max-len 512, batch 16.
rem Doc text = text sach da cat 3.000 ky tu trong docs_vi_all.part000..006 (export_docs.py).
rem Log: D:\r2ai\work\rerank_new.log  Out: D:\r2ai\work\rerank_new.json  Resume: D:\r2ai\work\rr_new_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start rerank_new > work\rerank_new.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\cands_new.json --out D:\r2ai\work\rerank_new.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 200 --resume-dir D:\r2ai\work\rr_new_parts >> D:\r2ai\work\rerank_new.log 2>&1"
echo [%date% %time%] DONE >> work\rerank_new.log
