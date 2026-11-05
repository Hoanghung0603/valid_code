@echo off
rem Rerank cac cap MOI cua pool2 (dense top-400/300 u BM25 top-300 tru pool cu, data\cands_pool2.json)
rem bang bge-reranker-v2-m3 tren z87 (uu tien thap, resume-dir MOI). Giong cau hinh z87_rerank_new.bat:
rem fp16, max-len 512, batch 16. Doc text = sach da cat 3.000 ky tu trong docs_vi_all.part000..006.
rem Log: D:\r2ai\work\rerank_pool2.log  Out: D:\r2ai\work\rerank_new2.json  Resume: D:\r2ai\work\rr_pool2_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start rerank_pool2 > work\rerank_pool2.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\cands_pool2.json --out D:\r2ai\work\rerank_new2.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 1000 --resume-dir D:\r2ai\work\rr_pool2_parts >> D:\r2ai\work\rerank_pool2.log 2>&1"
echo [%date% %time%] DONE >> work\rerank_pool2.log
