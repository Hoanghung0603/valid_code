@echo off
rem Chunk-rerank day du 1.200 cau x top-40 doc (bge-reranker-v2-m3, max-len 512, batch 16, fp16) tren z87
rem (uu tien thap, resume theo phan 50 cau). Giong cau hinh z87_rerank_new.bat.
rem Doc text = text sach cat 200.000 ky tu trong chunkdocs.part000..006 (export_docs.py --no-title).
rem Log: D:\r2ai\work\chunkrerank.log  Out: D:\r2ai\work\chunk_rerank_top40.json (.zst neu >90MB)
rem Resume: D:\r2ai\work\cr_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start chunkrerank > work\chunkrerank.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\chunk_rerank.py --docs D:\r2ai\work\chunkdocs\chunkdocs.part000.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\chunkdocs\cands_chunkdocs.json --out D:\r2ai\work\chunk_rerank_top40.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 40 --top-chunks 40 --chunk-units 600 --resume-dir D:\r2ai\work\cr_parts >> D:\r2ai\work\chunkrerank.log 2>&1"
echo [%date% %time%] DONE >> work\chunkrerank.log
