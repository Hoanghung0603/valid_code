@echo off
rem Chunk-rerank windowed day du 1.200 cau x top-50 doc (bge-reranker-v2-m3, max-len 512, batch 16, fp16) tren z87
rem (uu tien thap, resume theo phan 50 cau). Moi chunk 600 tu cat thanh cua so 300 tu buoc 250;
rem diem chunk = max logit cua so; chunk_text nop van la chunk day du 600 tu.
rem Doc text = text sach cat 200.000 ky tu trong chunkdocs50.partNNN (export_docs.py --no-title --n 50).
rem Log: D:\r2ai\work\chunkrerank_w.log  Out: D:\r2ai\work\chunk_rerank_w_top50.json (.zst neu >90MB) + .wfirst.json
rem Resume: D:\r2ai\work\crw_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start chunkrerank_w > work\chunkrerank_w.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\chunk_rerank.py --docs D:\r2ai\work\chunkdocs50\chunkdocs50.part000.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\chunkdocs50\cands_chunkdocs50.json --out D:\r2ai\work\chunk_rerank_w_top50.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 50 --top-chunks 40 --chunk-units 600 --window-words 300 --window-stride 250 --resume-dir D:\r2ai\work\crw_parts >> work\chunkrerank_w.log 2>&1"
echo [%date% %time%] DONE >> work\chunkrerank_w.log
