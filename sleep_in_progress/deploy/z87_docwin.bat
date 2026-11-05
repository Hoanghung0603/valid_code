@echo off
rem Docwin day du 1.200 cau x top-100 doc (bge-reranker-v2-m3, max-len 512, batch 16, fp16) tren z87
rem (uu tien thap, resume theo phan 50 cau). Moi doc cat thanh cua so 300 tu buoc 250 o MUC DOC;
rem luu [start_char, end_char, logit] cho moi (qid, doc_id). Doc text = text sach cat 200.000 ky tu
rem trong chunkdocs100.partNNN (export_docs.py --no-title --n 100).
rem Log: D:\r2ai\work\docwin\docwin.log  Out: D:\r2ai\work\docwin\docwin_top100.json.zst
rem Resume: D:\r2ai\work\docwin\dw_parts
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start docwin > work\docwin\docwin.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\docwin_run.py --docs D:\r2ai\work\docwin\chunkdocs100.part000.jsonl.zst D:\r2ai\work\docwin\chunkdocs100.part001.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\docwin\cands_chunkdocs100.json --out D:\r2ai\work\docwin\docwin_top100.json.zst --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 100 --window-words 300 --window-stride 250 --resume-dir D:\r2ai\work\docwin\dw_parts >> work\docwin\docwin.log 2>&1"
echo [%date% %time%] DONE >> work\docwin\docwin.log
