@echo off
rem Xuat text day du (KHONG title, cat 200.000 ky tu) top-100 doc cua moi cau trong cache_pool_rr
rem de docwin (cua so 300/250 o MUC DOC) tren z87. Out: D:\r2ai\data\sub\chunkdocs100\chunkdocs100.jsonl.zst (partNNN) + cands_chunkdocs100.json
rem Log: D:\r2ai\data\sub\chunkdocs100\export_chunkdocs100.log. Lock heavy_job.lock do coordinator quan ly qua SSH.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\export_docs.py --index D:\r2ai\data\index\vi --cache D:\r2ai\data\sub\cache_pool_rr.json --n 100 --chars 200000 --no-title --part-mb 80 --out-docs D:\r2ai\data\sub\chunkdocs100\chunkdocs100.jsonl.zst --out-cands D:\r2ai\data\sub\chunkdocs100\cands_chunkdocs100.json > D:\r2ai\data\sub\chunkdocs100\export_chunkdocs100.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\chunkdocs100\export_chunkdocs100.log
