@echo off
rem Xuat text day du (KHONG title, cat 200.000 ky tu) top-50 doc cua moi cau trong cache_pool_rr
rem de chunk-rerank windowed tren z87. Out: D:\r2ai\data\sub\chunkdocs50\chunkdocs50.jsonl.zst (partNNN) + cands_chunkdocs50.json
rem Log: D:\r2ai\data\sub\chunkdocs50\export_chunkdocs50.log. Lock heavy_job.lock do coordinator quan ly qua SSH.
rem Ban moi cua b660_export_chunkdocs.bat (truoc --n 40); file top-40 cu trong chunkdocs\ giu nguyen.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub\chunkdocs50 mkdir D:\r2ai\data\sub\chunkdocs50
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\export_docs.py --index D:\r2ai\data\index\vi --cache D:\r2ai\data\sub\cache_pool_rr.json --n 50 --chars 200000 --no-title --part-mb 80 --out-docs D:\r2ai\data\sub\chunkdocs50\chunkdocs50.jsonl.zst --out-cands D:\r2ai\data\sub\chunkdocs50\cands_chunkdocs50.json > D:\r2ai\data\sub\chunkdocs50\export_chunkdocs50.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\chunkdocs50\export_chunkdocs50.log
