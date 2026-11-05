@echo off
rem Xuat text ~112k doc trong top-200 cua cache BM25 VI de embed tren z87. Log: D:\r2ai\data\sub\export_docs.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\export_docs.py --index D:\r2ai\data\index\vi --cache D:\r2ai\data\sub\cache_vi_k300.json --n 200 --chars 3000 --out-docs D:\r2ai\data\sub\docs_top200.jsonl.zst --out-cands D:\r2ai\data\sub\cands_top200.json > D:\r2ai\data\sub\export_docs.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\export_docs.log
