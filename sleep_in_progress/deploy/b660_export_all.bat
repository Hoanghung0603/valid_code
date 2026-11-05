@echo off
rem Xuat text 613.087 doc cua chi muc VI (toan kho) de embed tren z87. Log: D:\r2ai\data\sub\export_all.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\export_docs.py --all --index D:\r2ai\data\index\vi --chars 3000 --out-docs D:\r2ai\data\sub\docs_vi_all.jsonl.zst --part-mb 80 > D:\r2ai\data\sub\export_all.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\export_all.log
