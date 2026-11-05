@echo off
rem Tim top-300 doc VI cho 1200 cau MOT lan, luu cache de dung lai cho cac bien the bai nop (khong phai tim lai ~20 phut).
rem In thoi gian va so posting/cau moi 100 cau de biet cham vi truy van nang hay vi doc dia.
rem Log: D:\r2ai\data\sub\search_cache_vi.log, ket qua: D:\r2ai\data\sub\cache_vi_k300.json
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist D:\r2ai\data\sub mkdir D:\r2ai\data\sub
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\search_cache.py --index D:\r2ai\data\index\vi --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\sub\cache_vi_k300.json -k 300 > D:\r2ai\data\sub\search_cache_vi.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\search_cache_vi.log
