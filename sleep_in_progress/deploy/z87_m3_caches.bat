@echo off
rem Xuat 3 cache xep hang tu D:\r2ai\work\cms\m3_scores.json (all = dense + 0.3*sparse + colbert):
rem cache_m3_sparse.json, cache_m3_colbert.json, cache_m3_all.json vao D:\r2ai\work\cms\
rem Log: D:\r2ai\work\cms\m3_caches.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start m3 caches > work\cms\m3_caches.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\m3_multi_score.py --export-caches --scores D:\r2ai\work\cms\m3_scores.json --out-dir D:\r2ai\work\cms >> D:\r2ai\work\cms\m3_caches.log 2>&1"
echo [%date% %time%] DONE >> work\cms\m3_caches.log
