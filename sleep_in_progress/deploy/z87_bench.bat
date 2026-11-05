@echo off
rem Benchmark bge-m3 tren z87 (GPU only, uu tien thap). Log: D:\r2ai\work\bench.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
echo [%date% %time%] start bench > work\bench.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\embed_bench.py --docs D:\r2ai\work\bench_docs.jsonl --max-len 512 --batch 16 --limit 1000 >> D:\r2ai\work\bench.log 2>&1"
echo [%date% %time%] DONE >> work\bench.log
