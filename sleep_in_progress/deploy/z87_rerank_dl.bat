@echo off
rem Tai BAAI/bge-reranker-v2-m3 (safetensors) vao HF cache D:\r2ai\hf tren z87 (uu tien thap). Log: D:\r2ai\work\rerank_dl.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
cd /d D:\r2ai
echo [%date% %time%] start rerank_dl > work\rerank_dl.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_dl.py >> D:\r2ai\work\rerank_dl.log 2>&1"
echo [%date% %time%] DONE >> work\rerank_dl.log
