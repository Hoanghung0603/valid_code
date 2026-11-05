@echo off
rem Sinh HyDE (doan tra loi gia dinh 80-150 tu) cho 1200 cau hoi y khoa VI bang Qwen3.5 9B (LM Studio cong 1234).
rem Chay lai duoc theo id. Log: D:\r2ai\data\llm\hyde.log ; out: D:\r2ai\data\llm\hyde.jsonl
rem Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
echo r2ai_rewrite_hyde > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > D:\r2ai\data\llm\hyde_run.log
%P% D:\r2ai\data\llm\rewrite_queries.py --queries D:\r2ai\data\llm\zh_queries.jsonl --out D:\r2ai\data\llm\hyde.jsonl --mode hyde --workers 4 --model qwen35-9b > D:\r2ai\data\llm\hyde.log 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> D:\r2ai\data\llm\hyde_run.log
