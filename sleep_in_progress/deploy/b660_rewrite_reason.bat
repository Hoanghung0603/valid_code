@echo off
rem Sinh Reason (suy luan ngan 40-90 tu) cho 1200 cau hoi y khoa VI bang Qwen3.5 9B (LM Studio cong 1234).
rem Chay lai duoc theo id. Log: D:\r2ai\data\llm\reason.log ; out: D:\r2ai\data\llm\reason.jsonl
rem Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
echo r2ai_rewrite_reason > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > D:\r2ai\data\llm\reason_run.log
%P% D:\r2ai\data\llm\rewrite_queries.py --queries D:\r2ai\data\llm\zh_queries.jsonl --out D:\r2ai\data\llm\reason.jsonl --mode reason --workers 4 --model qwen35-9b > D:\r2ai\data\llm\reason.log 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> D:\r2ai\data\llm\reason_run.log
