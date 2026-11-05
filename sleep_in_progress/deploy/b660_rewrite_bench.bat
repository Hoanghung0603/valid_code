@echo off
rem Bench 20 cau hyde + reason (do toc do, xem mau) truoc khi chay day du 1200 cau.
rem Script + log + out deu o D:\r2ai\data\llm\; lock D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\data\llm\rewrite_queries.py
set Q=D:\r2ai\data\llm\zh_queries.jsonl
echo r2ai_rewrite_bench > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > D:\r2ai\data\llm\bench_run.log
%P% %S% --queries %Q% --out D:\r2ai\data\llm\bench_hyde.jsonl --mode hyde --limit 20 --workers 4 --model qwen35-9b > D:\r2ai\data\llm\bench_hyde.log 2>&1
%P% %S% --queries %Q% --out D:\r2ai\data\llm\bench_reason.jsonl --mode reason --limit 20 --workers 4 --model qwen35-9b > D:\r2ai\data\llm\bench_reason.log 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> D:\r2ai\data\llm\bench_run.log
