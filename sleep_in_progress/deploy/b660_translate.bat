@echo off
rem Dich 1200 truy van VI -> ZH bang Qwen3.5 9B (LM Studio, cong 1234, phai dang chay: tac vu r2ai_llm). Chay lai duoc.
rem Log: D:\r2ai\data\llm\translate.log, ket qua: D:\r2ai\data\llm\zh_queries.jsonl
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\translate_queries.py --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\llm\zh_queries.jsonl --workers 3 > D:\r2ai\data\llm\translate.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\llm\translate.log
