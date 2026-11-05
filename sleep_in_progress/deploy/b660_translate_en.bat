@echo off
rem Dich 1200 truy van VI -> EN (Anh y khoa) bang Qwen3.5 9B (LM Studio cong 1234, tac vu r2ai_llm dang giu server).
rem Log: D:\r2ai\data\llm\translate_en.log ; out: D:\r2ai\data\llm\en_queries.jsonl. Chay lai duoc.
rem Heavy lock (LM Studio ~10GB RAM); doi lock cua job khac. Xong thi lms unload --all de tra RAM.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set L=D:\r2ai\data\llm\translate_en.log
if not exist D:\r2ai\data\llm mkdir D:\r2ai\data\llm
:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_tr_en" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_tr_en > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
%P% D:\r2ai\scripts\translate_queries.py --queries D:\r2ai\data\query.parquet --out D:\r2ai\data\llm\en_queries.jsonl --target en --workers 3 >> %L% 2>&1
set RC=%errorlevel%
%USERPROFILE%\.lmstudio\bin\lms.exe unload --all >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% rc=%RC% >> %L%
if not %RC%==0 goto fail
echo exit done >> %L%
goto :eof

:fail
echo TRANSLATE EN FAILED >> %L%
echo exit fail >> %L%
goto :eof
