@echo off
set PY=D:\r2ai\.venv\Scripts\python.exe
set LOG=D:\r2ai\data\sub\diag24\chunk_len_stats.log
echo START %DATE% %TIME% > %LOG%
%PY% D:\r2ai\scripts\chunk_len_stats.py D:\r2ai\data\sub\diag24\submission_vi24.json --out D:\r2ai\data\sub\diag24\chunk_len_stats.json --tokenizer ws --docs D:\r2ai\data\sub\diag24\docs_vi24_docwin.jsonl.zst D:\r2ai\data\sub\diag24\docs_vi24_old.part000.jsonl.zst D:\r2ai\data\sub\diag18\docwin_p3\docwin_p3.part000.jsonl.zst D:\r2ai\data\sub\diag18\docwin_p3\docwin_p3.part001.jsonl.zst D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part000.jsonl.zst D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part001.jsonl.zst D:\r2ai\data\sub\chunkdocs\chunkdocs.part000.jsonl.zst D:\r2ai\data\sub\chunkdocs50\chunkdocs50.part000.jsonl.zst --index D:\r2ai\data\index\vi24 --lock D:\r2ai\locks\heavy_job.lock >> %LOG% 2>&1
echo EXIT %ERRORLEVEL% %DATE% %TIME% >> %LOG%
