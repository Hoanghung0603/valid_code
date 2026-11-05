@echo off
rem chunks_from_docwin tren b660: tu docwin_top100 + chunkdocs100, tao chunk_dw_u{600,900,1200}
rem (top-120 chunk/cau) o D:\r2ai\data\sub\docwin\. Sanity U=600 top-15 (doc_rank<=50) voi
rem chunk_rerank_w_top50. Log: D:\r2ai\data\sub\docwin\chunks_from_docwin.log. Lock heavy_job.lock giu nguyen.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\chunks_from_docwin.py --docwin D:\r2ai\data\sub\docwin\docwin_top100.json.zst --docs D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part000.jsonl.zst D:\r2ai\data\sub\chunkdocs100\chunkdocs100.part001.jsonl.zst --cands D:\r2ai\data\sub\chunkdocs100\cands_chunkdocs100.json --units 600 900 1200 --top-chunks 120 --out-dir D:\r2ai\data\sub\docwin --sanity-ref D:\r2ai\data\sub\chunk_rerank_w_top50.json.zst --sanity-k 15 --sanity-rank 50 > D:\r2ai\data\sub\docwin\chunks_from_docwin.log 2>&1
echo exit=%ERRORLEVEL% >> D:\r2ai\data\sub\docwin\chunks_from_docwin.log
