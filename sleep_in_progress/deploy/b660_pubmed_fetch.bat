@echo off
rem Fetch text PubMed (NCBI ctxp API format=ris, batch 100 PMID, toi da 2 req/s, resume theo part) + export docs cho embed.
rem (efetch E-utilities bi chan IP misuse ca 2 duong ra; ctxp van la API NCBI chinh thuc - user duyet 16/10.)
rem Log: D:\r2ai\data\pubmed\fetch.log ; out: D:\r2ai\data\pubmed\pubmed_text.jsonl.zst + docs\docs_pm.partNNN.jsonl.zst
rem Viec mang nhe, khong lay heavy_job.lock. Chay lai duoc (part da co thi bo qua).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\pubmed
if not exist %W% mkdir %W%
echo start %DATE% %TIME% > %W%\fetch.log
%P% %S%\pubmed_fetch.py run --links D:\r2ai\data\sub\diag19\links_corpus_vi_en_zh.parquet --outdir %W% --backend ctxp >> %W%\fetch.log 2>&1
if errorlevel 1 goto fail
%P% %S%\pubmed_docs.py --texts %W%\pubmed_text.jsonl.zst --outdir %W%\docs --part-size 40000 >> %W%\fetch.log 2>&1
if errorlevel 1 goto fail
echo end %DATE% %TIME% >> %W%\fetch.log
echo exit done >> %W%\fetch.log
goto :eof

:fail
echo PUBMED FETCH FAILED >> %W%\fetch.log
echo exit fail >> %W%\fetch.log
goto :eof
