@echo off
rem vi23 buoc 3 (b660, CPU): BM25 top-300 tren chi muc gop vi23 cho 1.200 cau; do chenh lech top-200 voi
rem cache_vi_k300 (chi tham khao); xuat text doc MOI (title + sach 3.000 ky tu) cho embed/rerank tren z87;
rem lay mau cap da co logit lam sanity rerank. Log: D:\r2ai\data\sub\diag23\bm25.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set L=%W%\bm25.log

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_bm25" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_bm25 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === search_cache vi23 top-300 >> %L%
%P% %S%\search_cache.py --index D:\r2ai\data\index\vi23 --queries D:\r2ai\data\query.parquet --out %W%\cache_vi23_k300.json -k 300 >> %L% 2>&1
if errorlevel 1 goto fail
echo === bm25-diff >> %L%
%P% %S%\vi23.py bm25-diff --cache-old D:\r2ai\data\sub\cache_vi_k300.json --cache-new %W%\cache_vi23_k300.json --new-ids %W%\new_ids.json --top 200 --report %W%\bm25_diff.json >> %L% 2>&1
echo === export text doc moi (embed) >> %L%
%P% %S%\vi23.py export-new --shards %W%\shards --bp D:\r2ai\data\index\vi23\boilerplate.json --idx %W%\new_idx.json --mode embed --out %W%\docs_vi23_embed.jsonl.zst --report %W%\export_embed_report.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === sanity-pairs >> %L%
%P% %S%\vi23.py sanity-pairs --pool D:\r2ai\data\sub\cache_pool_rr.json --n-q 20 --per 10 --out-pairs %W%\sanity_pairs.json --out-ref %W%\sanity_ref.json --report %W%\sanity_pairs_report.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\docs_vi23_embed.jsonl.zst SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_BM25 FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
