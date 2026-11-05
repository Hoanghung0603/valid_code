@echo off
rem BM25 chunk 512 am tiet, overlap 50 (gia thuyet pool chunk cua BTC theo mentor 08/10).
rem Toan kho VI 613.087 doc (docs_vi_all.part000-006): cache + best chunk + bai nop chan doan top30
rem subm_bm25c_<tok>_top30 (doi chieu 4872 reranker top-30). Tokenizer: syl + pyvi (bench 2000 doc
rem truoc, bo qua neu uoc tinh > 4h). Log: D:\r2ai\data\sub\bm25c\*.log ; out: D:\r2ai\data\sub\bm25c\
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set OUT=D:\r2ai\data\sub\bm25c
set Q=D:\r2ai\data\query.parquet
set REF=D:\r2ai\data\sub\cache_pool_rr.json
set DOCS=D:\r2ai\data\sub\docs_vi_all.part000.jsonl.zst D:\r2ai\data\sub\docs_vi_all.part001.jsonl.zst D:\r2ai\data\sub\docs_vi_all.part002.jsonl.zst D:\r2ai\data\sub\docs_vi_all.part003.jsonl.zst D:\r2ai\data\sub\docs_vi_all.part004.jsonl.zst D:\r2ai\data\sub\docs_vi_all.part005.jsonl.zst D:\r2ai\data\sub\docs_vi_all.part006.jsonl.zst
if not exist %OUT% mkdir %OUT%
echo start %DATE% %TIME% > %OUT%\run.log

%P% D:\r2ai\scripts\bm25_chunk.py run --docs %DOCS% --queries %Q% --tok syl --outdir %OUT% > %OUT%\bm25c_syl.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %DOCS% --cands %OUT%\cache_bm25c_syl.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %OUT%\chunks_bm25c_syl > %OUT%\make_subs_syl.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %OUT%\cache_bm25c_syl.json --chunks-from %OUT%\chunks_bm25c_syl_rr_b10000k.json --out %OUT%\subm_bm25c_syl_top30.json --docs-n 30 >> %OUT%\make_subs_syl.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %OUT%\subm_bm25c_syl_top30.json >> %OUT%\make_subs_syl.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %OUT%\subm_bm25c_syl_top30.json --zip %OUT%\subm_bm25c_syl_top30.zip --level 9 >> %OUT%\make_subs_syl.log 2>&1

%P% D:\r2ai\scripts\bm25_chunk.py run --docs %DOCS% --queries %Q% --tok pyvi --outdir %OUT% > %OUT%\bm25c_pyvi.log 2>&1
if exist %OUT%\cache_bm25c_pyvi.json (
  %P% D:\r2ai\scripts\chunks_wholedoc.py --docs %DOCS% --cands %OUT%\cache_bm25c_pyvi.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %OUT%\chunks_bm25c_pyvi > %OUT%\make_subs_pyvi.log 2>&1
  %P% D:\r2ai\scripts\union_sub.py --docs-from %OUT%\cache_bm25c_pyvi.json --chunks-from %OUT%\chunks_bm25c_pyvi_rr_b10000k.json --out %OUT%\subm_bm25c_pyvi_top30.json --docs-n 30 >> %OUT%\make_subs_pyvi.log 2>&1
  %P% D:\r2ai\scripts\evaluate.py validate %OUT%\subm_bm25c_pyvi_top30.json >> %OUT%\make_subs_pyvi.log 2>&1
  %P% D:\r2ai\scripts\zip_submission.py --json %OUT%\subm_bm25c_pyvi_top30.json --zip %OUT%\subm_bm25c_pyvi_top30.zip --level 9 >> %OUT%\make_subs_pyvi.log 2>&1
)

%P% D:\r2ai\scripts\bm25_chunk.py report --cache %OUT%\cache_bm25c_syl.json --ref %REF% --n 30 --files %OUT%\subm_bm25c_syl_top30.zip > %OUT%\report.log 2>&1
if exist %OUT%\cache_bm25c_pyvi.json %P% D:\r2ai\scripts\bm25_chunk.py report --cache %OUT%\cache_bm25c_pyvi.json --ref %REF% --n 30 --files %OUT%\subm_bm25c_pyvi_top30.zip >> %OUT%\report.log 2>&1
echo exit done %DATE% %TIME% >> %OUT%\run.log
