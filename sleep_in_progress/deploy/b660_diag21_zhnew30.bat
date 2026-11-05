@echo off
rem Bai tham do submission_zhnew30: 30 doc ZH dau theo RRF(VI,ZH) tren tap ZH moi, chunk = toan van doc
rem (giong #5002/b660_zh_top30.bat), dong goi zip LZMA. KHONG nop bai.
rem Can: %W%\zhnew_{rrf,vi,zh}_top100.json + new_zh_ids.json (tu z87 qua local).
rem Out: %W%\submission_zhnew30.{json,zip} + zhnew_report.json. Log: %W%\make_zhnew30.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag21
set L=%W%\make_zhnew30.log
if not exist %W% mkdir %W%
if exist D:\r2ai\locks\heavy_job.lock (
  echo exit=locked >> %L%
  exit /b 1
)
echo zhnew30 %date% %time% > D:\r2ai\locks\heavy_job.lock
echo === start %date% %time% > %L%
%P% %S%\zhnew_report.py --rrf %W%\zhnew_rrf_top100.json --ids-npy D:\r2ai\data\index\zh\ids.npy --out-cache %W%\zhnew_rrf_idx.json --out %W%\zhnew_fixidx.json >> %L% 2>&1
%P% %S%\export_docs.py --index D:\r2ai\data\index\zh --cache %W%\zhnew_rrf_idx.json --n 30 --chars 200000 --no-title --out-docs %W%\chunkdocs_zhnew30.jsonl.zst --out-cands %W%\cands_zhnew30.json >> %L% 2>&1
%P% %S%\chunks_wholedoc.py --docs %W%\chunkdocs_zhnew30.jsonl.zst --cands %W%\zhnew_rrf_idx.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_zhnew30 >> %L% 2>&1
%P% %S%\union_sub.py --docs-from %W%\zhnew_rrf_idx.json --chunks-from %W%\chunks_zhnew30_rr_b10000k.json --out %W%\submission_zhnew30.json --docs-n 30 >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_zhnew30.json >> %L% 2>&1
echo validate_exit=%ERRORLEVEL% >> %L%
%P% %S%\zip_submission.py --json %W%\submission_zhnew30.json --zip %W%\submission_zhnew30.zip --method lzma >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_zhnew30.json >> %L% 2>&1
%P% %S%\zhnew_report.py --sub %W%\submission_zhnew30.json --allowed-zh %W%\new_zh_ids.json --rrf %W%\zhnew_rrf_top100.json --vi %W%\zhnew_vi_top100.json --zh %W%\zhnew_zh_top100.json --old5002 D:\r2ai\data\sub\zhprobe\zh_dense_top100.json --out %W%\zhnew_report.json >> %L% 2>&1
echo === exit %errorlevel% %date% %time% >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit done >> %L%
