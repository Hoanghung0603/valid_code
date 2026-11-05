@echo off
rem Bai chan doan submission_zh_top30: doc = top-30 ZH theo cosine bge-m3 (VI query -> ZH doc),
rem chunk = toan van dung 30 doc do (chunks_wholedoc --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30).
rem Can: %W%\zh_dense_top100.json (tu z87), %W%\dense_full_top200.json (VI, de so sanh), %W%\queries.jsonl.
rem Text ZH: export_docs.py che do cache (KHONG title, cat 200.000 ky tu, giong chunkdocs100) tu chi muc ZH.
rem Tao/xoa lock heavy_job.lock; neu lock dang co thi thoat (exit=locked). Log: %W%\make_zh.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set W=D:\r2ai\data\sub\zhprobe
set L=D:\r2ai\locks\heavy_job.lock
if exist %L% (
  echo exit=locked >> %W%\make_zh.log
  exit /b 1
)
echo zhprobe b660_zh_top30 %date% %time% > %L%
cd /d D:\r2ai\scripts
echo start %date% %time% > %W%\make_zh.log
%P% D:\r2ai\scripts\export_docs.py --index D:\r2ai\data\index\zh --cache %W%\zh_dense_top100.json --n 30 --chars 200000 --no-title --out-docs %W%\chunkdocs_zh30.jsonl.zst --out-cands %W%\cands_zh30.json >> %W%\make_zh.log 2>&1
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %W%\chunkdocs_zh30.jsonl.zst --cands %W%\zh_dense_top100.json --orders rr --budgets 10000000 --doc-cap 40000 --top-docs 30 --out-prefix %W%\chunks_zh30 >> %W%\make_zh.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from %W%\zh_dense_top100.json --chunks-from %W%\chunks_zh30_rr_b10000k.json --out %W%\submission_zh_top30.json --docs-n 30 >> %W%\make_zh.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %W%\submission_zh_top30.json >> %W%\make_zh.log 2>&1
echo validate_exit=%ERRORLEVEL% >> %W%\make_zh.log
%P% D:\r2ai\scripts\zip_submission.py --json %W%\submission_zh_top30.json --zip %W%\submission_zh_top30.zip --level 9 >> %W%\make_zh.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %W%\submission_zh_top30.json >> %W%\make_zh.log 2>&1
%P% D:\r2ai\scripts\zh_probe_stats.py --zh %W%\zh_dense_top100.json --vi %W%\dense_full_top200.json --queries %W%\queries.jsonl --index D:\r2ai\data\index\zh --k 30 --examples 3 --ex-docs 3 --out %W%\zh_probe_stats.json > %W%\stats_zh.log 2>&1
echo exit done %date% %time% >> %W%\make_zh.log
del %L%
